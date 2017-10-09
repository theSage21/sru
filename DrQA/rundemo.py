import os
import re
import torch
import spacy
import string
import random
import bottle
import msgpack
import unicodedata
import pandas as pd
from collections import Counter
from drqa.model import DocReaderModel


class CONFIG:
    def __init__(self):
        self.wv_file = 'glove/glove.840B.300d.txt'
        self.wv_dim = 300
        self.wv_cased = True
        self.sort_all = True
        self.sample_size = 0
        self.batch_size = 64
        self.model_dir = 'models'
        self.seed = 937
        self.resume = 'best_model.pt'
        self.fix_embeddings = False
        self.data_file = 'SQuAD/data.msgpack'
        self.tune_partial = 1000
        self.question_merge = 'self_attn'
        self.doc_layers = 5
        self.question_layers = 5
        self.hidden_size = 128
        self.num_features = 4
        self.pos = True
        self.reduce_lr = 0.
        self.optimizer = 'adamax'
        self.grad_clipping = 20
        self.weight_decay = 0
        self.learning_rate = 0.001
        self.momentum = 0
        self.pos_size = 56
        self.pos_dim = 56
        self.ner = True
        self.ner_size = 19
        self.ner_dim = 19
        self.use_qemb = True
        self.concat_rnn_layers = False
        self.dropout_emb = 0.5
        self.dropout_rnn = 0.2
        self.dropout_rnn_output = True
        self.max_len = 15
        self.rnn_type = 'lstm'


args = CONFIG()


def main():
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model_dir = os.path.abspath(model_dir)
    seed = args.seed if args.seed >= 0 else int(random.random()*1000)
    random.seed(seed)
    torch.manual_seed(seed)

    train, dev, dev_y, embedding, opt, v, vt, ve = load_data(vars(args))
    checkpoint = torch.load(os.path.join(model_dir, args.resume))
    state_dict = checkpoint['state_dict']

    # MAIN objects
    model = DocReaderModel(opt, embedding, state_dict)
    app = bottle.Bottle()
    nlp = spacy.load('en')

    print('Evaluating loaded model on SQuAD')
    batches = BatchGen(dev, batch_size=1, evaluation=True, gpu=args.cuda)
    predictions = []
    for batch in batches:
        predictions.extend(model.predict(batch))
    em, f1 = score(predictions, dev_y)
    print("[dev EM: {} F1: {}]".format(em, f1))

    @app.route('/<:re:.*>', method=['OPTIONS'])
    def enableCORSGenericOptionsRoute():
        "This allows for CORS usage"
        return 'OK'

    @app.hook('after_request')
    def add_cors_headers():
        bottle.response.headers['Access-Control-Allow-Origin'] = '*'
        key = 'Access-Control-Allow-Methods'
        bottle.response.headers[key] = 'POST, OPTIONS'
        string = 'Origin, Accept, Content-Type,'
        string += ' X-Requested-With, X-CSRF-Token'
        key = 'Access-Control-Allow-Headers'
        bottle.response.headers[key] = string

    @app.get('/')
    def demo_home():
        with open('home.html', 'r') as fl:
            html = fl.read()
        return html

    @app.get('/static/<filename:path>')
    def server_static(filename):
        root = os.path.join(os.getcwd(), 'static')
        print(root, 'is static root')
        return bottle.static_file(filename, root=root)

    @app.post('/get-context')
    def get_context():
        paragraph = bottle.request.json['paragraph']
        question = bottle.request.json['question']
        livedata = live_preprocess(paragraph, question, v, vt, ve, nlp)
        batches = BatchGen(livedata, batch_size=1,
                           evaluation=True, gpu=args.cuda)
        predictions = []
        for batch in batches:
            predictions.extend(model.predict(batch))
        return {'result': predictions[0]}

    app.run(debug=True, port=6006, host='0.0.0.0', reloader=False)


def load_data(opt):
    with open('SQuAD/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    vocab = meta['vocab']
    vocab_tag = meta['vocab_tag']
    vocab_ent = meta['vocab_ent']
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    if not opt['fix_embeddings']:
        means = torch.zeros(opt['embedding_dim'])
        embedding[1] = torch.normal(means=means, std=1.)
    with open(args.data_file, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train_orig = pd.read_csv('SQuAD/train.csv')
    dev_orig = pd.read_csv('SQuAD/dev.csv')
    train = list(zip(
        data['trn_context_ids'],
        data['trn_context_features'],
        data['trn_context_tags'],
        data['trn_context_ents'],
        data['trn_question_ids'],
        train_orig['answer_start_token'].tolist(),
        train_orig['answer_end_token'].tolist(),
        data['trn_context_text'],
        data['trn_context_spans']
    ))
    dev = list(zip(
        data['dev_context_ids'],
        data['dev_context_features'],
        data['dev_context_tags'],
        data['dev_context_ents'],
        data['dev_question_ids'],
        data['dev_context_text'],
        data['dev_context_spans']
    ))
    dev_y = dev_orig['answers'].tolist()[:len(dev)]
    dev_y = [eval(y) for y in dev_y]
    return train, dev, dev_y, embedding, opt, vocab, vocab_tag, vocab_ent


class BatchGen:
    def __init__(self, data, batch_size, gpu, evaluation=False):
        '''
        input:
            data - list of lists
            batch_size - int
        '''
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu

        # shuffle
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        # chunk into batches
        data = [data[i:i + batch_size]
                for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 7
            else:
                assert len(batch) == 9

            context_len = max(len(x) for x in batch[0])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[0]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[1][0][0])
            context_feature = torch.Tensor(batch_size,
                                           context_len,
                                           feature_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                context_tag[i, :len(doc)] = torch.LongTensor(doc)

            context_ent = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[3]):
                context_ent[i, :len(doc)] = torch.LongTensor(doc)
            question_len = max(len(x) for x in batch[4])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[4]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            if not self.eval:
                y_s = torch.LongTensor(batch[5])
                y_e = torch.LongTensor(batch[6])
            text = list(batch[-2])
            span = list(batch[-1])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval:
                yield (context_id, context_feature,
                       context_tag, context_ent,
                       context_mask, question_id,
                       question_mask, text, span)
            else:
                yield (context_id, context_feature,
                       context_tag, context_ent,
                       context_mask, question_id,
                       question_mask, y_s, y_e, text, span)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1


def pre_proc(text):
    '''normalize spaces in a string.'''
    text = re.sub('\s+', ' ', text)
    return text


def token2id(docs, vocab, unk_id=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def live_preprocess(context, question, vocab, vocab_tag, vocab_ent, nlp):
    "Produce live dictionary for running the code"
    questions = [question]
    contexts = [context]

    context_text = [pre_proc(c) for c in contexts]
    question_text = [pre_proc(q) for q in questions]

    question_docs = [nlp(doc) for doc in question_text]
    context_docs = [nlp(doc) for doc in context_text]
    question_tokens = [[normalize_text(w.text) for w in doc]
                       for doc in question_docs]
    context_tokens = [[normalize_text(w.text) for w in doc]
                      for doc in context_docs]
    context_token_span = [[(w.idx, w.idx + len(w.text)) for w in doc]
                          for doc in context_docs]
    context_tags = [[w.tag_ for w in doc] for doc in context_docs]
    context_ents = [[w.ent_type_ for w in doc] for doc in context_docs]
    context_features = []
    for question, context in zip(question_docs, context_docs):
        question_word = {w.text for w in question}
        question_lower = {w.text.lower() for w in question}
        question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()
                          for w in question}
        match_origin = [w.text in question_word for w in context]
        match_lower = [w.text.lower() in question_lower for w in context]
        match_lemma = [((w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower())
                        in question_lemma) for w in context]
        context_features.append(list(zip(match_origin, match_lower,
                                         match_lemma)))
        context_tf = []
        for doc in context_tokens:
            counter_ = Counter(w.lower() for w in doc)
            total = sum(counter_.values())
            context_tf.append([counter_[w.lower()] / total for w in doc])
        context_features = [[list(w) + [tf]
                            for w, tf in zip(doc, tfs)]
                            for doc, tfs in zip(context_features, context_tf)]

    question_ids = token2id(question_tokens, vocab, unk_id=1)
    context_ids = token2id(context_tokens, vocab, unk_id=1)
    context_tag_ids = token2id(context_tags, vocab_tag)
    context_ent_ids = token2id(context_ents, vocab_ent)
    data = {
            'dev_question_ids': question_ids,
            'dev_context_ids': context_ids,
            'dev_context_features': context_features,
            'dev_context_tags': context_tag_ids,
            'dev_context_ents': context_ent_ids,
            'dev_context_text': context_text,
            'dev_context_spans': context_token_span
            }
    dev = list(zip(
        data['dev_context_ids'],
        data['dev_context_features'],
        data['dev_context_tags'],
        data['dev_context_ents'],
        data['dev_question_ids'],
        data['dev_context_text'],
        data['dev_context_spans']
    ))
    return dev


if __name__ == '__main__':
    main()
