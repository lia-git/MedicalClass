# -*- coding: utf-8 -*- 
# author: Honay.King

import os
import json
import jieba
import numpy as np



def load_dict_1(data):
    # text_data = {}
    token_dict ={}
    for ind, row in data.iterrows():
        text_line = row['title'].strip() +'＃' +row['text'].strip()
        for w in text_line:
            token_dict[w] = token_dict.get(w,0) +1
    tokens = [(w,c) for w,c in token_dict.items()]
    tokens = sorted(tokens,key=lambda i:i[1],reverse=True)[:2700]
    text2id = {'_unk_':0,'_pad_':1}
    id2text = {0:'_unk_',1:'_pad_'}
    for ix, (w,c) in enumerate(tokens):
        text2id[w] = ix +2
        id2text[ix+2] = w

    return text2id,id2text





def load_dict(dictFile):
    if not os.path.exists(dictFile):
        print('[ERROR] load_dict failed! | The params {}'.format(dictFile))
        return None
    with open(dictFile, 'r', encoding='UTF-8') as df:
        dictF = json.load(df)
    tokens = [(w,c) for w,c in dictF.items() if c>10]
    tokens = sorted(tokens,key=lambda i:i[1],reverse=True)
    text2id, id2text = dict(), dict()
    count = 0
    for ix, (key, value) in enumerate(tokens):
        text2id[key] = ix
        id2text[ix] = key
        count += 1
    return text2id, id2text


def load_labeldict(dictFile):
    if not os.path.exists(dictFile):
        print('[ERROR] load_labeldict failed! | The params {}'.format(dictFile))
        return None
    with open(dictFile, 'r', encoding='UTF-8') as df:
        label2id = json.load(df)
    id2label = dict()
    for key, value in label2id.items():
        id2label[value] = key
    return label2id, id2label


def read_data(data, textdict, labeldict):
    text_data, label_data = list(), list()
    for ind, row in data.iterrows():
        # text_line = jieba.lcut(row['title'] + row['text'])
        text_line = row['title'].strip() +'＃' +row['text'].strip()
        tmp_text = list()
        for text in text_line:
            if text in textdict.keys():
                tmp_text.append(textdict[text])
            else:
                tmp_text.append(textdict['_unk_'])
        text_data.append(tmp_text)
        label = np.zeros(len(labeldict), dtype=int)
        label[labeldict[row['label']]] = 1
        label_data.append(label)
    return text_data, label_data


def pred_process(title, text, textdict, max_len=68):
    text_line = jieba.lcut(title+text)
    tmp_text = list()
    for item in text_line:
        if item in textdict.keys():
            tmp_text.append(textdict[item])
        else:
            tmp_text.append(textdict['_unk_'])
        if len(tmp_text) >= max_len:
            tmp_text = tmp_text[:max_len]
        else:
            tmp_text = tmp_text + [textdict['_pad_']] * (max_len - len(tmp_text))
    return [np.array(tmp_text)]


def batch_padding(text_batch, padding, max_len=68):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的text_length
    参数：
    - text_batch
    - padding: <PAD>对应索引号
    '''
    batch_text = list()
    for text in text_batch:
        if len(text) >= max_len:
            batch_text.append(np.array(text[:max_len]))
        else:
            batch_text.append(np.array(text + [padding] * (max_len-len(text))))
    return batch_text


def get_batches(texts, labels, batch_size, text_padding):
    for batch_i in range(0, len(labels) // batch_size):
        start_i = batch_i * batch_size
        texts_batch = texts[start_i: start_i + batch_size]
        labels_batch = labels[start_i: start_i + batch_size]

        pad_texts_batch = batch_padding(texts_batch, text_padding)
        yield pad_texts_batch, labels_batch


def get_val_batch(texts, labels, batch_size, text_padding):
    texts_batch = texts[:batch_size]
    labels_batch = labels[:batch_size]
    pad_texts_batch = batch_padding(texts_batch, text_padding)
    return pad_texts_batch, labels_batch


if __name__ == "__main__":
    from prediction import Prediction

    model = Prediction()
    model.load_model()

    result = model.predict(title='甲状腺功能减退能治好吗？', text='无')
    print(result)

    exit(0)