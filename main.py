# -*- coding: utf-8 -*-
import os
import argparse

# import keras
# from keras.layers import *
from tensorflow import keras
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from flyai.framework import FlyAI
from flyai.data_helper import DataHelper
from path import MODEL_PATH, DATA_PATH
import pandas as pd
import numpy as np
from data_helper import load_dict, load_labeldict, get_batches, read_data, get_val_batch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()



class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=True):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions



class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''
    def download_data(self):
        # 下载数据
        data_helper = DataHelper()
        data_helper.download_from_ids("MedicalClass")
        print('=*=数据下载完成=*=')

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        # 加载数据
        self.data = pd.read_csv(os.path.join(DATA_PATH, 'MedicalClass/train.csv'))
        # 划分训练集、测试集
        self.train_data, self.valid_data = train_test_split(self.data, test_size=0.01, random_state=6, shuffle=True)
        self.text2id, _ = load_dict(os.path.join(DATA_PATH, 'MedicalClass/words_fr.dict'))
        # self.text2id, _ = load_dict(self.train_data)
        self.label2id, _ = load_labeldict(os.path.join(DATA_PATH, 'MedicalClass/label.dict'))
        self.train_text, self.train_label = read_data(self.train_data, self.text2id, self.label2id)
        self.val_text, self.val_label = read_data(self.valid_data, self.text2id, self.label2id)
        print('=*=数据处理完成=*=')

    def train(self):
        # rnn_unit_1 = 128    # RNN层包含cell个数
        embed_dim = 64      # 嵌入层大小
        ff_dim = 64
        class_num = len(self.label2id)
        num_word = len(self.text2id)
        batch_size = args.BATCH
        MAX_SQUES_LEN = 68  # 最大句长
        num_heads=8
        text_input = layers.Input(shape=(MAX_SQUES_LEN,), dtype='int32')
        embedding_layer = TokenAndPositionEmbedding(MAX_SQUES_LEN, num_word, embed_dim)
        x = embedding_layer(text_input)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(class_num, activation="softmax")(x)

        k_model = keras.Model(inputs=text_input, outputs=outputs)
        k_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

        # text_input = Input(shape=(MAX_SQUES_LEN,), dtype='int32')
        # embedden_seq = Embedding(input_dim=num_word, output_dim=embed_dim, input_length=MAX_SQUES_LEN)(text_input)
        # BN1 = BatchNormalization()(embedden_seq)
        # bGRU1 = Bidirectional(GRU(rnn_unit_1, activation='selu', return_sequences=True,
        #                           implementation=1), merge_mode='concat')(BN1)
        # bGRU2 = Bidirectional(GRU(rnn_unit_1, activation='selu', return_sequences=True,
        #                           implementation=1), merge_mode='concat')(bGRU1)
        #
        # drop = Dropout(0.5)(bGRU2)
        # avgP = GlobalAveragePooling1D()(drop)
        # maxP = GlobalMaxPooling1D()(drop)
        #
        # conc = concatenate([avgP, maxP])
        #
        # pred = Dense(class_num, activation='softmax')(conc)
        # k_model = keras.Model(text_input, pred)
        # k_model.compile(optimizer=RMSprop(lr=0.0005), loss='categorical_crossentropy', metrics=['acc'])

        batch_nums = int(len(self.train_data)/batch_size)
        best_score = 0
        for epoch in range(args.EPOCHS):
            for batch_i, (x_train, y_train) in \
                    enumerate(get_batches(self.train_text, self.train_label,
                                          batch_size=batch_size, text_padding=self.text2id['_pad_'])):

                history = k_model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, verbose=0)
                CurEpoch = str(epoch+1) + "/" + str(args.EPOCHS)
                CurBatch = str(batch_i+1) + "/" + str(batch_nums)
                # print('CurEpoch: {} | CurBatch: {}| ACC: {}'.format(CurEpoch, CurBatch, history.history['acc'][0]))

                x_val, y_val = get_val_batch(self.val_text, self.val_label,
                                             batch_size=1024, text_padding=self.text2id['_pad_'])
                if batch_i % 100 == 0:
                    score = k_model.evaluate(np.array(x_val), np.array(y_val), batch_size=1024)
                    acc = score[1]
                    if acc > best_score:
                        best_score = acc
                        print(f'new acc {acc}')
                        # k_model.save(os.path.join(MODEL_PATH, 'model.h5'))
                    print('best acc:', best_score)







if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()

    exit(0)