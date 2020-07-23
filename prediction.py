# -*- coding: utf-8 -*
import os
import numpy as np
from flyai.framework import FlyAI
from path import MODEL_PATH, DATA_PATH
from keras.models import load_model
from data_helper import pred_process, load_dict, load_labeldict


class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在构造方法中加载模型
        '''
        self.text2id, _ = load_dict(os.path.join(DATA_PATH, 'MedicalClass/words_fr.dict'))
        _, self.id2label = load_labeldict(os.path.join(DATA_PATH, 'MedicalClass/label.dict'))
        self.model = load_model(os.path.join(MODEL_PATH, 'model.h5'))

    def predict(self, title, text):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"title": "心率为72bpm是正常的吗", "text": "最近不知道怎么回事总是感觉心脏不舒服..."}
        :return: 模型预测成功中户 {"label": "心血管科"}
        '''
        text_line = pred_process(title, text, self.text2id, max_len=68)
        pred = self.id2label[np.argmax(self.model.predict(np.array(text_line)))]
        return {'label': pred}