# coding: utf-8

import pickle
import numpy as np

from collections import defaultdict
from gensim.models import KeyedVectors
from pyhanlp import *
from scipy.spatial.distance import cosine

from parsertoos import Parsertool
from utils import split_doc
from config import Config


class Extractor:
    def __init__(self):
        self.tokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
        self.word_vectors = KeyedVectors.load_word2vec_format(Config.W2V_MODEL_PATH)
        self.parser = Parsertool()
        self.prob_dict = self.get_word_freq_dict(Config.CORPUS_FREQUENCY_PATH)

    def find_opinions(self, doc, alpha=Config.alpha, expect_min_cont_len=Config.expect_min_cont_len):
        """处理输入的多句话组成的文档。

        doc：新闻篇章， string；
        alpha: 句子相似性的阈值，判断两句话是否是主语说的内容。
        expect_min_cont_len： 设置提取内容的最短长度。
        """
        sent_list = split_doc(doc)
        sent_vec_matrix = self.__cal_sentences_vec_mat(sent_list, param_a=Config.param_a)

        person_opinion_dict = defaultdict(str)
        FIND_SBV_FLAG = False  # 是否找到主谓关系。
        idx = 0
        content_sent_id = 0
        while idx < len(sent_list):
            sentence = sent_list[idx]
            print('Sent: ', sentence)
            if FIND_SBV_FLAG is False:
                parse_res = self.parser.parse(sentence)
                if parse_res in {'No_subject', 'No_verb'}:
                    idx += 1
                    continue

                subject, verb, content = parse_res
                # 分句导致: ['“xxx”', '小明说。']的情况
                if content == '' and (idx - 1) > 0:
                    if sent_list[idx - 1].startswith('“'):
                        content = sent_list[idx - 1][1: -1]
                if content == '' or len(content) <= expect_min_cont_len:
                    print("没有找到言论内容")
                    subject = None
                    verb = None
                    idx += 1
                    continue
                person_opinion_dict[(subject, verb)] += content
                content_sent_id = idx
                FIND_SBV_FLAG = True
                idx += 1


            elif FIND_SBV_FLAG is True:
                # NOTE: cosine_distance = 1 - cosine_similarity
                cosine_distance = cosine(sent_vec_matrix[:, content_sent_id], sent_vec_matrix[:, idx])
                print("========cosine_distance: ", cosine_distance)
                if cosine_distance < 0.2:
                    person_opinion_dict[(subject, verb)] += sentence
                    idx += 1
                else:
                    FIND_SBV_FLAG = False

            # 因为指代混乱而没有parse出正确subject的情况
            if subject == '':
                person_opinion_dict[(subject, verb)] +=  '::Ref source: ' + sent_list[content_sent_id]

        return person_opinion_dict

    def __cal_sentences_vec_mat(self, sent_list, param_a=Config.param_a):
        """计算sentence vector，以此来判断两个句子的相似度。
        来自paper:
        A SIMPLE  BUTTOUGH-TO-BEATBASELINE  FORSEN-TENCEEMBEDDINGS. ICLR2017

        sent_list: 来自待识别文档的分句结果, list；
        prob_dict: 语料库中的token概率值, dict；
        param_a: 论文中实验得到的效果比较好的参数取值, 1e-3 ~ 1e-5, 在skipgram + negetive sampling的情况下，1e-4效果较好；

        return: (vector_dim, sentence_num)形状的matrix，每一列代表sentence的向量
        """
        word_vectors = self.word_vectors
        row_size = word_vectors.vector_size
        col_size = len(sent_list)
        default_p = max(self.prob_dict.values())  # 默认的p，算法计算逆corpus频率的权重
                                             # 因此选用max(prob_dict.values())
        matrix = np.zeros((row_size, col_size))
        for i, sentence in enumerate(sent_list):
            sentence = self.tokenizer.segment(sentence)
            
            sent_len = len(sentence)
            sent_vector = matrix[:, i]
            for item in sentence:  # 计算第i句的sent_vector
                token = str(item.word)
                pw = self.prob_dict.setdefault(token, default_p)
                weight = param_a / (param_a + pw)
                try:
                    word_vector = np.array(word_vectors.get_vector(token))
                    sent_vector += weight * word_vector
                except Exception:
                    continue

            matrix[:, i] = sent_vector / sent_len

        # PCA找到整个矩阵中，每个句子中最相似的部分（第一个主成分），然后减去相似部分
        U, s, Vh = np.linalg.svd(matrix)  # 默认s降序
        u = U[:, 0]  # 第一个主成分
        matrix -= np.outer(u, u.T) @ matrix  # 每个sent_vector减去在第一个主成分方向的投影

        return matrix

    @staticmethod
    def get_word_freq_dict(path):
        with open(path, 'rb+') as f:
            prob_dict = pickle.load(f, encoding='utf-8')
        return prob_dict
