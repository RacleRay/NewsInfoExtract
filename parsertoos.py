# coding: utf-8

from pyhanlp import *
from config import Config


class Parsertool:
    """在pyhanlp依存句法分析基础上，定制的分析工具类。pyhanlp在中文NER方面效果相对其他包效果更好一些。

    say_similarities: words that are similar to ‘say’. set()
    noun_pos_candis: 语句抽取主语时的词性候选集合， set()
    """
    def __init__(self):
        self.say_similarities = self.get_say_similarities(Config.SIM_TO_SAY_FILE_PATH)
        self.noun_pos_candis = Config.NOUN_POS_CANDIDATE
        self.hanlp_dep_parser = HanLP

    def parse(self, sentence):
        """解析sentence的主要方法

        sentence: sentence to process, string
        """
        # print("原句: ", sentence)
        subject_candinates = []
        content_candidates = []
        verb_candinates = self.say_similarities
        noun_pos_candis = self.noun_pos_candis

        verb_node = None
        subject_node = None  # 分词器识别的初始主语
        closest_verb = None  # 分词器分词错误情况修正

        parseRes = self.hanlp_dep_parser.parseDependency(sentence)
        for node in parseRes.iterator():
            print(node.toString())
            # 从头遍历出现的名词，作为可能的主语
            if verb_node is None or node.ID < verb_node.ID:
                if node.POSTAG in noun_pos_candis:
                    subject_candinates.append(node)
                # 处理多个主语并列的情况
                elif node.LEMMA == '、' and node.HEAD.POSTAG in noun_pos_candis:
                    subject_candinates.append(node)

            #默认一句只有一个主谓关系
            if node.DEPREL == '主谓关系' and \
              node.HEAD.LEMMA in verb_candinates and \
              verb_node is None:
                verb_node = node.HEAD
                subject_node = node
                # print(subject_node)

            if verb_node and node.DEPREL == '并列关系' and node.HEAD.ID == verb_node.ID:
                closest_verb = node

            # 谓词后面选择对应的内容
            if verb_node is not None and node.ID > verb_node.ID:
                # print('content node: ', node.toString())
                content_candidates.append(node)

        if len(subject_candinates)  == 0:
            print("没有符合条件的主语")
            return 'No_subject'
        if verb_node is None:
            components = sentence.split('：')
            # 出现： “xxx部门：ssssss。”  的情况
            if len(sentence.split('：')) == 2:
                group1 = HanLP.parseDependency(components[0])
                node_array = group1.getWordArray()
                if all([(node.POSTAG in noun_pos_candis or node.POSTAG == 'w') for node in node_array]):
                    subject = components[0]
                    verb = '：'
                    content = components[1]
                    return subject, verb, content
            print("没有符合条件的谓语")
            return 'No_verb'

        subject = self.get_subject(subject_candinates, subject_node)
        content = self.get_content(content_candidates, sentence, verb_node, closest_verb)

        return subject, verb_node.LEMMA, content

    @staticmethod
    def get_subject(subject_candinates, subject_node):
        """获取主语

        candinates_list: 保存依存分析的节点，list；
        subject_node: 当前找到的主语节点, hanlp node；
        """
        subject = ''
        for node in subject_candinates:
            # print()
            # print(node.toString())
            if node.DEPREL in {'定中关系', '主谓关系', '并列关系', '左附加关系', '右附加关系'}:
                try:
                    if node.ID == subject_node.ID:
                        subject += node.LEMMA
                    if node.HEAD.ID == subject_node.ID:  # 防止没有HEAD异常中断，分开if
                        subject += node.LEMMA
                    if node.HEAD.HEAD.ID == subject_node.ID:
                        subject += node.LEMMA
                    if node.HEAD.HEAD.HEAD.ID == subject_node.ID:
                        subject += node.LEMMA
                except AttributeError:
                    continue

            elif node.LEMMA == '、' and subject != '':
                subject += node.LEMMA

        return subject

    @staticmethod
    def get_content(content_candidates, sentence, verb_node, closest_verb):
        """获取言论内容

        sentence: 原文, string
        content_candidates: hanlp分词的node list
        """
        # 存在 ： 的特殊情况
        if "：" in sentence:
            return sentence[sentence.index("：")+1: ]
        elif "，“" in sentence:
            return sentence[sentence.index("，“")+2: ]
        elif ":" in sentence:
            return sentence[sentence.index(":")+1: ]

        # 处理以下情况：“没有人会在意的”，小明说。
        if len(content_candidates) == 0:
            if '“' in sentence:
                from_ = sentence.rfind('“')
                to_ = sentence.rfind('”')
                return sentence[from_ + 1: to_]
            else:
                return ''

        content = ''
        WP_FLAG = False  # 防止谓词后面有直接宾语等的复杂情况
        FIRST_WP = True
        cur_verb_node = verb_node if not closest_verb else closest_verb
        for idx, node in enumerate(content_candidates):
            if node.DEPREL == '标点符号' and node.HEAD.ID == cur_verb_node.ID and FIRST_WP:
                WP_FLAG = True
                FIRST_WP = False
                content = ''
                continue
            if node.DEPREL in {'兼语', '右附加关系', '动补结构'} and not WP_FLAG:  # 主句谓词的修饰:
                if node.HEAD.ID == cur_verb_node.ID or node.HEAD.ID == verb_node.ID:
                    continue
            content += node.LEMMA

        return content

    @staticmethod
    def get_say_similarities(path):
        """根据Config.SIM_TO_SAY_FILE_PATH，修改文件路径"""
        with open(path, 'r',encoding='utf-8') as f:
            say_similarities = set()
            while True:
                line = f.readline()
                if line == '': break
                say_similarities.add(line.strip())
            return say_similarities
