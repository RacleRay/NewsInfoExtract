# coding: utf-8


from collections import Counter
from pyhanlp import *
from utils import split_doc


OUTPUT_PATH_FREQ = r'../data/corpus_token_prob_dic.data'

tokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")

def cal_word_prob(input_file):
    """计算每个token在语料库中的出现概率"""
    total_tokens = []
    for doc in input_file:
        try:
            tokens = get_tokens_of_doc(doc)
            total_tokens.extend(tokens)
        except Exception:
            continue

    total_num = len(tokens)
    word_counter = Counter(tokens)
    prob_dict = {key: word_counter[key]/total_num for key in tokens}
    return prob_dict


def get_tokens_of_doc(doc):
    """分词获得token"""
    sent_list = split_doc(doc)
    tokens = []
    for sent in sent_list:
        tmp_tokens = tokenizer.segment(sent.strip())
        for token in tmp_tokens:
            tokens.append(str(token).split('/')[0].strip())
    return tokens


# if __name__ == "__main__":
    # 文本可以先通过pandas分析清理，再统计词频
    # 见jupyter notebook文件
    # prob_dict = cal_word_prob()