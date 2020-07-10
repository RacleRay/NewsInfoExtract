# coding: utf-8


class Config:
    alpha = 0.2  # 句子余弦距离的阈值，判断两句话是否是统一主语说的内容。 1 - cosine similarity
    expect_min_cont_len = 5  # 设置提取内容的最短长度
    param_a = 0.0001  # sentence vector中权重计算公式中的参数，论文实现显示在skipgram + negetive sampling的情况下，1e-4效果较好0.0001
                     # A SIMPLE BUTTOUGH-TO-BEATBASELINE  FORSEN-TENCEEMBEDDINGS. ICLR2017

    NOUN_POS_CANDIDATE = {'nh', 'ni', 'nt', 'nr', 'n', 'ns', 'nz', 'nto', 'm', 'nrf', 'nx'}

    CORPUS_FREQUENCY_PATH = r'./data/corpus_token_prob_dic.data'
    W2V_MODEL_PATH = r'./data/selected_words_vec_300.txt'
    # W2V_MODEL_PATH = r'./data/samll_w2v'
    SIM_TO_SAY_FILE_PATH = r'./data/similar_say_words.txt'
