# coding: utf-8

from gensim.models import KeyedVectors
from collections import deque

PATH_W2V_FILE = r'../data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'
OUTPUT_PATH = r'../data/similar_say_words.txt'

# designed by yourself
initial_list = ['说', '指出', '称', '声称',  '表示', '感叹', '认为', '宣称', '坦言', '赞叹', '回复', '介绍'
                 '提及', '相信', '重申', '承认', '证实', '揭示', '深信', '断定', '斥责', '强调', '批评']
word_vectors = KeyedVectors.load_word2vec_format(PATH_W2V_FILE)

def get_related_words(initial_words, model, top_k, max_size):
    """
    获取与‘说’、‘表示’、‘称’等词的相似词
    NOTE：寻找的结果需要手动检查，结合测试修改。size不要太大。

    initial_words: initial words we already know, list
    model: the word2vec model
    top_k: top k similar
    """
    cur_size = len(initial_words)
    words_set = set(initial_words)
    bfs_queue = deque(initial_words)

    while cur_size <= max_size and len(bfs_queue) > 0:
        cur_word = bfs_queue.popleft()
        cur_sims = [w for w, s in model.similar_by_word(cur_word, topn=top_k)]
        new_words = set(cur_sims).difference(set(words_set))

        words_set.update(new_words)
        bfs_queue.extend(new_words)

        cur_size = len(words_set)

    return words_set


if __name__ == "__main__":
    say_similarities = get_related_words(initial_list, word_vectors, 10, len(initial_list) * 15)
    with open(OUTPUT_PATH, 'w',encoding='utf-8') as  f:
        for word in say_similarities:
            f.write(str(word)+'\n')
