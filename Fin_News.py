import re
import jieba
import time
import pandas as pd
# from gensim.summarization.bm25 import BM25

import math
from six import iteritems
from six.moves import xrange


PARAM_K1 = 1.51
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.doc_len = []
        self.initialize()

    def initialize(self):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        for document in self.corpus:
            frequencies = {}
            self.doc_len.append(len(document))
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document, index, average_idf):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[index] / self.avgdl)))
        return score

    def get_scores(self, document, average_idf):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores

def show(msg):
    print("[INFO]", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), msg)


def key_in_dic(key, dic):
    try:
        dic[key]
    except(KeyError):
        return False
    return True

punctuations = ['“', '”']
# stopword = [line.strip() for line in open("/media/shelly/X/FinTech_News/stop_words.txt").readlines()]

# train
show('Process train data.')
train_data = pd.read_csv('train_data.csv')
train_corpus = []
temp_count = 0
for line_tr in range(len(train_data)):
    train_titles = train_data['title'][line_tr]
    train_titles = re.sub("[A-Za-z0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%]", "", train_titles)

    # train_cut_word_list = jieba.cut(train_titles)
    train_cut_word_list = [w for w in jieba.cut(train_titles) if w not in punctuations]
    words_vec = [word for word in train_cut_word_list]
    train_corpus.append(words_vec)
    temp_count += 1
    # print(temp_count)
print(train_corpus[:5])


# pre process
LF_word_vec = []     #低频词列表
count_dic = {}
word_line_count = 0
for line in train_corpus:
    word_line_count += 1
    for s in line:
        if s in count_dic:
            count_dic[s] += 1
        else:
            count_dic[s] = 1
    # print(word_line_count)
print("done")
LF_word_dic = {}
for k, v in count_dic.items():
    if v <= 1:
        LF_word_dic[k] = ''
print("done")

# corpus_once = set(c for c in set(all_corpus) if all_corpus.count(c) <= 2)
# final_corpus = [[c for c in corpus if c not in LF_word] for corpus in train_corpus]  #处理后最终的语料库
final_corpus = []  #处理后最终的语料库
for line in train_corpus:
    temp_word = []
    for s in line:
        if s not in LF_word_dic:
            temp_word.append(s)
    final_corpus.append(temp_word)
print(final_corpus[:5])


# test
test_data = pd.read_csv('test_data.csv', encoding='gbk')
test_corpus = []
test_titles = test_data['title']
for line_te in test_titles:
    # test_cut_word_list = [w for w in jieba.cut(line_te)]
    test_cut_word_list = [w for w in jieba.cut(line_te) if w not in punctuations]
    test_corpus.append(test_cut_word_list)


# analyze
def id_analyze(corpus):
    show('Begin Analyze~')
    bm25_model = BM25(corpus)
    average_idf = sum(float(val) for val in bm25_model.idf.values()) / len(bm25_model.idf)

    weights = []
    for test_doc in test_corpus:
        scores = bm25_model.get_scores(test_doc, average_idf)
        weights.append(scores)

    result_vec = []
    for i, index in enumerate(test_data['id']):
        source_id = index
        similarities = sorted(enumerate(weights[i]), key=lambda item: -item[1])[:21]   #根据相似度从大到小排序，取前20个
        for j in range(21):
            target_id = similarities[j][0] + 1
            if (source_id != target_id):
                result_vec.append([source_id, target_id])
    return result_vec


# reserve
def id_reserve(res):
    show('Reserve the results.')
    with open("bm_results_05102200.txt", "w") as f:
        f.write("source_id" + "\t" + "target_id" + "\n")
        for i in range(len(res)):
            f.write(str(res[i][0]) + '\t' + str(res[i][1]) + '\n')


if __name__ == "__main__":
    results = id_analyze(final_corpus)
    id_reserve(results)
    show('Reserve over.')
