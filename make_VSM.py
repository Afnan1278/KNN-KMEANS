import math
from collections import OrderedDict

import numpy as np


def reading_tf():  # tf calculated in index file and stored on disk
    tf = {}
    with open('tf', "r") as file:
        rows = (line.split(':') for line in file)
        for row in rows:
            res = row[0].strip(')(').split(', ')
            res[0] = res[0].strip("''")
            res[1] = int(res[1])
            row[1] = int(row[1])
            tf[res[0], res[1]] = row[1]

    file.close()
    return tf


def read_df():  # df calculated in index file
    df = {}
    with open('df', "r") as file:
        rows = (line.split(':') for line in file)
        for row in rows:
            row[1] = int(row[1])
            df[row[0]] = row[1]
    file.close()
    return df


def tf_idf_score(tf, df):
    tf_idf = {}
    for (term, doc) in tf:
        tf_idf[term, doc] = ((tf[term, doc]) * ((math.log(df[term], 10)) / 737))  # idf = tf * idf
    return tf_idf


def doc_vectorization(vocab, tf_idf):
    d = np.zeros((737, len(vocab)))
    for (term, doc) in tf_idf:
        d[doc][vocab.index(term)] = tf_idf[term, doc]
    return d


def feature_selection(tf_idf):
    tf_idf = OrderedDict(sorted(tf_idf.items(), key=lambda x: x[1], reverse=True))
    i = 0
    vocab = []
    selected = {}
    for term, doc in tf_idf:
        if term not in vocab:
            vocab.append(term)
            i += 1
        if i == 4613:
            break
    for term, doc in tf_idf:
        if term in vocab:
            selected[term, doc] = tf_idf[term, doc]
    return selected, vocab


def dic_save(dictionary):  # save dictionary on disk
    with open('dictionary', 'w') as f:
        for term in dictionary:
            f.write('%s\n' % term)
    f.close()


def start():
    df = read_df()
    tf = reading_tf()
    tf_idf = tf_idf_score(tf, df)
    selected, vocab = feature_selection(tf_idf)
    d_vec = doc_vectorization(vocab, selected)
    print('shape of document vectors', d_vec.shape)
    print(selected)
    np.savetxt('d_vec.csv', d_vec, delimiter=',', fmt='%1.7f')
    dic_save(vocab)


def main():
    start()


if __name__ == '__main__':
    main()
