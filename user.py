import numpy as np
import knn_split_eval
from math import log
from nltk.stem import WordNetLemmatizer
from collections import defaultdict                                # only for mapping of pos_tag as noun kept at default
from nltk import pos_tag
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()                                   # Init Lemmatizer


def reading_dic():                                                  # reading dictionary from disk
    with open('dictionary', "r") as file:
        dictionary = [word.strip() for word in file]
    file.close()
    return dictionary


def read_df():                                                       # df calculated in index file
    df = {}
    with open('df', "r") as file:
        rows = (line.split(':') for line in file)
        for row in rows:
            row[1] = int(row[1])
            df[row[0]] = row[1]
    file.close()
    return df


def q_tf_idf_score(query, df, vocab):
    q_tf = {}
    q_tf_idf = {}
    for token in query:
        q_tf[token] = query.count(token)
    for token in q_tf:
        if token in vocab:
            q_tf_idf[token] = np.round(((q_tf[token]) * ((log(df[token], 10))/737)), 5)
    return q_tf_idf


def make_query(query):
    tokens = []
    stop_words = set(stopwords.words('english'))
    tag_map = defaultdict(lambda: 'n')                                       # for mapping of pos_tag
    tag_map['J'] = 'a'
    tag_map['V'] = 'v'
    tag_map['R'] = 'r'
    query = [word for word in query if word not in stop_words]
    query = [''.join(w for w in string if w.isalnum()) for string in query]
    for token, tag in pos_tag(query):
        tokens.append(lemmatizer.lemmatize(token, tag_map[tag[0]]))         # lemmatization with POS_tag
    print(tokens)
    return tokens


def q_vectorization(q_tf_idf, vocab):
    q = np.zeros((len(vocab)))
    for token in q_tf_idf:
        if token in vocab:
            q[vocab.index(token)] = q_tf_idf[token]
    return q


def start(query):
    vocab = reading_dic()
    query = query.split()
    df = read_df()
    d_vec = knn_split_eval.load()
    query = make_query(query)
    features = []
    labels = []
    features, labels = knn_split_eval.assign_label(features, d_vec, labels)
    q_tf_idf = q_tf_idf_score(query, df, vocab)
    q_vec = q_vectorization(q_tf_idf, vocab)
    q_vecc = [q_vec]
    y_pred = knn_split_eval.knn(features, q_vecc, labels)
    return y_pred
