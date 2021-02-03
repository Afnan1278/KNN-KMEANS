import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from collections import Counter


def split(features, labels):
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.3, random_state=43)
    return x_train, x_test, y_train, y_test


def cosine_similarity(x_train, test_doc):
    similar = []
    train_docs = len(x_train)
    doc = 0
    while doc != train_docs:
        similar.append(np.dot(x_train[doc], test_doc)
                       / (np.sqrt(np.sum(x_train[doc] ** 2)) * np.sqrt(np.sum(test_doc ** 2))))
        doc += 1
    return similar


def voter(labels_list):
    count = Counter(labels_list)
    return count.most_common()[0][0]


def find_top_k_docs(similar, k):
    final_list = []

    for i in range(0, k):
        max1 = 0

        for j in range(len(similar)):
            if similar[j] > max1:
                max1 = similar[j]
        if np.isnan(similar).all():
            return -1
        final_list.append(similar.index(max1))
        similar.remove(max1)
    return final_list


def knn(x_train, x_test, y_train):
    y_pred = []
    l = len(x_test)
    k_docs_labels = []
    for i in range(0, l):
        similar = cosine_similarity(x_train, x_test[i])
        if np.isnan(similar).all():                             # no document is similar
            return [-1]
        k_docs = find_top_k_docs(similar, 3)
        for j in range(len(k_docs)):
            k_docs_labels.append(y_train[k_docs[j]])

        winner = voter(k_docs_labels)
        k_docs_labels = []
        y_pred.append(winner)
    return y_pred


def assign_label(features, d_vec, labels):
    for i in range(0, 101):
        features.append(d_vec[i])
        labels.append(1)
    for i in range(101, 225):
        features.append(d_vec[i])
        labels.append(2)
    for i in range(225, 490):
        features.append(d_vec[i])
        labels.append(3)
    for i in range(490, 637):
        features.append(d_vec[i])
        labels.append(4)
    for i in range(637, 737):
        features.append(d_vec[i])
        labels.append(5)
    return features, labels


def load():
    d_vec = np.loadtxt('d_vec.csv', delimiter=',')
    return d_vec


def evaluation(y_pred, y_test):
    res = accuracy_score(y_test, y_pred)
    return res


def main():
    d_vec = load()
    features = []
    labels = []
    features, labels = assign_label(features, d_vec, labels)
    x_train, x_test, y_train, y_test = split(features, labels)
    y_pred = knn(x_train, x_test, y_train)
    accuracy = evaluation(y_pred, y_test)
    print("accuracy", accuracy)


if __name__ == '__main__':
    main()
