import os
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()


def pre_process(words, stop_words, tag_map):
    filtered_words = []
    dic = [word.lower() for word in words]
    dic = [''.join(w for w in string if w.isalnum()) for string in dic]
    dic = [w for w in dic if w not in stop_words]
    dic = [word for word in dic if word]
    for token, tag in pos_tag(dic):
        filtered_words.append(lemmatizer.lemmatize(token, tag_map[tag[0]]))
    return filtered_words


def create_index(df, index, filtered_words, doc_id):
    for word in filtered_words:
        if word not in index:
            index.append(word)
            df[word] = [doc_id]
        else:
            if doc_id not in df[word]:
                df[word].append(doc_id)
    return index, df


def tag_match():                                                                # for mapping of pos_tag
    tag_map = defaultdict(lambda: 'n')
    tag_map['J'] = 'a'
    tag_map['V'] = 'v'
    tag_map['R'] = 'r'
    return tag_map


def cal_df(df):
    for term in df:
        df[term] = len(df[term])
    return df


def cal_tf(tf, tokens, doc_id):
    for token in tokens:
        tf[token, doc_id] = tokens.count(token)
    return tf


def df_save(df):                                                            # save document freq on disk
    f = open('df', 'w')
    for key, value in df.items():
        f.write('%s:%s\n' % (key, value))
    f.close()


def tf_save(tf):                                       # save term freq on disk
    with open('tf', 'w') as f:
        for key, value in tf.items():
            f.write('%s:%s\n' % (key, value))
    f.close()


def reading_dataset():
    dirr = 'bbcsport/athletics/'
    files = os.listdir("bbcsport/athletics")
    athleticFiles = [dirr + file for file in files]
    tag_map = tag_match()
    df = {}
    term_freq = {}
    index = []
    doc_id = 0
    stop_words = set(stopwords.words('english'))

    for file in athleticFiles:
        f = open(file, 'r')
        dictionary = (f.read().split())
        filtered_words = pre_process(dictionary, stop_words, tag_map)
        index, df = create_index(df, index, filtered_words, doc_id)
        term_freq = cal_tf(term_freq, filtered_words, doc_id)
        doc_id += 1
        f.close()
    print(doc_id)

    dirr = 'bbcsport/cricket/'
    files = os.listdir("bbcsport/cricket")
    cricketFiles = [dirr + file for file in files]
    for file in cricketFiles:
        f = open(file, 'r')
        dictionary = (f.read().split())
        filtered_words = pre_process(dictionary, stop_words, tag_map)
        index, df = create_index(df, index, filtered_words, doc_id)
        term_freq = cal_tf(term_freq, filtered_words, doc_id)
        doc_id += 1
        f.close()
    print(doc_id)

    dirr = 'bbcsport/football/'
    files = os.listdir("bbcsport/football")
    footballFiles = [dirr + file for file in files]
    for file in footballFiles:
        f = open(file, 'r')
        dictionary = (f.read().split())
        filtered_words = pre_process(dictionary, stop_words, tag_map)
        index, df = create_index(df, index, filtered_words, doc_id)
        term_freq = cal_tf(term_freq, filtered_words, doc_id)
        doc_id += 1
        f.close()
    print(doc_id)

    dirr = 'bbcsport/rugby/'
    files = os.listdir("bbcsport/rugby")
    rugbyFiles = [dirr + file for file in files]
    for file in rugbyFiles:
        f = open(file, 'r')
        dictionary = (f.read().split())
        filtered_words = pre_process(dictionary, stop_words, tag_map)
        index, df = create_index(df, index, filtered_words, doc_id)
        term_freq = cal_tf(term_freq, filtered_words, doc_id)
        doc_id += 1
        f.close()
    print(doc_id)

    dirr = 'bbcsport/tennis/'
    files = os.listdir("bbcsport/tennis")
    tennisFiles = [dirr + file for file in files]
    for file in tennisFiles:
        f = open(file, 'r')
        dictionary = (f.read().split())
        filtered_words = pre_process(dictionary, stop_words, tag_map)
        index, df = create_index(df, index, filtered_words, doc_id)
        term_freq = cal_tf(term_freq, filtered_words, doc_id)
        doc_id += 1
        f.close()
    print(doc_id)
    df = cal_df(df)
    return df, term_freq

def main():

    df, term_freq = reading_dataset()
    tf_save(term_freq)
    df_save(df)


if __name__ == '__main__':
    main()