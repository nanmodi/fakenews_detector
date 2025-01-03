import DataPrep
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize



count_vectorizer = CountVectorizer()
train_count = count_vectorizer.fit_transform(DataPrep.train_news['Statement'].values)


def get_countVectorizer_stats():
    
    print("Vocabulary Size:", train_count.shape[1])
   
    print("Vocabulary:", count_vectorizer.vocabulary_)
 
    print("Feature Names:", count_vectorizer.get_feature_names()[:25])


tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_count)

def get_tfidf_stats():
    
    print("TF-IDF Matrix Shape:", train_tfidf.shape)
   
    print(train_tfidf.A[:10])


tfidf_ngram = TfidfVectorizer(stop_words='english', ngram_range=(1, 4), use_idf=True, smooth_idf=True)


tagged_sentences = nltk.corpus.treebank.tagged_sents()
cutoff = int(0.75 * len(tagged_sentences))
training_sentences = DataPrep.train_news['Statement']


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


with open("glove.6B.50d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)
            for words in X
        ])