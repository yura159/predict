import re
import os
import nltk
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from gensim.corpora.dictionary import Dictionary
from gensim.models import LsiModel, TfidfModel
from gensim.models import Phrases
from gensim.models.phrases import Phraser

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class GensimVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, path=None):
        self.path = path
        self.id2word = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            self.id2word = Dictionary.load(self.path)

    def fit(self, documents, labels=None):
        bigram = Phrases(documents, min_count=10, threshold=2, delimiter=b' ')
        self.bigram_phraser = Phraser(bigram)
        bigram_token = []
        for sent in documents:
            bigram_token.append(self.bigram_phraser[sent])
        self.dct = Dictionary(bigram_token)
        self.corpus = [self.dct.doc2bow(line) for line in bigram_token]
        self.id2word = TfidfModel(self.corpus)
        return self

    def transform(self, documents):
        corpus = [self.dct.doc2bow(text) for text in documents]
        return self.id2word[corpus]


class GensimLsi(BaseEstimator, TransformerMixin):

    def __init__(self, mydict, num_topics, path=None):
        self.path = path
        self.mydict = mydict
        self.num_topics = num_topics
        self.model = None

    def load(self):
        if os.path.exists(self.path):
            self.model = LsiModel.load(self.path)

    def save(self):
        self.model.save(self.path)

    def make_vec(self, row_matrix, num_top):
        matrix = np.zeros((len(row_matrix), num_top))
        for i, row in enumerate(row_matrix):
            matrix[i, list(map(lambda tup: tup[0], row))] = list(map(lambda tup: tup[1], row))
        return matrix

    def fit(self, documents, labels=None):
        self.model = LsiModel(documents, id2word=self.mydict, num_topics=self.num_topics)
        return self

    def transform(self, documents):
        corpus = self.model[documents]
        documents = self.make_vec(corpus, self.model.num_topics)
        return documents


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, norm=None):
        self.stopwords = set(nltk.corpus.stopwords.words('russian'))
        self.norm = norm

    def remove_email(self, text):
        email = re.compile(r'\S+@\S+\.\S+$')
        return email.sub(r' ', text)

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r' ', text)

    def remove_html(self, text):
        html = re.compile(r'<.*?>')
        return html.sub(r' ', text)

    def remove_mail(self, text):
        mail = re.compile(r'^([a-z0-9_\.-]+)@([a-z0-9_\.-]+)\.([a-z\.]{2,6})$')
        return mail.sub(r' ', text)

    def remove_emoji(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r' ', text)

    stop_words = set(stopwords.words("russian"))

    def mytokenize(self, text):
        text = self.remove_email(text)
        text = self.remove_html(text)
        text = self.remove_URL(text)
        text = self.remove_emoji(text)
        text = self.remove_mail(text)
        text = text.lower()
        text = re.sub("[^а-яёйa-z0-9]", " ", text)
        text = re.sub("\s+", " ", text)
        text = word_tokenize(text)
        text = [word for word in text if word.isalpha()]
        return text

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        return [self.mytokenize(document) for document in documents]
