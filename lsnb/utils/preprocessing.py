import os
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def read_enron(dir, n_emails=None):
    mails = []
    mails_labels = []
    spam_dir = os.path.join(dir, "spam")
    ham_dir = os.path.join(dir, "ham")

    i = 0
    ham_dir_files = os.listdir(ham_dir)
    for file in ham_dir_files:
        with open(os.path.join(ham_dir, file), "r", encoding="latin-1") as f:
            mail = f.read()
            mails.append(mail)
            mails_labels.append(0)

            i += 1
            if n_emails is not None and i == n_emails:
                break
            
    i = 0
    spam_dir_files = os.listdir(spam_dir)
    for file in spam_dir_files:
        with open(os.path.join(spam_dir, file), "r", encoding="latin-1") as f:
            mail = f.read()
            mails.append(mail)
            mails_labels.append(1)

            i += 1
            if n_emails is not None and i == n_emails:
                break

    return mails, mails_labels

# only IRIS so far
def read_csv(path):
    df = pd.read_csv(path)
    data = df[df.columns[0:4]]
    labels = df['species']
    return data.values, labels.values

_porter_stemmer = nltk.stem.porter.PorterStemmer()

# Sebastian Raschka
def _tokenize(text, stemmer=_porter_stemmer):
    lower_text = text.lower()
    tokens = nltk.wordpunct_tokenize(lower_text)
    stems = [_porter_stemmer.stem(token) for token in tokens]
    punct_less = [stem for stem in stems if re.match(
        '^[a-zA-Z]+$', stem
    ) is not None]
    return punct_less

_vectorizer = CountVectorizer(
    encoding="latin-1",
    decode_error="replace",
    strip_accents="unicode",
    analyzer="word",
    binary=False,
    tokenizer = _tokenize,
    ngram_range=(1,1),
    max_df=0.99,
    min_df=2
)

def preprocess(data, labels, test_ratio=0.0):

    with open("./stop_words.txt", "r") as f:
        _stop_words = f.read().splitlines()

    # get rid of inconsistent tokenize warning
    stop_words = []
    for word in _stop_words:
        stop_words.append(_tokenize(word)[0])
    stop_words = list(dict.fromkeys(stop_words))  # remove duplicates

    _vectorizer.stop_words = stop_words

    if test_ratio == 0:
        X_train_count = _vectorizer.fit_transform(data)
        y_train = labels

        X_test_count = None
        y_test = None

    else:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = test_ratio)

        X_train_count = _vectorizer.fit_transform(X_train)
        X_test_count = _vectorizer.transform(X_test)

    return X_train_count, X_test_count, y_train, y_test

# used to get the same vectorization settings for other data (testing mails eg.)
def vectorize(data):
    return _vectorizer.transform(data)

# getter
@property
def vectorizer():
    return _vectorizer