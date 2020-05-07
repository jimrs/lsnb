from LooselySymmetricNB import LooselySymmetricNB
import os
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

mails_dir = "datasets/enron1/"
spam_dir = os.path.join(mails_dir, "spam")
ham_dir = os.path.join(mails_dir, "ham")

mails = []
mails_labels = []

i = 0
ham_dir_files = os.listdir(ham_dir)
for file in ham_dir_files:
    with open(os.path.join(ham_dir, file), "r", encoding="latin-1") as f:
        mail = f.read()
        mails.append(mail)
        mails_labels.append(0)

        i += 1
        if i == 100:
            break
            
i = 0
spam_dir_files = os.listdir(spam_dir)
for file in spam_dir_files:
    with open(os.path.join(spam_dir, file), "r", encoding="latin-1") as f:
        mail = f.read()
        mails.append(mail)
        mails_labels.append(1)

        i += 1
        if i == 100:
            break

porter_stemmer = nltk.stem.porter.PorterStemmer()

# Sebastian Raschka
def tokenize(text, stemmer=porter_stemmer):
    lower_text = text.lower()
    tokens = nltk.wordpunct_tokenize(lower_text)
    stems = [porter_stemmer.stem(token) for token in tokens]
    punct_less = [stem for stem in stems if re.match(
        '^[a-zA-Z]+$', stem
    ) is not None]
    return punct_less

# _stop_words = nltk.corpus.stopwords.words("english")
# with open("./stop_words.txt", "w") as f:
#     f.write("\n".join(_stop_words))

with open("./stop_words.txt", "r") as f:
    _stop_words = f.read().splitlines()

# get rid of inconsistent tokenize warning
stop_words = []
for word in _stop_words:
    stop_words.append(tokenize(word)[0])
stop_words = list(dict.fromkeys(stop_words))  # remove duplicates

vectorizer = CountVectorizer(
    encoding="latin-1",
    decode_error="replace",
    strip_accents="unicode",
    analyzer="word",
    binary=False,
    stop_words = stop_words,
    tokenizer = tokenize,
    ngram_range=(1,1),
    max_df=0.99,
    min_df=2
)

X_train, X_test, y_train, y_test = train_test_split(mails, mails_labels, test_size = 0.5)

X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

lsnb = LooselySymmetricNB()
lsnb.fit(X_train_count, y_train)
elsnb = LooselySymmetricNB(enhance=True)
elsnb.fit(X_train_count, y_train)

test_mails = [
    "Hello, I would like to set up a meeting if that's possible. Regards, Wyatt Schwarz.",
    "ONE TIME OFFER NOW!!! Click here and don't miss because you won a million dollars!"
]

test_mails_count = vectorizer.transform(test_mails)
print(lsnb.predict(test_mails_count))
print(elsnb.predict(test_mails_count))