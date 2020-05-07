from LooselySymmetricNB import LooselySymmetricNB
import utils.preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

enron_dir = "datasets/enron1/"
iris_path = "datasets/iris.csv"

lsnb = LooselySymmetricNB()
# elsnb = LooselySymmetricNB(enhance=True)
lsnb.read_enron(dir=enron_dir, n_emails=1000, test_ratio=0.2)

# elsnb.read_enron(dir=enron_dir, n_emails=500, test_ratio=0)
# OR
# elsnb.X_internal_train = lsnb.X_internal_train
# elsnb.Y_internal_train = lsnb.Y_internal_train
# elsnb.X_internal_test = lsnb.X_internal_test
# elsnb.Y_internal_test = lsnb.Y_internal_test
# lsnb.read_csv(iris_path)

# print(lsnb.X_internal_train)
# print(elsnb.Y_internal_train)

lsnb.fit_internal()
# elsnb.fit_internal()

test_mails = [
    "Hello, I would like to set up a meeting if that's possible. Regards, Wyatt Schwarz.",
    "ONE TIME OFFER NOW!!! Click here and don't miss because you won a million dollars!"
]

test_mails_count = utils.preprocessing.vectorize(test_mails)
print(lsnb.predict(test_mails_count))
# print(elsnb.predict(test_mails_count))

print(f"LSNB accuracy: {lsnb.score(lsnb.X_internal_test.toarray(), lsnb.Y_internal_test)}")
print(f"LSNB precision: {precision_score(lsnb.Y_internal_test, lsnb.predict(lsnb.X_internal_test.toarray()))}")
print(f"LSNB recall: {recall_score(lsnb.Y_internal_test, lsnb.predict(lsnb.X_internal_test.toarray()))}")
print(f"LSNB f1 score: {f1_score(lsnb.Y_internal_test, lsnb.predict(lsnb.X_internal_test.toarray()))}")