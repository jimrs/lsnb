from LooselySymmetricNB import LooselySymmetricNB
import utils.preprocessing
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

lsnb = LooselySymmetricNB()
data, labels = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, shuffle=True)

clf = OneVsRestClassifier(lsnb).fit(X_train, y_train)

print(X_train)
print(X_test[:10])
print(y_test[:10])

print(clf.predict(X_test[:10]))