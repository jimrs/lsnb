# Loosely Symmetric Naive Bayes classifier
Implementation of Loosely Symmetric Naive Bayes classifier as described by Taniguchi et al. in [here](http://doi.org/10.1038/s41598-018-25679-z).

The code here is based on scikit-learn code of MultinomialNB. It inherits from BaseDiscreteNB and BaseNB classes similarly to MultinomialNB. Both BaseDiscreteNB and BaseNB classes are implemented here for clarity, learning purposes, and for a slight small change in BaseDiscreteNB.

## Requirements
* scikit-learn
* nltk
* numpy
* pandas for csv reading

## Installation
* todo

## Jupyter notebooks
* experiment for showing precision and accuracy compared to MNB, BNG, GNB
* experiment for other datasets (iris)
* experiment for data intrusion

## Simple example for email spam classification
### Loading the Enron dataset
This creates a LSNB class instance, reads the first 200 emails of Enron dataset (or any dataset in the same format - parent directory with 2 spam and ham directories inside, and .txt files for each mail inside those directories, inside those files subject and body), 100 mails spam and the other 100 ham. Then the mails are processed with NLTK tokenization, stemming, stopwords removal, and vectorized with scikit CountVectorizer.

```python
from LooselySymmetricNB import LooselySymmetricNB

enron_dir = "datasets/enron1/"
lsnb = LooselySymmetricNB()
lsnb.read_enron(dir=enron_dir, n_emails=200)
```

### Fitting the classifier
Fitting the classifier is as easy as calling the method `fit_internal()`, which will fit the LSNB to the data loaded in previous steps.

```python
lsnb.fit_internal()
```

### Classify a few emails
Now you can try classifying emails with the LSNB. First you define an array with the mails, then with the use of utils.preprocessing function, you vectorize it. The same vectorizer is used in the loading of the dataset, which ensures the same results. Then you call the predict() method.

```python
test_mails = [
    "Hello, I would like to set up a meeting if that's possible. Regards, Wyatt Schwarz.",
    "ONE TIME OFFER NOW!!! Click here and don't miss because you won a million dollars!"
]

test_mails_count = utils.preprocessing.vectorize(test_mails)
lsnb.predict(test_mails_count)
```

The output of `lsnb.predict()` can be printed, yielding the result `[0 1]`. This means that the classifier has labeled the first mail as non-spam, and the second mail as spam. The result depends on the training data and it's size of course.

### eLSNB
There is also an enhanced version of the LSNB classifier. You can use it the same way as the plain LSNB classifier, just pass an `enhanced=True` argument to the constructor.

```python
from LooselySymmetricNB import LooselySymmetricNB

elsnb = LooselySymmetricNB(enhanced=True)
```

If you want to work with both LSNB and eLSNB at the same time, there is no need to call `read_enron()` twice, you can simply assign the data from one classifier to the other.

```python
lsnb.read_enron(dir=enron_dir, n_emails=200)
elsnb.X_internal_train = lsnb.X_internal_train
elsnb.Y_internal_train = lsnb.Y_internal_train
```

### Metrics using scikit-learn
To test the performance of these classifiers, you can use the metrics functions scikit-learn provides. But first, you should divide the dataset into a training and testing part by passing a `test_ratio=0.5` argument to the `read_enron()` method. The test_ratio parameter accepts any float in the range <0.0 ; 1).

```python
lsnb.read_enron(dir=enron_dir, n_emails=200, test_ratio=0.5)
```

Now the LSNB instance has both X_internal_train and X_internal_test properties, together with their Y (labels) counterparts. You can pass these test data properties to the specific scikit metrics functions.

```python
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print(f"LSNB accuracy:      {lsnb.score(lsnb.X_internal_test.toarray(), lsnb.Y_internal_test)}")
print(f"LSNB precision:     {precision_score(lsnb.Y_internal_test, lsnb.predict(lsnb.X_internal_test.toarray()))}")
print(f"LSNB recall:        {recall_score(lsnb.Y_internal_test, lsnb.predict(lsnb.X_internal_test.toarray()))}")
print(f"LSNB f1 score:      {f1_score(lsnb.Y_internal_test, lsnb.predict(lsnb.X_internal_test.toarray()))}")
```