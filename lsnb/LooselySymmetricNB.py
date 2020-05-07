import numpy as np
from BaseDiscreteNB import _BaseDiscreteNB
from sklearn.utils.validation import check_non_negative
from sklearn.utils.extmath import safe_sparse_dot
import utils.preprocessing

class LooselySymmetricNB(_BaseDiscreteNB):
    
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, enhance=False):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.enhance = enhance
        self.X_internal_train = None
        self.Y_internal_train = None
        self.X_internal_test = None
        self.Y_internal_test = None

    def _count(self, X, Y, y):
        """Count and smooth feature occurrences."""
        
        check_non_negative(X, "LooselySymmetricNB (input X)")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)
        
        # we need these two values in order to calculate document frequency in _calculate_df()
        self.X = X
        self.y = y

    def _update_feature_log_prob(self, alpha):

        self.smoothed_fc = self.feature_count_ + alpha
        self.smoothed_cc = self.smoothed_fc.sum(axis=1)
        
        self._calculate_df()
        self._calculate_abcd(self.smoothed_fc, self.smoothed_cc.reshape(-1, 1), self.enhance)
        
        # HAM
        self.bd = (self.b * self.d) / (self.b + self.d)
        self.ac = (self.a * self.c) / (self.a + self.c)
        bd = (self.b * self.d) / (self.b + self.d)
        ac = (self.a * self.c) / (self.a + self.c)
        numerator = self.a + bd
        denumerator = self.a + self.b + ac + bd
        
        # index 0 is for ham, index 1 is for spam
        self.feature_log_prob_ = np.empty(self.feature_count_.shape) 
        self.feature_log_prob_[0] = np.log(numerator) - np.log(denumerator)
        
        # SPAM
        numerator = self.c + bd
        denumerator = self.c + self.d + ac + bd
        
        self.feature_log_prob_[1] = np.log(numerator) - np.log(denumerator)

    def _calculate_df(self):
        
        self.df = np.zeros(self.feature_count_.shape, dtype=np.int32)
        for mail_idx, mail in enumerate(self.X.toarray()):
            for word_idx, word in enumerate(mail):
                if word >= 1:
                    self.df[self.y[mail_idx]][word_idx] += 1

    def _calculate_abcd(self, fc, cc, enhance):
        
        # at 0 is ham info, at 1 is spam info
        if enhance:
            word_density_ham = fc[0] / cc[0]
            word_density_spam = fc[1] / cc[1]
        
        else:
            word_density_ham = 1
            word_density_spam = 1
        
        self.a = (self.df[0] / self.class_count_[0]) * word_density_ham
        self.b = (1 - self.a) * word_density_spam
        self.c = (self.df[1] / self.class_count_[1]) * word_density_spam
        self.d = (1 - self.c) * word_density_ham
    
    def _joint_log_likelihood(self, X):
       
        return (safe_sparse_dot(X, self.feature_log_prob_.T) + 
                self.class_log_prior_)

    def read_enron(self, dir, n_emails, test_ratio=0):
        mails, mails_labels = utils.preprocessing.read_enron(dir, n_emails)

        # check input
        if test_ratio is not 0:
            self.X_internal_train, self.X_internal_test, self.Y_internal_train, self.Y_internal_test = utils.preprocessing.preprocess(mails, mails_labels, test_ratio)
        else:
            self.X_internal_train, _, self.Y_internal_train, _ = utils.preprocessing.preprocess(mails, mails_labels, test_ratio)


    def read_csv(self, path):
        data, labels = utils.preprocessing.read_csv(path)
        print(data)
        print(labels)

    def fit_internal(self):
        self.fit(self.X_internal_train, self.Y_internal_train)