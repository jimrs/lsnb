import numpy as np
from lsnb.lsnb.BaseDiscreteNB import _BaseDiscreteNB
from sklearn.utils.validation import check_non_negative
from sklearn.utils.extmath import safe_sparse_dot
import lsnb.lsnb.utils.preprocessing

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
        
        # CLASS 0
        self.bd = (self.b * self.d) / (self.b + self.d)
        self.ac = (self.a * self.c) / (self.a + self.c)
        bd = (self.b * self.d) / (self.b + self.d)
        ac = (self.a * self.c) / (self.a + self.c)
        numerator = self.a + bd
        denumerator = self.a + self.b + ac + bd
        
        # index 0 is for class 0, index 1 is for class 1
        self.feature_log_prob_ = np.empty(self.feature_count_.shape) 
        self.feature_log_prob_[0] = np.log(numerator) - np.log(denumerator)
        
        # CLASS 1
        numerator = self.c + bd
        denumerator = self.c + self.d + ac + bd
        
        self.feature_log_prob_[1] = np.log(numerator) - np.log(denumerator)

    def _calculate_df(self):
        
        self.df = np.zeros(self.feature_count_.shape, dtype=np.int32)
        for sample_idx, sample in enumerate(self.X.toarray()):
            for word_idx, word in enumerate(sample):
                if word >= 1:
                    self.df[self.y[sample_idx]][word_idx] += 1

    def _calculate_abcd(self, fc, cc, enhance):
        
        # at 0 is class 1 info, at 1 is class 1 info
        if enhance:
            word_density_class0 = fc[0] / cc[0]
            word_density_class1 = fc[1] / cc[1]
        
        else:
            word_density_class0 = 1
            word_density_class1 = 1
        
        self.a = (self.df[0] / self.class_count_[0]) * word_density_class0
        self.b = (1 - self.a) * word_density_class1
        self.c = (self.df[1] / self.class_count_[1]) * word_density_class1
        self.d = (1 - self.c) * word_density_class0
    
    def _joint_log_likelihood(self, X):
       
        return (safe_sparse_dot(X, self.feature_log_prob_.T) + 
                self.class_log_prior_)

    def read_enron(self, dir, n_emails, test_ratio=0):
        mails, mails_labels = lsnb.lsnb.utils.preprocessing.read_enron(dir, n_emails)

        # check input
        if test_ratio is not 0:
            self.X_internal_train, self.X_internal_test, self.Y_internal_train, self.Y_internal_test = lsnb.lsnb.utils.preprocessing.preprocess(mails, mails_labels, test_ratio)
        else:
            self.X_internal_train, _, self.Y_internal_train, _ = lsnb.lsnb.utils.preprocessing.preprocess(mails, mails_labels, test_ratio)

    def fit_internal(self):
        self.fit(self.X_internal_train, self.Y_internal_train)