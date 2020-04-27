""" 
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np
import random
from scipy.special import gammaln
from tqdm import tqdm


class LDA(object):
    def __init__(self, *, inference):
        self.inference = inference
        self.topic_words = None


    def fit(self, *, X, iterations):
        self.inference.inference(X=X, iterations=iterations)
        

    def predict(self, *, vocab, K):
        self.topic_words = {}
        preds = []
        for i, topic_dist in enumerate(self.inference.phi):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(K+1):-1]
            self.topic_words[i] = topic_words.tolist()
            preds.append('Topic {}: {}'.format(i, ' '.join(topic_words)))
        return preds


class Inference(object):
    """ 
    Abstract inference object.
    """

    def __init__(self, num_topics, num_docs, num_words, alpha, beta):
        self.num_topics = num_topics
        self.num_docs = num_docs
        self.num_words = num_words
        self.alpha = alpha
        self.beta = beta
        self.theta = np.zeros((num_docs, num_topics))
        self.phi = np.zeros((num_topics, num_words))
        self.loglikelihoods = []

    def inference(self, *, X, iterations):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of ints with shape
                [num_docs, num_words].
            iterations: int giving number of iterations
        """
        raise NotImplementedError()


class GibbsSampling(Inference):

    def __init__(self, *, num_topics, num_docs, num_words, alpha, beta):
        super().__init__(num_topics, num_docs, num_words, alpha, beta)
        self.ndz = np.zeros((self.num_docs, self.num_topics))
        self.nzw = np.zeros((self.num_topics, self.num_words))
        self.nz = np.zeros(self.num_topics)
        self.topics = {} # this is Z in the assignment sheet


    def initialize(self, X):
        """
        Helper function to initialize z as described in Section 2.3.
        Call this function from inference.
        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_docs, num_words].
        """
        # TODO: Implement this!
        t = 0
        K = self.nzw.shape[0]
        for wordFreq, docNum, wordNum in zip(X.data, X.row, X.col):
            topic = t % K
            self.ndz[docNum][topic]+= 1 # or is it += wordFreq |||| the number of words assigned to topic z in document d
            self.nzw[wordNum][topic]+= wordFreq # the number of times word w is assigned topic z
            self.nz[topic]+=1 # the number of times any word is assigned to topic z
            t +=1
        # raise Exception("You must implement this method!")


    def inference(self, *, X, iterations):
        """
        Sample topic assignments and use to determine document portions and
        topic-word distributions.

        Args:
            X: A compressed sparse row matrix of ints with shape
                [num_docs, num_words].
            iterations: int giving number of iterations
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")


    def _conditional(self, d, w):
        """
        Compute the posterior probability p(z=k|·).
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")


    def _loglikelihood(self):
        """
        Compute model likelihood: log p(w,z) = log p(w|z) + log p(z)
        You should call this method at the end of each iteration and append
        result to self.likelihoods.
        """
        ll = 0
        for k in range(self.num_topics):
            ll += np.sum(gammaln(self.nzw[k, :] + self.beta)) - gammaln(np.sum(self.nzw[k, :] + self.beta))
            ll -= self.num_words * gammaln(self.beta) - gammaln(self.num_words * self.beta)

        for d in range(self.num_docs):
            ll += np.sum(gammaln(self.ndz[d, :] + self.alpha)) - gammaln(np.sum(self.ndz[d, :] + self.alpha))
            ll -= self.num_topics * gammaln(self.alpha) - gammaln(self.num_topics * self.alpha)
        
        return ll


class SumProduct(Inference):

    def __init__(self, *, num_topics, num_docs, num_words, num_nonzero, alpha, beta):
        super().__init__(num_topics, num_docs, num_words, alpha, beta)
        self.mu_wd = np.zeros((num_nonzero, self.num_topics)) # mu_wd -- assignments for all words
        self.mu_negw_d = np.zeros((num_words, self.num_topics)) # mu_-w,d
        self.mu_w_negd = np.zeros((num_docs, self.num_topics)) # mu_w,-d
        self.mu_theta_to_z = np.zeros(self.num_topics) # message from factor node theta_d to variable node z_wd
        self.mu_phi_to_z = np.zeros(self.num_topics) # message from factor node phi_k to variable node z_wd


    def initialize(self, X):
        """
        Helper function to initialize z as described in Section 3.2.
        Call this function from inference.
        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_docs, num_words].
        """
        self.mu_wd = np.random.RandomState(seed=0).rand(X.nnz, self.num_topics)
        self.mu_wd /= self.mu_wd.sum(axis=1, keepdims = True)
        # TODO: Implement this!
        raise Exception("You must implement this method!")


    def inference(self, *, X, iterations):
        """
        Use message-passing to for topic assignments and use to determine 
        document portions and topic-word distributions.

        Make sure to transpose phi when setting to self.phi

        Args:
            X: A compressed sparse row matrix of ints with shape
                [num_docs, num_words].
            iterations: int giving number of clustering iterations
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")


    def _loglikelihood(self, X):
        """
        Compute model likelihood: log p(w,z) = log p(w|z) + log p(z)
        You should call this method at the end of each iteration and append
        result to self.likelihoods.
        """
        docs, words, data = X.row, X.col, X.data.astype(float)
        ll = 0
        for k in range(self.num_topics):
            for d, w in zip(docs, words):
                wrows = words[np.where(docs == d)[0]]
                drows = docs[np.where(words == w)[0]]
                ll += gammaln(data[wrows] @ self.mu_wd[wrows, k] + self.alpha) - gammaln(np.sum(data[wrows] @ self.mu_wd[wrows] + self.alpha))
                ll += gammaln(data[drows] @ self.mu_wd[drows, k] + self.beta) - gammaln(np.sum(data @ self.mu_wd + self.beta))

        return ll
