""" 
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np
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
        self.topics = {}


    def initialize(self, X):
        """
        Helper function to initialize z as described in Section 2.3.
        Call this function from inference.
        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_docs, num_words].
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")


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
        # TODO: Implement this!
        raise Exception("You must implement this method!")


    def inference(self, *, X, iterations):
        """
        Use message-passing to for topic assignments and use to determine 
        document portions and topic-word distributions.

        Args:
            X: A compressed sparse row matrix of ints with shape
                [num_docs, num_words].
            iterations: int giving number of clustering iterations
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")


    def _loglikelihood(self):
        """
        This code will be provided to you soon.

        Compute model likelihood: log p(w,z) = log p(w|z) + log p(z)
        You should call this method at the end of each iteration and append
        result to self.likelihoods.
        """
        pass
