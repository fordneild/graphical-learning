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
        data = X.data
        print('data',data, data.shape, type(data), 'hello?')
        for wordFreq, docNum, wordNum in zip(X.data, X.row, X.col):
            topic = t % K
            self.topics[(wordNum, docNum)] = topic
            self.ndz[docNum][topic]+= 1 # the number of words assigned to topic z in document d
            self.nzw[topic][wordNum]+= 1 # the number of times word w is assigned topic z
            self.nz[topic]+= 1 # the number of times any word is assigned to topic z
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
        self.initialize(X)
        for _ in range(iterations):
            for wordFreq, docNum, wordNum in zip(X.data, X.row, X.col):
                topic = self.topics[(wordNum, docNum)]
                self.ndz[docNum][topic] -= 1 if self.ndz[docNum][topic] else 0
                self.nzw[topic][wordNum] -= 1 if self.nzw[topic][wordNum] else 0
                self.nz[topic] -= 1 if self.nz[topic] else 0
                posterior = self._conditional(docNum, wordNum)
                # numpy sample from distribution using dirichlet function
                # which of these?
                #   https://numpy.org/doc/1.18/reference/random/generated/numpy.random.dirichlet.html?highlight=dirichlet#numpy.random.dirichlet
                #   https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html?highlight=random%20multinomial#numpy.random.multinomial
                # sample = np.random.dirichlet(posterior)
                numExperiments = 1 # ? what is this
                # print('post', posterior, np.sum(posterior))
                sample = np.random.multinomial(numExperiments, posterior)
                # then argmax to get the index of whatever has 1 in it
                # set topic to that topic
                topic = np.argmax(sample)
                # increment counters using that topic
                self.ndz[docNum][topic] += 1
                self.nzw[topic][wordNum] += 1
                self.nz[topic] += 1
            # compute log-likelihood
            self.loglikelihoods.append(self._loglikelihood())
            print('iteration', len(self.loglikelihoods), self.loglikelihoods[-1])
        # compute theta
        top = np.add(self.ndz, self.alpha)
        bot = np.sum(np.add(self.ndz, self.alpha), 1)
        # print('bot.shape',bot.shape)
        bot = np.transpose(np.tile(bot, (top.shape[1], 1)))
        
        self.theta = np.divide(top, bot)
        # print('theta',self.theta, self.theta.shape)
        # compute phi
        top = np.add(self.nzw, self.beta)
        # print('top.shape', top.shape)
        bot = np.sum(np.add(self.nzw, self.beta), 1)
        # print('bot.shape1',bot.shape)
        bot = np.transpose(np.tile(bot, (top.shape[1], 1)))
        self.phi = np.divide(top, bot)
        
        
        # print('phi',self.phi, self.phi.shape)



    def _conditional(self, d, w):
        """
        Compute the posterior probability p(z=k|·).
        """
        # TODO: Implement this!
        topic = self.topics[(w, d)]
        # ndz = np.add(np.delete(self.ndz, topic, 1), self.alpha)
        # nzw = np.add(np.delete(self.nzw, topic, 0), self.beta)
        ndz = np.add(self.ndz[d], self.alpha)
        nzw = np.add(self.nzw[:, w], self.beta)
        top = np.multiply(ndz, nzw)
        bot = np.sum(np.add(self.nzw, self.beta), 1)
        raw = np.divide(top, bot).astype('float64')
        # normalize
        normalized = np.divide(raw, np.sum(raw))
        return normalized
        


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
