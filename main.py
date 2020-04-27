""" Main file. This is the starting point for your code execution.

You shouldn't need to change anything in this code.
"""

import os
import argparse as ap
import pickle
import numpy as np

import models
from data import build_dtm


def get_args():
    p = ap.ArgumentParser(description="This is the main test harness for your models.")

    # Meta arguments
    p.add_argument("--data", type=str, help="Data directory")
    p.add_argument("--predictions-file", type=str,
                    help="Where to dump predictions")
    p.add_argument("--inference", type=str,
                   choices=['gibbs-sampling', 'sum-product'],
                    help="The type of model to use.")
    p.add_argument("--num-documents", type=int, help="Number of documents to include in document-word matrix.", default=100)
    p.add_argument("--top-k", type=int, help="Top k words per topic to be outputted.", default=10)

    # Model Hyperparameters
    p.add_argument("--num-topics", type=int,
                        help="The number of topics", default=10)
    p.add_argument("--alpha", type=float,
                        help="Dirichlet parameter alpha", default=0.1)
    p.add_argument("--beta", type=float,
                        help="Dirichlet parameter beta", default=0.01)
    p.add_argument("--iterations", type=int,
                        help="Number of training iterations", default=100)
    return p.parse_args()


def check_args(args):
    mandatory_args = {'data', 'inference', 'predictions_file', 'num_topics', 'iterations',
                      'alpha', 'beta', 'num_documents', 'top_k'}
    if not mandatory_args.issubset(set(dir(args))):
        raise Exception("You're missing essential arguments!"
                         "We need these to run your code.")

    if args.inference is None:
        raise Exception("--inference method should be specified")
    if args.predictions_file is None:
        raise Exception("--predictions-file should be specified")
    if args.data is None:
        raise Exception("--data directory should be specified")
    elif not os.path.exists(args.data):
        raise Exception("data directory specified by --data does not exist.")
    

def main(args):
    """ 
    Fit a model's parameters given the parameters specified in args.
    """
    # Load the data.
    X, vocab = build_dtm(args.data, args.num_documents)

    # build the appropriate model
    if args.inference.lower() == 'gibbs-sampling':
        inference = models.GibbsSampling(num_topics=args.num_topics, num_docs=X.shape[0], 
            num_words=X.shape[1], alpha=args.alpha, beta=args.beta)
    elif args.inference.lower() == 'sum-product':
        inference = models.SumProduct(num_topics=args.num_topics, num_docs=X.shape[0], 
            num_words=X.shape[1], num_nonzero=X.nnz, alpha=args.alpha, beta=args.beta)
    else:
        raise Exception('The method given by --inference is not yet supported.')

    # Run the inference method
    model = models.LDA(inference=inference)
    model.fit(X=X, iterations=args.iterations)

    # predict topic assignments for words in corpus
    preds = model.predict(vocab=vocab, K=args.top_k)

    # output model predictions
    with open(args.predictions_file, 'w') as file:
        for pred in preds:
            file.write(pred + '\n')

if __name__ == "__main__":
    args = get_args()
    check_args(args)
    main(args)
