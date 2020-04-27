import numpy as np
import scipy.sparse as sparse
import os
from collections import defaultdict


def _read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])


def _clean_text(raw_text, stopwords):
    terms = []
    for line in raw_text:
        line = line.strip()
        for word in line.split():
            if '_meta_' not in word and word not in stopwords:
                terms.append(word)
    return terms


def build_dtm(datadir, num_docs):
    files = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f)) and f != 'common_words.txt']
    stopwords = _read_stopwords(os.path.join(datadir, 'common_words.txt'))
    if num_docs == 0:
        num_docs = len(files)

    docs = defaultdict(list)
    for i, file in enumerate(files[:num_docs], start=1):
        with open(os.path.join(datadir, file)) as f:
            docs['doc' + str(i)] = _clean_text(f.readlines(), stopwords)

    n_nonzero = 0
    vocab = set()
    for docterms in docs.values():
        unique_terms = set(docterms)
        vocab |= unique_terms
        n_nonzero += len(unique_terms)

    docnames = np.array(list(docs.keys()))
    vocab = np.array(list(vocab))  

    vocab_sorter = np.argsort(vocab)
    ndocs, nvocab = len(docnames), len(vocab)

    data = np.empty(n_nonzero, dtype=np.intc)
    rows = np.empty(n_nonzero, dtype=np.intc)
    cols = np.empty(n_nonzero, dtype=np.intc)

    ind = 0
    for docname, terms in docs.items():
        term_indices = vocab_sorter[np.searchsorted(vocab, terms, sorter=vocab_sorter)]

        uniq_indices, counts = np.unique(term_indices, return_counts=True)
        n_vals = len(uniq_indices)
        ind_end = ind + n_vals

        data[ind:ind_end] = counts
        cols[ind:ind_end] = uniq_indices
        doc_idx = np.where(docnames == docname)
        rows[ind:ind_end] = np.repeat(doc_idx, n_vals)

        ind = ind_end

    dtm = sparse.coo_matrix((data, (rows, cols)), shape=(ndocs, nvocab), dtype=np.intc)

    return dtm, vocab
