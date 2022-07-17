"""Vectorization module."""
from typing import List

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from child_abuse_detection import stopword

# TODO:
#  Config countVectorizer & tfidfVectorizer. Complete word2vec.


def word2vec(text: List[str]) -> np.array:
    """
    Args:
        text (List[str]):
    Returns:

    """
    w2vModel = Word2Vec(sentences=text, alpha=0.05, min_alpha=0.0007)
    w2vModel.save("word2vec.model")
    w2vModel.train(
        text, total_examples=w2vModel.corpus_count, epochs=30, report_delay=1
    )
    return w2vModel.load


def countVectorizer(text: List[str]) -> np.array:
    """
    Args:
        text (List[str]):
    Returns:

    """
    vectorizer = CountVectorizer()
    textVector = vectorizer.fit_transform(text).toarray()
    return textVector


def tfidfVectorizer(text: List[str]) -> np.array:
    """
    Args:

    Returns:

    """
    vectorizer = TfidfVectorizer()
    textVector = vectorizer.fit_transform(text).toarray()
    return textVector
