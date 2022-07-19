"""Vectorization module."""
from typing import List

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# TODO:
#  Config countVectorizer & tfidfVectorizer. Complete word2vec.


def word2vec(text: List[str]) -> np.array:
    """Creates word2vec vector.
    Args:
        text (List[str]):
    Returns:

    """
    w2v_model = Word2Vec(sentences=text, alpha=0.05, min_alpha=0.0007)
    w2v_model.save("word2vec.model")
    w2v_model.train(
        text, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1
    )
    return w2v_model.load


def count_vectorizer(text: List[str]) -> np.array:
    """
    Args:
        text (List[str]):
    Returns:

    """
    vectorizer = CountVectorizer()
    text_vector = vectorizer.fit_transform(text).toarray()
    return text_vector


def tfidf_vectorizer(text: List[str]) -> np.array:
    """
    Args:

    Returns:

    """
    vectorizer = TfidfVectorizer()
    text_vector = vectorizer.fit_transform(text).toarray()
    return text_vector
