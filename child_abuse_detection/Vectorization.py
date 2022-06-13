"""Vectorization module."""
import numpy as np
from gensim import models
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from child_abuse_detection import stopword


def countVectorizer(text) -> np.array:
    """
    Args:

    Returns:

    """
    vectorizer = CountVectorizer()
    textVector = vectorizer.fit_transform(text).toarray()
    return textVector


def tfidfVectorizer(text):
    vectorizer = TfidfVectorizer()
    textVector = vectorizer.fit_transform(text).toarray()
    return textVector
