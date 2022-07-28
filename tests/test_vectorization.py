"""Test for vectorization module."""
import numpy as np

from child_abuse_detection.text_utils import clean_text, pars_csv
from child_abuse_detection.vectorization import count_vectorizer, tfidf_vectorizer


def test_count_vectorizer():
    """Test for count vectorizer."""
    data = pars_csv("ChildAbuse")
    clean_data = clean_text(data)
    vector = count_vectorizer(clean_data)
    assert isinstance(vector, np.ndarray)


def test_tfidf_vectorizer():
    """Test for tfidf vectorizer."""
    data = pars_csv("ChildAbuse")
    clean_data = clean_text(data)
    vector = tfidf_vectorizer(clean_data)
    assert isinstance(vector, np.ndarray)
