"""Vectorization module."""
from typing import Any

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# TODO:
#  Config countVectorizer & tfidfVectorizer. Complete word2vec.


# def word2vec(text: List[str]) -> np.array:
#     """Creates word2vec vector.
#     Args:
#         text (List[str]):
#     Returns:

#     """
#     w2v_model = Word2Vec(sentences=text, alpha=0.05, min_alpha=0.0007)
#     w2v_model.save("word2vec.model")
#     w2v_model.train(
#         text, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1
#     )
#     return w2v_model.load


def count_vectorizer(df_data: pd.DataFrame) -> Any:
    """Convert a collection of text documents to a matrix of token counts bases on words count.
    Args:
        df_data (pd.DataFrame):
            data-frame of newses.
    Returns:
        text_vector (Any):
            vector of content column of df_data.
    """
    content = df_data["content"]
    vectorizer = CountVectorizer()
    text_vector = vectorizer.fit_transform(content).toarray()
    return text_vector


def tfidf_vectorizer(df_data: pd.DataFrame) -> Any:
    """Convert a collection of raw documents to a matrix of TF-IDF features.
    Args:
        df_data (pd.DataFrame):
            data-frame of newses.
    Returns:
        text_vector (Any):
            vector of content column of df_data.
    """
    content = df_data["content"]
    vectorizer = TfidfVectorizer()
    text_vector = vectorizer.fit_transform(content).toarray()
    return text_vector
