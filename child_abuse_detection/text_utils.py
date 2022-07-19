"""stopword module for project."""
from typing import Any, List

import os
from html import unescape
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from hazm import SentenceTokenizer, WordTokenizer, stopwords_list


def pars_csv(filename: str) -> pd.DataFrame:
    """Parsing csv files for project.
    Args:
        filename (str):
            gets a csv filename in assets directory
    Returns:
        df_data (pd.DataFrame):
            Pandas data-frame of given csv file
    """
    cwd = Path.cwd()
    file_path = os.path.join((cwd / "./assets/").resolve(), filename + ".csv")
    # --------Making data-frame from csv file--------
    df_data = pd.read_csv(file_path)
    df_data = df_data.dropna(how="any", axis=0)
    df_data.drop("id", axis=1, inplace=True)
    df_data.reset_index(inplace=True, drop=True)
    for i in range(0, len(df_data)):
        soup = BeautifulSoup(df_data["content"][i], features="html.parser")
        text = soup.get_text()
        unescaped_text = unescape(text)
        non_unicode_text = unescaped_text.replace("\u200c", " ")
        non_unicode_text = non_unicode_text.replace("\xa0", " ")
        non_unicode_text = non_unicode_text.replace("\u200e", "")
        df_data["content"][i] = non_unicode_text
    return df_data


def clean_text(texts: List[str]) -> List[str]:
    # TODO Union type checking
    """Deletes stopwords, punctuations and digits of text.
    Args:
        texts (List[str]):
            a list of newses.
    Returns:
        withoutStopWords (List[str]):
            a list of newses without StopWords.
    """
    without_stop_words = []
    for text in texts:
        text = "".join(filter(lambda x: not x.isdigit(), text))
        puncts = "!\"#%'()*+,-./:;<=>?@\\[\\]^_`{|}~’”“′‘\\\\]؟؛«»،٪"
        text = text.translate(str.maketrans("", "", puncts))
        stop_words = set(stopwords_list())
        text = " ".join([word for word in text.split() if word not in stop_words])
        without_stop_words.append(text)
    return without_stop_words


def tokenize_text(texts: List[str], tokenize_type: str = "word") -> Any:
    """Tokenize text based on tokenize_type(word or sentences).
    Args:
        texts (List[str]):
            a List of sentences of the corpus.
        tokenizeType (str):
            Determines the tokenization type.
    Returns:

    """
    type_dict = {"word": WordTokenizer(), "sentence": SentenceTokenizer()}
    tokenizer = type_dict[tokenize_type]
    tokenized_texts = []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        tokenized_texts.append(tokenized_text)
    return tokenized_texts
