"""stopword module for project."""
from typing import Any, List

import os
from html import unescape
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from hazm import SentenceTokenizer, WordTokenizer, stopwords_list


def path_finder(filename: str) -> str:
    """Returns a file path.
    Args:
        filename (str):
            gets the filename.
    Returns:
        file_path (str):
            file's path.
    """
    try:
        cwd = Path.cwd()
        file_path = os.path.join((cwd / "./assets/").resolve(), filename + ".csv")
        return file_path
    except:
        raise ValueError("Incorrect file name.")


def pars_csv(filename: str) -> pd.DataFrame:
    """Parsing csv files for project.
    Args:
        filename (str):
            gets a csv filename in assets directory
    Returns:
        df_data (pd.DataFrame):
            Pandas data-frame of the given csv file
    """
    file_path = path_finder(filename)
    is_abuse_dict = {
        "ChildAbuse": 1,
        "SampleNews": 0,
    }
    # --------Making data-frame from csv file--------
    df_data = pd.read_csv(file_path)
    df_data = df_data.dropna(how="any", axis=0)
    df_data.drop("id", axis=1, inplace=True)
    df_data.reset_index(inplace=True, drop=True)
    df_data.drop_duplicates(inplace=True)
    df_data = df_data[(df_data.content != "\n")]
    df_data["is_childAbuse"] = [is_abuse_dict[filename]] * len(df_data)
    df_data.drop(columns=["title"], inplace=True)
    df_data.reset_index(inplace=True, drop=True)
    return df_data


def clean_text(df_data: pd.DataFrame) -> pd.DataFrame:
    """Deletes stopwords, punctuations and digits of text.
    Args:
        df_data (pd.DataFrame):
            a data-frame of newses.
    Returns:
        df_data (pd.DataFrame):
            cleaned data-frame of df-data.
    """
    without_stop_words = []  # TODO Ask if is it efficient to create new list?
    contents = df_data["content"]
    for text in contents:
        soup = BeautifulSoup(text, features="html.parser")
        text = soup.get_text()
        unescaped_text = unescape(text)
        non_unicode_text = unescaped_text.replace("\u200c", " ")
        non_unicode_text = non_unicode_text.replace("\xa0", " ")
        non_unicode_text = non_unicode_text.replace("\u200e", "")
        non_digit_text = "".join(filter(lambda x: not x.isdigit(), non_unicode_text))
        puncts = "!\"#%'()*+,-./:;<=>?@\\[\\]^_`{|}~’”“′‘\\\\]؟؛«»،٪"
        non_puncts_text = non_digit_text.translate(str.maketrans("", "", puncts))
        stop_words = set(stopwords_list())
        non_stopword_text = " ".join(
            [word for word in non_puncts_text.split() if word not in stop_words]
        )
        without_stop_words.append(non_stopword_text)
    df_data["content"] = without_stop_words
    return df_data


def tokenize_text(texts: List[str], tokenize_type: str = "word") -> Any:
    """Tokenize text based on tokenize_type(word or sentences).
    Args:
        texts (List[str]):
            a List of sentences of the corpus.
        tokenizeType (str):
            Determines the tokenization type.
    Returns:
        tokenized_texts (Any):
            tokenized form of sentences.
    """
    type_dict = {"word": WordTokenizer(), "sentence": SentenceTokenizer()}
    tokenizer = type_dict[tokenize_type]
    tokenized_texts = []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        tokenized_texts.append(tokenized_text)
    return tokenized_texts
