"""stopword module for project."""
from typing import Any, List

import os
from html import unescape
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from hazm import SentenceTokenizer, WordTokenizer, stopwords_list


def ParsCSV(filename: str) -> pd.DataFrame:
    """Parsing csv files for project.
    Args:

    Returns:

    """
    cwd = Path.cwd()
    filePath = os.path.join((cwd / "./assets/").resolve(), filename + ".csv")
    # --------Making data-frame from csv file--------
    df_data = pd.read_csv(filePath)
    df_data = df_data.dropna(how="any", axis=0)
    df_data.drop("id", axis=1, inplace=True)
    df_data.reset_index(inplace=True, drop=True)
    for i in range(0, len(df_data)):
        soup = BeautifulSoup(df_data["content"][i], features="html.parser")
        # tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "xcms", "xcms:video"]
        text = soup.get_text()
        unescapedText = unescape(text)
        nonUnicodeText = unescapedText.replace("\u200c", " ")
        nonUnicodeText = nonUnicodeText.replace("\xa0", " ")
        nonUnicodeText = nonUnicodeText.replace("\u200e", "")
        df_data["content"][i] = nonUnicodeText
    return df_data


def textCleaner(texts: List[str]) -> List[str]:
    # TODO Union type checking
    """
    Args:
        texts (List[str]):
            a list of newses.
    Returns:
        withoutStopWords (List[str]):
            a list of newses without StopWords.
    """
    withoutStopWords = []
    for text in texts:
        text = "".join(filter(lambda x: not x.isdigit(), text))
        puncts = "!\"#%'()*+,-./:;<=>?@\\[\\]^_`{|}~’”“′‘\\\\]؟؛«»،٪"
        text = text.translate(str.maketrans("", "", puncts))
        stopWords = set(stopwords_list())
        text = " ".join([word for word in text.split() if word not in stopWords])
        withoutStopWords.append(text)
    return withoutStopWords


def textTokenizer(texts: List[str], tokenizeType: str = "word") -> Any:
    """
    Args:
        texts (List[str]):
            a List of sentences of the corpus.
        tokenizeType (str):
            Determines the tokenization type.
    Returns:

    """
    typeDict = {"word": WordTokenizer(), "sentence": SentenceTokenizer()}
    tokenizer = typeDict[tokenizeType]
    tokenizedTexts = []
    for text in texts:
        tokenizedText = tokenizer.tokenize(text)
        tokenizedTexts.append(tokenizedText)
    return tokenizedTexts
