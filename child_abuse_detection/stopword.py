"""stopword module for project."""
from typing import Any, List

import os
from html import unescape
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from hazm import SentenceTokenizer, stopwords_list


def csvParser(filename: str) -> List[str]:
    """Parsing csv files for project.
    Args:

    Returns:

    """
    cwd = Path.cwd()
    filePath = os.path.join((cwd / "./assets/").resolve(), filename + ".csv")
    with open(filePath) as file:
        soup = BeautifulSoup(file, features="html.parser", from_encoding="utf-8")
    texts = []
    for group in soup(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        for text in group.stripped_strings:
            unescapedText = unescape(text)
            nonUnicodeText = unescapedText.replace("\u200c", " ")
            nonUnicodeText = nonUnicodeText.replace("\xa0", " ")
            nonUnicodeText = nonUnicodeText.replace("\u200e", "")
            texts.append(nonUnicodeText)
    return texts


def textCleaner(texts: List[str]) -> List[str]:
    """
    Args:
        texts: a list of newses.
    Returns:
        withoutStopWords: a list of newses without StopWords.
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


def textTokenizer(texts: List[str]) -> Any:
    tokenizer = SentenceTokenizer()
    tokenizedTexts = []
    for text in texts:
        tokenizedText = tokenizer.tokenize(text)
        tokenizedTexts.append(tokenizedText)
    return tokenizedTexts
