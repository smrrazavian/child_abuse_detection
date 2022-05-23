"""stopword module for project."""
from typing import List

import os
from html import unescape
from pathlib import Path

from bs4 import BeautifulSoup
from hazm import stopwords_list


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
            nonUnicodeText = nonUnicodeText.replace("\xa0", "")
            nonUnicodeText = nonUnicodeText.replace("\u200e", "")
            texts.append(unescapedText)
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
        text = text.lower()
        stopWords = set(stopwords_list())
        text = "".join([word for word in text.split() if word not in stopWords])
        withoutStopWords.append(text)
    return withoutStopWords
