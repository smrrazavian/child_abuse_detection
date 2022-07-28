"""Unit tests for stopword module."""
import os
from pathlib import Path

import pandas as pd
import pytest

from child_abuse_detection.text_utils import clean_text, pars_csv, path_finder


def test_path_finder():
    """Test path_finder"""
    path = path_finder("ChildAbuse")
    cwd = Path.cwd()
    file_path = os.path.join((cwd / "./assets/").resolve(), "ChildAbuse" + ".csv")
    assert path == file_path


def test_path_finder_fail():
    """Test if path_finder is failing"""
    with pytest.raises(Exception) as error:
        path_finder("childabuse")
    assert error.value.args[0] == "Incorrect file name."


def test_pars_csv():
    """Test parsing csv files."""
    data_frame = pars_csv("SampleNews")
    assert isinstance(data_frame, pd.DataFrame)


def test_clean_text():
    """Test for clean text module."""
    data_frame = pars_csv("SampleNews")
    data_frame = clean_text(data_frame)
    assert isinstance(data_frame, pd.DataFrame)
