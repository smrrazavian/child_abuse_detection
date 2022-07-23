"""Unit tests for stopword module."""
import pytest

from child_abuse_detection.text_utils import path_finder


def test_path_finder():
    """Test path_finder"""
    path = path_finder("ChildAbuse")
    assert (
        path
        == "/home/smrrazavian/Documents/Personal-Projects/child_abuse/child_abuse_detection/assets/ChildAbuse.csv"
    )


def test_path_finder_fail():
    """Test if path_finder is failing"""
    with pytest.raises(Exception) as error:
        path_finder("childabuse")
        assert error.value.args[0] == "Incorrect file name."
