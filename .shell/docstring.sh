#!/bin/bash
find . -name 'docstring.txt' -delete
poetry run python3 -m interrogate -v child_abuse_detection >>.logs/docstring.txt
poetry run python3 -m interrogate child_abuse_detection
