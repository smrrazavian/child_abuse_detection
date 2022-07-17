#!/bin/bash
find . -name 'coverage.txt' -delete
poetry run pytest --cov-report term --cov child_abuse_detection tests/ >>.logs/coverage.txt
