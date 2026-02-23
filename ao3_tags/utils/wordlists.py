import nltk
import os
from pandas import read_csv, read_parquet
from pathlib import Path

from ao3_tags import DATA_PATH, RESOURCE_PATH

WORDLIST_DIR = RESOURCE_PATH / 'wordlists'


def prepare_stopwords():
    # Load nltk stopwords
    try:
        stopwords = set(nltk.corpus.stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stopwords = set(nltk.corpus.stopwords.words('english'))

    # Load stopwords and characters collected by Wiegmann et al.
    with open(WORDLIST_DIR / 'ao3-characters.txt') as f:
        characters = set(f.read().split('\n'))
    with open(WORDLIST_DIR / 'ao3-stopwords.txt') as f:
        stopwords.update(set(f.read().split('\n')))
    with open(WORDLIST_DIR / 'terrier-stopwords.txt') as f:
        stopwords.update(set(f.read().split('\n')))

    return stopwords, characters