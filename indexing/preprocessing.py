import logging
from pathlib import Path
from typing import List

import spacy

log = logging.getLogger('index preprocessing')


class Preprocessor:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_file(self, file: Path) -> List[str]:
        """
        Preprocesses a text file and returns the found tokens without stop words and punctuation.

        :param file: The file to preprocess
        :return: List of stemmed tokens
        """
        log.debug('Preprocess file %s', file)
        with file.open(encoding='UTF8') as html_text:
            return self.preprocess(html_text.read())

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocesses a text and returns the found tokens without stop words and punctuation

        :param text: The text to process
        :return: List of stemmed tokens
        """
        return [token.lemma_.lower() for token in self.nlp(text) if not (token.is_stop or token.is_punct)]
