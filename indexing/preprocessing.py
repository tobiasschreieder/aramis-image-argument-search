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
        tokens = []
        log.debug('Preprocess file %s', file)
        with file.open(encoding='UTF8') as html_text:
            for line in html_text:
                tokens += [token.lemma_.lower() for token in self.nlp(line) if not (token.is_stop or token.is_punct)]
        return tokens
