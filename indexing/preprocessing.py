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
            return [token.lemma_.lower() for token in self.nlp(html_text.read())
                    if not (token.is_stop or token.is_punct)]
