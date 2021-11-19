import logging
from pathlib import Path
from typing import List, Type

import spacy

from indexing import DataEntry

log = logging.getLogger('index preprocessing')


def get_preprocessor(name: str) -> Type['Preprocessor']:
    """
    Returns the preprocessor class for the given name.

    :param name: the name of the preprocessor
    :return: class of named preprocessor
    :raise ValueError: if name is not a preprocessor name
    """
    if name == 'Spacy':
        return SpacyPreprocessor
    else:
        raise ValueError('{} is not a preprocessor name'.format(name))


class Preprocessor:

    def __init__(self, **kwargs):
        pass

    def preprocess_doc(self, doc_id: str) -> List[str]:
        """
        Preprocesses the sources for a given document and
        returns the found tokens without stop words and punctuation.

        :param doc_id: The document id to process
        :return: List of stemmed tokens
        """
        return []

    def preprocess_file(self, file: Path) -> List[str]:
        """
        Preprocesses a text file and returns the found tokens without stop words and punctuation.

        :param file: The file to preprocess
        :return: List of stemmed tokens
        """
        return []

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocesses a text and returns the found tokens without stop words and punctuation.

        :param text: The text to process
        :return: List of stemmed tokens
        """
        return []

    @staticmethod
    def get_name() -> str:
        """
        Returns the name of this preprocessor class.

        :return: name of preprocessor
        """
        return 'none'


class SpacyPreprocessor(Preprocessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def get_name() -> str:
        return 'Spacy'

    def preprocess_doc(self, doc_id: str) -> List[str]:
        """
        Preprocesses the sources for a given document and
        returns the found tokens without stop words and punctuation.

        :param doc_id: The document id to process
        :return: List of stemmed tokens
        """
        data = DataEntry.load(doc_id)
        tokens = []
        for page in data.pages:
            tokens += self.preprocess_file(page.snp_text)
        return tokens

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
        Preprocesses a text and returns the found tokens without stop words and punctuation.

        :param text: The text to process
        :return: List of stemmed tokens
        """
        return [token.lemma_.lower() for token in self.nlp(text) if not (token.is_stop or token.is_punct)]
