import logging
from pathlib import Path
from typing import AnyStr, Hashable, List

import numpy as np

from .data_entry import DataEntry
from .preprocessing import Preprocessor

log = logging.getLogger('index')


class Index:

    document_ids: np.ndarray
    index_terms: np.ndarray
    num_docs: np.ndarray
    num_terms: np.ndarray
    inverted: np.ndarray

    @classmethod
    def create_index(cls, max_images: int = -1) -> 'Index':
        """
        Create in index object from the stored data.
        If max_images is < 1 use all images found else stop after max_images.

        :param max_images: Number to determine the maximal number of images to index
        :return: An index object
        """
        index = cls()

        log.debug('create index with max_images %s', max_images)
        index.document_ids = np.array(DataEntry.get_image_ids(max_size=max_images))
        prep = Preprocessor()
        doc_terms = dict()

        # TODO: parallel execution for preprocessing
        for doc_id in index.document_ids:
            log.debug('Process %s', doc_id)
            data = DataEntry.load(doc_id)
            tokens = []
            for page in data.pages:
                tokens += prep.preprocess_file(page.snapshot_path.joinpath('text.txt'))
            doc_terms[doc_id] = tokens
        index.index_terms = np.array(list({term for terms in doc_terms.values() for term in terms}))
        index.num_docs = index.document_ids.shape[0]
        index.num_terms = index.index_terms.shape[0]

        log.debug('build doc-term matrix')
        # Build the document-term matrix
        X = np.zeros(shape=(index.num_docs, index.num_terms), dtype=np.int32)
        for i in range(index.num_docs):
            for j in range(index.num_terms):
                X[i, j] = doc_terms[index.document_ids[i]].count(index.index_terms[j])

        # Build the inverted index based on the document-term matrix
        # Note that since we dont do any compression or sparse matrices,
        # this is just the transposed document-term matrix
        index.inverted = X.transpose()

        return index

    def save(self) -> None:
        """
        Saves the object in a file.

        :return: None
        """
        log.debug('save index to file')
        Path('index').mkdir(exist_ok=True)
        np.savez_compressed(Path('index/index_{}'.format(self.inverted.shape[1])), inverted=self.inverted,
                            index_terms=self.index_terms, doc_ids=self.document_ids)
        log.debug('Done')

    @classmethod
    def load(cls, indexed_images: int) -> 'Index':
        """
        Loads an index from a file.

        :param indexed_images: number of indexed images in saved index
        :return: Index object loaded from file
        :raise ValueError: if file for index with number of indexed images doesn't exists
        """
        file = Path('index/index_{}.npz'.format(indexed_images))

        if not file.exists():
            raise ValueError('No saved index with {} indexed images'.format(indexed_images))

        log.debug('Load index from file %s', file)
        loaded = np.load(file)
        index = cls()
        index.document_ids = loaded['doc_ids']
        index.index_terms = loaded['index_terms']
        index.num_docs = index.document_ids.shape[0]
        index.num_terms = index.index_terms.shape[0]
        index.inverted = loaded['inverted']

        log.debug('Done')
        return index

    def get_term_frequency(self, term: AnyStr, doc_id: Hashable) -> int:
        """
        Returns the term frequency for a specified term and document
        :param term: term to return the frequency for
        :param doc_id: document to return the frequency for
        :return: term frequency
        """
        i = np.where(self.index_terms == term)[0]
        j = np.where(self.document_ids == doc_id)[0]
        if i.shape != (1,) or j.shape != (1,):
            return 0
        else:
            return self.inverted[i, j]

    def get_total_term_frequency(self, term: AnyStr) -> int:
        """
        Returns the total number of occurrences of a term across all documents
        :param term: term to return the frequency for
        :return: total term frequency
        """
        i = np.where(self.index_terms == term)[0]
        if i.shape != (1,):
            return 0
        else:
            return np.sum(self.inverted[i])

    def get_document_frequency(self, term: AnyStr) -> int:
        """
        Returns the number of documents that contain the given term at least once
        :param term: term to return the frequency for
        :return: document frequency
        """
        i = np.where(self.index_terms == term)
        if i.shape != (1,):
            return 0
        else:
            return np.count_nonzero(self.inverted[i])

    def get_document_count(self):
        """
        Returns the number of documents in the index
        :return: number of indexed documents
        """
        return self.num_docs

    def get_document_length(self, doc_id: Hashable):
        """
        Returns the number of tokens in a document (including multiple occurences of the same token)
        :param doc_id: document to count tokens for
        :return: total number of token occurences in the specified document
        """
        j = np.where(self.document_ids == doc_id)[0]
        if j.shape != (1,):
            return 0
        else:
            return np.sum(self.inverted[:, j])

    def get_total_document_length(self):
        """
        Returns the total number of tokens in the collection
        :return: total number of token occurences in the collection
        """
        return np.sum(self.inverted)

    def get_term_count(self):
        """
        Returns the number of terms in the index
        :return: number of indexed terms
        """
        return self.num_terms

    def get_document_ids(self) -> List:
        """
        Returns a list of IDs for all documents in the index
        :return:
        """
        return list(self.document_ids)

    def get_index_terms(self) -> List:
        """
        Returns a list of all indexed terms
        :return:
        """
        return list(self.index_terms)
