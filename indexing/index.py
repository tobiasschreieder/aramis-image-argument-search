import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AnyStr, Hashable, List

import numpy as np
from joblib import Parallel, delayed

from utils import MemmappStore
from . import DataEntry
from .preprocessing import Preprocessor, SpacyPreprocessor, get_preprocessor


class Index(ABC):
    log = logging.getLogger('index')

    document_ids: np.ndarray
    index_terms: np.ndarray
    num_docs: np.ndarray
    num_terms: np.ndarray
    inverted: np.ndarray
    prep: Preprocessor

    @classmethod
    @abstractmethod
    def create_index(cls, prep: Preprocessor = SpacyPreprocessor(), **kwargs) -> 'Index':
        """
        Create in index object from the stored data.

        :param prep: Preprocessor to use, default SpacyPreprocessor
        :return: An index object
        """
        pass

    def _get_doc_terms_single(self) -> List[List[str]]:
        """
        Calculates the document terms in a single process.
        Returns a list of lists, where L[i] the list of tokens for the document self.document_ids[i] represents.

        :return: List of document terms
        """
        doc_terms = []

        for doc_id in self.document_ids:
            self.log.debug('Process %s', doc_id)
            data = DataEntry.load(doc_id)
            tokens = []
            for page in data.pages:
                tokens += self.prep.preprocess_file(page.snp_text)
            doc_terms.append(tokens)
        return doc_terms

    def _gen_doc_terms_parallel(self, n_jobs: int = -2) -> List[List[str]]:
        """
        Calculates the document terms in n_jobs process's.
        Returns a list of lists, where L[i] the list of tokens for the document self.document_ids[i] represents.

        :param n_jobs: the number of processes to use, if -1 use all,
            if < -1 use max_processes+1+n_jobs, example n_jobs = -2 -> use all processors except 1.
            see joblib.parallel.Parallel
        :return: List of document terms
        """
        with Parallel(n_jobs=n_jobs, verbose=2) as parallel:
            doc_terms = parallel(delayed(self.prep.preprocess_doc)(doc_id) for doc_id in self.document_ids)
        return doc_terms

    def _build_matrix_single(self, doc_terms: np.ndarray) -> np.ndarray:
        """
        Calculates the document x terms matrix in a single process.

        :return: the document x term matrix
        """
        x = np.zeros(shape=(self.num_docs, self.num_terms), dtype=np.int32)
        for i in range(self.num_docs):
            for j in range(self.num_terms):
                x[i, j] = doc_terms[i].count(self.index_terms[j])

        return x

    def _build_matrix_parallel(self, doc_terms: np.ndarray, n_jobs: int = -2) -> np.ndarray:
        """
        Calculates the document x terms matrix in n_jobs process's.

        :param n_jobs: the number of processes to use, if -1 use all,
            if < -1 use max_processes+1+n_jobs, example n_jobs = -2 -> use all processors except 1.
            see joblib.parallel.Parallel
        :return: the document x term matrix
        """
        def calc_term_frequencies(doc_number) -> np.ndarray:
            freq = np.zeros(shape=self.num_terms, dtype=np.int32)
            for j in range(self.num_terms):
                freq[j] = doc_terms[doc_number].count(self.index_terms[j])
            return freq

        with Parallel(n_jobs=n_jobs, verbose=2, max_nbytes=None) as parallel:
            y = parallel(delayed(calc_term_frequencies)(doc_number) for doc_number in range(self.num_docs))

        return np.array(y, dtype=np.int32)

    def _build_matrix_parallel_memmapping(self, doc_terms: np.ndarray, n_jobs: int = -2) -> np.ndarray:
        """
        Calculates the document x terms matrix in n_jobs process's. Using joblib's memmapping functions.

        :param n_jobs: the number of processes to use, if -1 use all,
            if < -1 use max_processes+1+n_jobs, example n_jobs = -2 -> use all processors except 1.
            see joblib.parallel.Parallel
        :return: the document x term matrix
        """
        mem_store = MemmappStore()
        mem_doc_terms = mem_store.store_in_memmap(doc_terms, 'doc_terms')
        mem_index_terms = mem_store.store_in_memmap(self.index_terms, 'index_terms')

        del doc_terms
        gc.collect()

        def calc_term_frequencies(doc_number) -> np.ndarray:
            freq = np.zeros(shape=self.num_terms, dtype=np.int32)
            for j in range(self.num_terms):
                freq[j] = mem_doc_terms[doc_number].count(mem_index_terms[j])
            return freq

        with Parallel(n_jobs=n_jobs, verbose=2, max_nbytes=None) as parallel:
            y = parallel(delayed(calc_term_frequencies)(doc_number) for doc_number in range(self.num_docs))

        x = np.array(y, dtype=np.int32)
        mem_store.cleanup()
        return x

    @abstractmethod
    def save(self, **kwargs) -> None:
        """
        Saves the object in a file.
        return x

        :return: None
        """
        pass

    def _save(self, file: Path) -> None:
        """
        Saves the object in a file.

        :return: None
        """
        self.log.debug('save index to file')
        file.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(file, inverted=self.inverted,
                            index_terms=self.index_terms, doc_ids=self.document_ids)
        self.log.debug('Done')

    @classmethod
    @abstractmethod
    def load(cls, **kwargs) -> 'Index':
        """
        Loads an index from a file.

        :return: Index object loaded from file
        """
        pass

    @classmethod
    def _load(cls, file: Path, prep_name: str = SpacyPreprocessor.get_name(), **prep_kwargs) -> 'Index':
        """
        Loads an index from a file.

        :param prep_name:
        :return: Index object loaded from file
        :raise ValueError: if file for index with number of indexed images doesn't exists
        """

        if not file.exists():
            raise ValueError('No saved index for file {}'.format(file))

        cls.log.debug('Load index from file %s', file)
        loaded = np.load(file)
        index = cls()
        index.document_ids = loaded['doc_ids']
        index.index_terms = loaded['index_terms']
        index.num_docs = index.document_ids.shape[0]
        index.num_terms = index.index_terms.shape[0]
        index.inverted = loaded['inverted']

        index.prep = get_preprocessor(prep_name)(**prep_kwargs)

        cls.log.debug('Done')
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
