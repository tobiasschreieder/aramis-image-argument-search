import logging
from pathlib import Path
from typing import Dict

import numpy as np
from joblib import Parallel, delayed

from config import Config
from indexing.data_entry import DataEntry
from indexing.preprocessing import Preprocessor, SpacyPreprocessor
from .term_index import TermIndex

cfg = Config.get()


class TopicQueryTermIndex(TermIndex):
    log = logging.getLogger('TopicQueryIndex')

    @classmethod
    def create_index(cls, prep: Preprocessor = SpacyPreprocessor(), **kwargs) -> 'TermIndex':
        """
        Create in index object from the stored data.
        If max_images is < 1 use all images found else stop after max_images.

        :param prep: Preprocessor to use, default SpacyPreprocessor
        :return: An index object
        """
        index = cls()
        index.prep = prep

        ids = DataEntry.get_image_ids()
        topic_queries = {}
        for image in ids:
            entry = DataEntry.load(image)
            for page in entry.pages:
                for rank in page.rankings:
                    if rank.topic in topic_queries.keys():
                        topic_queries[rank.topic].add(rank.query)
                    else:
                        topic_queries[rank.topic] = {rank.query}

        cls.log.debug('create index')
        index.document_ids = np.array(sorted(topic_queries.keys()))

        for topic_id in index.document_ids:
            query_set = topic_queries[topic_id]
            query_str = ''
            for q in query_set:
                query_str += ' ' + q
            topic_queries[topic_id] = query_str

        with Parallel(n_jobs=-2, verbose=2) as parallel:
            doc_terms = parallel(delayed(prep.preprocess)(topic_queries[doc_id]) for doc_id in index.document_ids)

        index.index_terms = np.array(list({term for terms in doc_terms for term in terms}))
        index.num_docs = index.document_ids.shape[0]
        index.num_terms = index.index_terms.shape[0]

        # Build the document-term matrix
        cls.log.debug('build doc-term matrix')

        index.inverted = index._build_matrix_parallel(doc_terms).transpose()

        return index

    def save(self) -> None:
        """
        Saves the object in a file.

        :return: None
        """
        self._save(cfg.working_dir.joinpath(Path('topic_query_index_{}.npz'.format(self.prep.get_name()))))

    @classmethod
    def load(cls, prep_name: str = SpacyPreprocessor.get_name(), **prep_kwargs) -> 'TermIndex':
        """
        Loads an index from a file.

        :param prep_name:
        :return: Index object loaded from file
        :raise ValueError: if file for index with number of indexed images doesn't exists
        """
        file = cfg.working_dir.joinpath(Path('topic_query_index_{}.npz'.format(prep_name)))

        if not file.exists():
            raise ValueError('No saved topic query index with {} preprocessor'
                             .format(prep_name))

        return cls._load(file, prep_name, **prep_kwargs)


class TopicTermIndex(TermIndex):
    log = logging.getLogger('TopicIndex')

    topic_id: int

    @classmethod
    def create_index(cls, topic_id: int = 1, prep: Preprocessor = SpacyPreprocessor(), **kwargs) -> 'TermIndex':
        """
        Create in index object from the stored data.

        :param topic_id: The topic id for this index
        :param prep: Preprocessor to use, default SpacyPreprocessor
        :return: An index object
        """
        cls.log.debug('create topic index for topic {}'.format(topic_id))
        index = cls()
        index.prep = prep
        index.topic_id = topic_id
        index.log = logging.getLogger('TopicIndex {}'.format(topic_id))

        ids = DataEntry.get_image_ids()
        topic_queries = {}
        for image in ids:
            entry = DataEntry.load(image)
            for page in entry.pages:
                for rank in page.rankings:
                    if rank.topic == topic_id:
                        if image in topic_queries.keys():
                            topic_queries[image].append(rank)
                        else:
                            topic_queries[image] = [rank]

        cls.log.debug('create index')
        index.document_ids = np.array(sorted(topic_queries.keys()))

        doc_terms = index._gen_doc_terms_parallel()

        index.index_terms = np.array(list({term for terms in doc_terms for term in terms}))
        index.num_docs = index.document_ids.shape[0]
        index.num_terms = index.index_terms.shape[0]

        # Build the document-term matrix
        cls.log.debug('build doc-term matrix')

        index.inverted = index._build_matrix_parallel(doc_terms).transpose()

        return index

    def save(self) -> None:
        """
        Saves the object in a file.

        :return: None
        """
        self._save(cfg.working_dir.joinpath(
            Path('topic_index/t_index_{}_{}.npz'.format(self.topic_id, self.prep.get_name()))))

    @classmethod
    def load(cls, topic_id: int = 1, prep_name: str = SpacyPreprocessor.get_name(), **prep_kwargs) -> 'TermIndex':
        """
        Loads an index from a file.

        :param topic_id: The topic id for the loaded index
        :param prep_name:
        :return: Index object loaded from file
        :raise ValueError: if file for index with number of indexed images doesn't exists
        """
        file = cfg.working_dir.joinpath(Path('topic_index/t_index_{}_{}.npz'.format(topic_id, prep_name)))

        if not file.exists():
            raise ValueError('No saved topic query index for topic {} with {} preprocessor'
                             .format(topic_id, prep_name))

        loaded = cls._load(file, prep_name, **prep_kwargs)
        loaded.topic_id = topic_id
        loaded.log = logging.getLogger('TopicIndex {}'.format(topic_id))
        return loaded


def get_all_topic_indexes() -> Dict[int, TopicTermIndex]:
    indexes = {}
    for i in range(1, 51):
        try:
            indexes[i] = TopicTermIndex.load(i)
        except ValueError:
            # 7h
            indexes[i] = TopicTermIndex.create_index(i)
            indexes[i].save()
    return indexes
