import logging
from pathlib import Path
from typing import Dict

import numpy as np
from joblib import Parallel, delayed

from config import Config
from indexing.data_entry import DataEntry, Topic
from indexing.preprocessing import Preprocessor, SpacyPreprocessor
from .term_index import TermIndex

cfg = Config.get()


class TopicQueryTermIndex(TermIndex):
    log = logging.getLogger('TopicQueryIndex')

    @classmethod
    def create_index(cls, prep: Preprocessor = SpacyPreprocessor(), n_jobs: int = -2,
                     max_images: int = -1, **kwargs) -> 'TermIndex':
        """
        Create in index object from the stored data.
        If max_images is < 1 use all images found else stop after max_images.

        :param max_images: TODO
        :param n_jobs: TODO
        :param prep: Preprocessor to use, default SpacyPreprocessor
        :return: An index object
        """
        index = cls()
        index.prep = prep

        ids = DataEntry.get_image_ids(max_images)
        topic_queries = {}
        for i, image in enumerate(ids):
            entry = DataEntry.load(image)
            for page in entry.pages:
                for rank in page.rankings:
                    if rank.topic in topic_queries.keys():
                        topic_queries[rank.topic].add(rank.query)
                    else:
                        topic_queries[rank.topic] = {rank.query}
            if i % 100 == 0:
                cls.log.debug('Done with %s/%s', i, len(ids))

        cls.log.debug('create index')
        index.document_ids = np.array(sorted(topic_queries.keys()))

        for topic_id in index.document_ids:
            query_set = topic_queries[topic_id]
            query_str = ''
            for q in query_set:
                query_str += ' ' + q
            topic_queries[topic_id] = query_str

        with Parallel(n_jobs=n_jobs, verbose=2) as parallel:
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
    def create_index(cls, topic_id: int = 1, prep: Preprocessor = SpacyPreprocessor(),
                     n_jobs: int = -2, max_images: int = -1, **kwargs) -> 'TermIndex':
        """
        Create in index object from the stored data.

        :param max_images: TODO
        :param n_jobs: TODO
        :param topic_id: The topic id for this index
        :param prep: Preprocessor to use, default SpacyPreprocessor
        :return: An index object
        """
        cls.log.debug('create topic index for topic {}'.format(topic_id))
        index = cls()
        index.prep = prep
        index.topic_id = topic_id
        index.log = logging.getLogger('TopicIndex {}'.format(topic_id))

        # ids = DataEntry.get_image_ids(max_images)
        # topic_queries = {}
        # for i, image in enumerate(ids):
        #     entry = DataEntry.load(image)
        #     for page in entry.pages:
        #         for rank in page.rankings:
        #             if rank.topic == topic_id:
        #                 if image in topic_queries.keys():
        #                     topic_queries[image].append(rank)
        #                 else:
        #                     topic_queries[image] = [rank]
        #     if i % 1000 == 0:
        #         cls.log.debug('Done with %s/%s', i, len(ids))

        # doc_ids = np.array(sorted(topic_queries.keys()))
        doc_ids = np.array(sorted(Topic.get(topic_id).get_image_ids()))

        cls.log.debug('Done with all ids, found %s id for Topic %s', len(doc_ids), topic_id)
        cls.log.debug('create index')
        index.document_ids = doc_ids

        doc_terms = index._gen_doc_terms_parallel(n_jobs=n_jobs)

        cls.log.debug('Done with doc term generation')
        index.index_terms = np.array(list({term for terms in doc_terms for term in terms}))
        index.num_docs = index.document_ids.shape[0]
        index.num_terms = index.index_terms.shape[0]

        # Build the document-term matrix
        cls.log.debug('build doc-term matrix')

        index.inverted = index._build_matrix_parallel(doc_terms, n_jobs=n_jobs).transpose()

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


def get_all_topic_indexes(n_jobs: int = -2, max_images: int = -1,
                          force_create: bool = False) -> Dict[int, TopicTermIndex]:
    indexes = {}
    for topic in Topic.load_all():
        create = force_create
        if not create:
            try:
                indexes[topic.number] = TopicTermIndex.load(topic.number)
            except ValueError:
                create = True

        if create:
            # 7h
            indexes[topic.number] = TopicTermIndex.create_index(topic.number, n_jobs=n_jobs, max_images=max_images)
            indexes[topic.number].save()

    return indexes
