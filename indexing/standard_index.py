import logging
from pathlib import Path

import numpy as np

from .data_entry import DataEntry
from .index import Index
from .preprocessing import Preprocessor, SpacyPreprocessor


class StandardIndex(Index):
    log = logging.getLogger('StandardIndex')

    @classmethod
    def create_index(cls, max_images: int = -1, prep: Preprocessor = SpacyPreprocessor()) -> 'Index':
        """
        Create in index object from the stored data.
        If max_images is < 1 use all images found else stop after max_images.

        :param prep: Preprocessor to use, default SpacyPreprocessor
        :param max_images: Number to determine the maximal number of images to index
        :return: An index object
        """
        index = cls()
        index.prep = prep

        index.log.debug('create index with max_images %s', max_images)
        index.document_ids = np.array(sorted(DataEntry.get_image_ids(max_size=max_images)))

        doc_terms = index._gen_doc_terms_parallel()

        index.index_terms = np.array(list({term for terms in doc_terms for term in terms}))
        index.num_docs = index.document_ids.shape[0]
        index.num_terms = index.index_terms.shape[0]

        # Build the document-term matrix
        index.log.debug('build doc-term matrix')
        index.inverted = index._build_matrix_parallel_memmapping(doc_terms).transpose()

        return index

    def save(self) -> None:
        """
        Saves the object in a file.

        :return: None
        """
        super()._save(Path('index/index_{}_{}.npz'.format(self.prep.get_name(), self.inverted.shape[1])))

    @classmethod
    def load(cls, indexed_images: int, prep_name: str = SpacyPreprocessor.get_name(), **prep_kwargs) -> 'Index':
        """
        Loads an index from a file.

        :param prep_name:
        :param indexed_images: number of indexed images in saved index
        :return: Index object loaded from file
        :raise ValueError: if file for index with number of indexed images doesn't exists
        """
        file = Path('index/index_{}_{}.npz'.format(prep_name, indexed_images))

        if not file.exists():
            raise ValueError('No saved index with {} indexed images and {} preprocessor'
                             .format(indexed_images, prep_name))

        return cls._load(file, prep_name=prep_name, **prep_kwargs)