import dataclasses
from pathlib import Path
from typing import List


@dataclasses.dataclass
class WebPage:

    url_hash: str
    url: str
    snapshot_path: Path

    @classmethod
    def load(cls, page_path: Path) -> 'WebPage':
        """
        Create a WebPage object for the given page path.

        :param page_path: The path from witch the new object is generated
        :return: WebPage for given page path
        :raises ValueError: if page_path doesn't exists or isn't a directory
        """
        if not (page_path.exists() and page_path.is_dir()):
            raise ValueError('{} is not a valid directory'.format(page_path))

        with page_path.joinpath('page-url.txt').open() as file:
            url = file.readline()

        return cls(
            url_hash=page_path.name,
            url=url,
            snapshot_path=page_path.joinpath('snapshot'),
        )


@dataclasses.dataclass
class DataEntry:

    url_hash: str
    url: str
    png_path: Path
    webp_path: Path
    pages: List[WebPage]

    @classmethod
    def load(cls, image_id) -> 'DataEntry':
        """
        Create a DataEntry object for the given image id.

        :param image_id: The image id of the new object
        :return: DataEntry for given image id
        :raises ValueError: if image_id doesn't exists
        """
        im_path = 'images/{}/{}/'.format(image_id[0:3], image_id)
        if not Path('data/touche22-images-main/').joinpath(im_path).exists():
            raise ValueError('{} is not a valid image hash'.format(image_id))

        with Path('data/touche22-images-main/').joinpath(im_path).joinpath('image-url.txt').open() as file:
            url = file.readline()

        pages = []
        for page in Path('data/touche22-images-main/').joinpath(im_path).joinpath('pages').iterdir():
            pages.append(WebPage.load(page))

        return cls(
            url_hash=image_id,
            url=url,
            png_path=Path('data/touche22-images-png-images/').joinpath(im_path).joinpath('image.png'),
            webp_path=Path('data/touche22-images-main/').joinpath(im_path).joinpath('image.webp'),
            pages=pages,
        )

    @staticmethod
    def get_image_ids(max_size: int = -1) -> List[str]:
        """
        Returns number of images ids in a sorted list. If max_size is < 1 return all image ids.

        :param max_size: Parameter to determine maximal length of returned list.
        :return: List of image ids as strings
        """
        id_list = []
        main_path = Path('data/touche22-images-main/images')
        count = 0
        check_length = max_size > 0
        for idir in main_path.iterdir():
            for image_hash in idir.iterdir():
                id_list.append(image_hash.name)
                count += 1
                if check_length and count >= max_size:
                    return sorted(id_list)
        return sorted(id_list)