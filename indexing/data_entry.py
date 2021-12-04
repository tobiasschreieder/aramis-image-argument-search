import dataclasses
import json
from pathlib import Path
from typing import List

from config import Config

cfg = Config.get()

faulty_ids = []

# Error with cv2
faulty_ids += ['I37377d0aa6480791', 'I8d5ddcc5ee8c7cd2', 'I96a19b1859d5898b']

# removed from dataset
faulty_ids += ['I06c858b97ae2147c', 'I0e95a1d02cf4e3be', 'I273c7849db3492b0', 'I3c1c0cc8be157979',
               'I643e21bd29dc75c6', 'I65265d8f12c69f3e', 'I8d8e4d640697fa8f', 'I9a1b2e093f34d1ae',
               'Ia9bf7858d6922cf7', 'Ie9b890a76eb9233f', 'Ieb02df7b9ccf1bd3', 'If14e0ff8f1757dfa',
               'Ifbc537388349522f']


@dataclasses.dataclass
class Ranking:
    query: str
    topic: int
    rank: int

    @classmethod
    def load(cls, ranking: str) -> 'Ranking':
        """
        Create a Ranking object for the given ranking string.

        :param ranking: the string to parse
        :return: Ranking for given string
        """
        j_rank = json.loads(ranking)
        try:
            return cls(
                query=j_rank['query'],
                topic=int(j_rank['topic']),
                rank=int(j_rank['rank']),
            )
        except KeyError or ValueError or json.JSONDecodeError:
            raise ValueError("The given string isn't correctly formalized.")


@dataclasses.dataclass
class WebPage:
    url_hash: str
    url: str

    snp_dom: Path
    snp_image_xpath: Path
    snp_nodes: Path
    snp_screenshot: Path
    snp_text: Path
    snp_archive: Path

    rankings: List[Ranking]

    @classmethod
    def load(cls, page_path: Path, image_id: str) -> 'WebPage':
        """
        Create a WebPage object for the given page path.

        :param image_id: The id of the parent image
        :param page_path: The path from witch the new object is generated
        :return: WebPage for given page path
        :raises ValueError: if page_path doesn't exists or isn't a directory
        """
        if not (page_path.exists() and page_path.is_dir()):
            raise ValueError('{} is not a valid directory'.format(page_path))

        path_main = cfg.data_location.joinpath(Path('touche22-images-main'))
        path_from_image = Path('images/' + image_id[0:3] + '/' + image_id + '/pages').joinpath(page_path.name)

        with page_path.joinpath('page-url.txt').open(encoding='utf8') as file:
            url = file.readline()

        snp_dom = path_main.joinpath(path_from_image).joinpath('snapshot/dom.html')
        snp_image_xpath = path_main.joinpath(path_from_image).joinpath('snapshot/image-xpath.txt')
        snp_nodes = cfg.data_location.joinpath(Path('touche22-images-nodes/').joinpath(path_from_image)
                                               .joinpath('snapshot/nodes.jsonl'))
        snp_screenshot = cfg.data_location.joinpath(Path('touche22-images-screenshots/').joinpath(path_from_image)) \
            .joinpath('snapshot/screenshot.png')
        snp_text = path_main.joinpath(path_from_image).joinpath('snapshot/text.txt')
        snp_archive = cfg.data_location.joinpath(Path('touche22-images-archives/').joinpath(path_from_image)) \
            .joinpath('snapshot/web-archive.warc.gz')

        with cfg.data_location.joinpath(Path('touche22-images-rankings/')).joinpath(path_from_image) \
                .joinpath('rankings.jsonl').open() as file:
            rankings = [Ranking.load(line) for line in file]

        return cls(
            url_hash=page_path.name,
            url=url,
            snp_dom=snp_dom, snp_image_xpath=snp_image_xpath, snp_nodes=snp_nodes,
            snp_screenshot=snp_screenshot, snp_text=snp_text, snp_archive=snp_archive,
            rankings=rankings,
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
        if not cfg.data_location.joinpath(Path('touche22-images-main/')).joinpath(im_path).exists():
            raise ValueError('{} is not a valid image hash'.format(image_id))

        with cfg.data_location.joinpath(Path('touche22-images-main/')).joinpath(im_path) \
                .joinpath('image-url.txt').open() as file:
            url = file.readline()

        pages = []
        for page in cfg.data_location.joinpath(Path('touche22-images-main/')).joinpath(im_path) \
                .joinpath('pages').iterdir():
            pages.append(WebPage.load(page, image_id))

        png = cfg.data_location.joinpath(Path('touche22-images-png-images/')).joinpath(im_path).joinpath('image.png')
        webp = cfg.data_location.joinpath(Path('touche22-images-main/')).joinpath(im_path).joinpath('image.webp')

        return cls(
            url_hash=image_id,
            url=url,
            png_path=png,
            webp_path=webp,
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
        main_path = cfg.data_location.joinpath(Path('touche22-images-main/images'))
        count = 0
        check_length = max_size > 0
        for idir in main_path.iterdir():
            for image_hash in idir.iterdir():
                if image_hash in faulty_ids:
                    continue
                id_list.append(image_hash.name)
                count += 1
                if check_length and count >= max_size:
                    return sorted(id_list)
        return sorted(id_list)
