import logging
from pathlib import Path
from typing import List, Union

from bs4 import BeautifulSoup
from bs4.element import Tag, Comment, NavigableString

from indexing import DataEntry

"""
Fix-Variables
"""
log = logging.getLogger('html_preprocessing')

# Texts bellow images shorter than min_len_texts will be ignored
min_len_texts = 50

# Threshold for calculating the reduction in the number of tags for an xpath
xpath_threshold = 0.4


def read_html(path: Path) -> BeautifulSoup:
    """
    Open HTML-File
    :param path: Path of the document
    :return: BeautifulSoup-Object for the HTML-File
    """
    with open(path, encoding="utf8") as f:
        doc = BeautifulSoup(f, "html.parser")
    f.close()

    return doc


def read_xpath(path: Path) -> List[str]:
    """
    Open xpath
    :param path: Path of the xpath.txt
    :return: List with all extracted xpathes as String for the picture in the HTML-File
    """
    with open(path, encoding="utf8") as f:
        pathes = list()
        for line in f:
            pathes.append(str(line))
    f.close()

    xpathes = list()
    for path in pathes:
        # ignore wrong xpathes
        if '"' not in path and ':' not in path:
            xpathes.append(path)

    return xpathes


def get_image_soup(xpath: str, html_soup: BeautifulSoup):
    def get_soup(inner_soup: BeautifulSoup, tag: str, number: int) -> BeautifulSoup:
        count = 0
        tag = tag.lower()

        for i in range(0, len(inner_soup.contents)):

            if type(inner_soup.contents[i]) is not Tag:
                continue
            if inner_soup.contents[i].name.lower() == tag:
                count += 1
                if count == number:
                    return inner_soup.contents[i]

        raise ValueError('Wrong xPath')

    a_soup = html_soup
    for s in xpath.split('/'):
        if len(s) != 0 and a_soup is not True:
            inner = s.split('[')
            number = int(inner[1][:-1].replace(']', ''))
            tag = inner[0]
            a_soup = get_soup(a_soup, tag, number)
    return a_soup


def find_img(soup: BeautifulSoup, image_url: str) -> List[Tag]:
    return soup.find_all(name='img', src=image_url)


def node_to_xpath(node):
    node_type = {
        Tag: getattr(node, "name"),
        Comment: "comment()",
        NavigableString: "text()"
    }
    same_type_siblings = list(node.parent.find_all(lambda x: getattr(node, "name", True) == getattr(x, "name", False),
                                                   recursive=False))
    # if len(same_type_siblings) <= 1:
    #     return node_type[type(node)]
    pos = same_type_siblings.index(node) + 1
    return f"{node_type[type(node)]}[{pos}]"


def get_node_xpath(node: Union[Tag, Comment]):
    xpath = "/"
    elements = [f"{node_to_xpath(node)}"]
    for p in node.parents:
        if p.name == "[document]":
            break
        elements.insert(0, node_to_xpath(p))

    xpath = "/" + xpath.join(elements)
    return xpath


def get_image_html_text(doc: BeautifulSoup, xpathes: List[str], image_id: str) -> str:
    """
    Extract all texts connected to the picture from the HTML-File in a preprocessed form
    :param doc: BeautifulSoup-Object for the HTML-File
    :param xpathes: List with all extracted xpathes as String for the picture in the HTML-File
    :return: String which includes all extracted texts (separated with /n)
    """
    texts = list()
    final_text = str()

    for xpath in xpathes:
        try:
            a_soup = get_image_soup(xpath, doc)
        except ValueError as a:
            entry = DataEntry.load(image_id)
            imgs = find_img(doc, entry.url)
            found_better = False
            print(xpath.replace('\n', ''))
            for i in imgs:
                new_xpath = get_node_xpath(i)

                found = False
                for old in xpathes:
                    if new_xpath.lower() == old.lower().replace('\n', ''):
                        found = True
                        break
                if found:
                    continue

                try:
                    a_soup = get_image_soup(new_xpath, doc)
                    print(new_xpath)
                    print('Given xpath faulty, found better')
                    found_better = True
                except ValueError:
                    pass
            if not found_better:
                log.debug('For image %s the xpath: %s is faulty -> ignored', image_id, xpath.replace('\n', ''))
                continue

        count_tags = xpath.count("/")
        text_range = round(count_tags * xpath_threshold)
        # print(text_range, count_tags)

        if text_range < 1 and count_tags > 1 and len(xpathes) < 2:
            text_range = 1

        for i in range(0, text_range):
            a_soup = a_soup.parent

        texts.append(a_soup.get_text(separator=' ', strip=True))

        """
        counter_figure = xpath.count("figure")

        # receive texts from html in figure tags
        for i in range(0, counter_figure):
            xpath_till_figure = cut_till_figure(xpath)
            xpath_figure = "normalize-space(" + xpath_till_figure + ")"
            text = str(doc.xpath(xpath_figure))
            if len(text) >= min_len_texts:
                texts.append(text)

        shorter_xpath = xpath_original
        shorter_xpath_texts = list()
        
        get_texts_range = round(shorter_xpath.count("/") * xpath_threshold)
        if get_texts_range < 1:
            get_texts_range = 1

        # receive texts from html for all shorter xpathes
        for i in range(0, get_texts_range):
            shorter_xpath = cut_last_tag(shorter_xpath)
            text = str(doc.xpath("normalize-space(" + shorter_xpath + ")"))
            if len(text) > min_len_texts and text not in shorter_xpath_texts:
                shorter_xpath_texts.append(text)

        if len(shorter_xpath_texts) > 0:
            for text in shorter_xpath_texts:
                text_lower = text.lower()

                for text_h in html_text:
                    if text_lower in text_h or measure_similarity(text_h, text_lower) >= correctness_threshold:
                        texts.append(text)
    """

    # combine texts to one string
    for text in texts:
        if text not in final_text and len(text) >= min_len_texts:
            final_text += text + "\n"

    return final_text


def run_html_preprocessing(image_id: str) -> str:
    """
    Execute extraction of text for a specific document
    :param image_id: String of image_id
    :return: String which includes all extracted texts (separated with /n)
    """
    entry = DataEntry.load(image_id)
    doc_path = entry.pages[0].snp_dom
    xpath_path = entry.pages[0].snp_image_xpath

    doc = read_html(doc_path)
    xpath = read_xpath(xpath_path)

    text = get_image_html_text(doc, xpath, image_id)

    return text


def html_test() -> dict:
    """
    Testing html_preprocessing
    :return: Dictionary dataset with extracted texts
    """
    data = DataEntry.get_image_ids(100)
    dataset = dict()

    for d in data:
        pathes = dict()
        pathes.setdefault("snp_dom", DataEntry.load(d).pages[0].snp_dom)
        pathes.setdefault("snp_xpath", DataEntry.load(d).pages[0].snp_image_xpath)

        dataset.setdefault(d, pathes)

    counter = int()
    for d in dataset:
        doc = read_html(dataset[d]["snp_dom"])
        xpath = read_xpath(dataset[d]["snp_xpath"])
        # print(d)
        text = get_image_html_text(doc, xpath, d)
        dataset[d].setdefault("text", text)
        # print(text)

        if len(text) > 0:
            counter += 1

    print(str(counter) + " : " + str(len(data)))

    return dataset
