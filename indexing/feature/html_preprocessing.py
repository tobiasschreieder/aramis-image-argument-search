import logging
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from bs4.element import Tag

from indexing import DataEntry

import re

"""
Fix-Variables
"""
# logging html_preprocessing
log = logging.getLogger('html_preprocessing')

# extract texts of parent xpath till given length
extract_texts_till_length = 100

# Threshold for calculating the reduction in the number of tags for a xpath [0, 1]
xpath_threshold = 0.9


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
        path = path.strip()
        # ignore wrong xpathes
        if '"' not in path and ':' not in path:
            xpathes.append(path)

    return xpathes


def prettify_strings(text: str) -> str:
    """
    Prettify text in given String
    :param text: Text as String
    :return: prettifyed text as String
    """
    text = text.replace("  ", " ")
    text = re.sub('http[s]?://\S+', '', text)

    return text


def get_image_soup(xpath: str, html_soup: BeautifulSoup):
    """
    Get BeautifulSoup object for position of last tag of xpath
    :param xpath: String with current xpath
    :param html_soup: BeautifulSoup Object for the HTML-File
    :return: BeautifulSoup Object for position of last tag of xpath
    """

    def get_soup(inner_soup: BeautifulSoup, tag: str, number: int) -> BeautifulSoup:
        """
        Get BeautifulSoup object for next position in xpath
        :param inner_soup: current BeautifulSoup object
        :param tag: String with current tag of xpath
        :param number: Int with number behind current tag of xpath
        :return: BeautifulSoup object for current tag[number] of xpath
        """
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
            num = int(inner[1][:-1].replace(']', ''))
            a_soup = get_soup(a_soup, inner[0], num)
    return a_soup


def get_image_html_text(doc: BeautifulSoup, xpathes: List[str], image_id: str) -> str:
    """
    Extract all texts connected to the picture from the HTML-File in a preprocessed form
    :param doc: BeautifulSoup-Object for the HTML-File
    :param xpathes: List with all extracted xpathes as String for the picture in the HTML-File
    :param image_id: Image-ID as String
    :return: String which includes all extracted texts (separated with /n)
    """
    texts = list()
    final_text = str()

    # extract texts near to all xpathes of image
    for xpath in xpathes:
        try:
            a_soup = get_image_soup(xpath, doc)
        except ValueError:
            log.debug('For image %s the xpath: %s is faulty -> ignored', image_id, xpath.replace('\n', ''))
            continue

        count_tags = xpath.count("/")
        text_range = round(count_tags * xpath_threshold)

        # text_rang must be >= 1
        if (text_range < 1) and (count_tags > 1) and (len(xpathes) < 2):
            text_range = 1

        # extract texts of parent tags in HTML-File till text length >= extract_texts_till_length
        current_text = ""
        for i in range(0, text_range):
            if len(current_text) < extract_texts_till_length:
                a_soup = a_soup.parent
                current_text = a_soup.get_text(separator=' ', strip=True)
                current_text = prettify_strings(current_text)

        texts.append(current_text)

    # combine texts of different xpathes to one string
    for text in texts:
        if text not in final_text:
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
    data = DataEntry.get_image_ids()  # specify amount of images which should be considered, (): all images
    dataset = dict()

    for d in data:
        pathes = dict()
        pathes.setdefault("snp_dom", DataEntry.load(d).pages[0].snp_dom)
        pathes.setdefault("snp_xpath", DataEntry.load(d).pages[0].snp_image_xpath)

        dataset.setdefault(d, pathes)

    counter = int()
    average_text_length = 0

    for d in dataset:
        doc = read_html(dataset[d]["snp_dom"])
        xpath = read_xpath(dataset[d]["snp_xpath"])
        text = get_image_html_text(doc, xpath, d)
        dataset[d].setdefault("text", text)
        average_text_length += len(text)
        print(d)
        print(text)

        if len(text) > 0:
            counter += 1

    average_text_length = round(average_text_length / len(dataset))

    print("\nFound texts in", str(counter) + " : " + str(len(data)), "HTML documents")
    print("Average length of texts:", str(average_text_length), "characters")

    return dataset
