from pathlib import Path

from bs4 import BeautifulSoup, Tag
from lxml import etree
from typing import List

from indexing import DataEntry


"""
Fix-Variables
"""
# Texts bellow images shorter than min_len_texts will be ignored
min_len_texts = 50

# Threshold for calculating the reduction in the number of tags for an xpath
xpath_threshold = 0.5

# Threshold for correctness of two strings
correctness_threshold = 0.4


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


def read_html_text(path: Path) -> List[str]:
    """
    Open TXT-File of HTML-Document
    :param path: Path of text.txt
    :return: List with all extracted lines of text as String
    """
    texts = list()
    with open(path, encoding="utf8") as f:
        for line in f:
            texts.append(line.lower())
    f.close()

    return texts


def cut_last_tag(xpath: str) -> str:
    """
    Get xpath without last tag
    :param xpath: String
    :return: Shorter xpath as String
    """
    counter = xpath.count("/")
    pos = 0

    for p in range(0, counter):
        pos = xpath.find("/", pos)
        pos += 1

    last_tag = xpath[pos:]
    new_xpath = xpath[:len(xpath) - len(last_tag) - 1]

    return new_xpath


def cut_till_figure(xpath: str) -> str:
    """
    Cut xpath till last figure-tag
    :param xpath: String
    :return: Shorter xpath as String
    """
    figure_counter = xpath.count("figure")

    if figure_counter > 0:
        counter = xpath.count("/")
        pos = 0

        for p in range(0, counter):
            pos = xpath.find("/", pos)
            pos += 1

        last_tag = xpath[pos:]

        if "figure" not in last_tag:
            new_xpath = xpath[:len(xpath) - len(last_tag) - 1]
            return cut_till_figure(new_xpath)

        else:
            return xpath


def measure_correctness(string_1: str, string_2: str) -> float:
    """
    Measure the Correctness of a string to another string
    :param string_1: String
    :param string_2: String
    :return: Score as float
    """
    # add fill-characters to shorter string
    while len(string_1) < len(string_2):
        string_1 += "_"

    while len(string_2) < len(string_1):
        string_2 += "_"

    max_length = len(string_1)
    counter_difference = 0

    # counter differences in strings
    for i in range(0, max_length):
        if string_1[i] != string_2[i]:
            counter_difference += 1

    # calculate score
    score = 1 - (counter_difference / max_length)

    return score


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

    a_soup = html_soup
    for s in xpath.split('/'):
        if len(s) != 0:
            inner = s.split('[')
            number = int(inner[1][:-1])
            tag = inner[0]
            a_soup = get_soup(a_soup, tag, number)

    return a_soup


def get_image_html_text(doc: BeautifulSoup, xpathes: List[str], html_text: List[str]) -> str:
    """
    Extract all texts connected to the picture from the HTML-File in a preprocessed form
    :param doc: BeautifulSoup-Object for the HTML-File
    :param xpathes: List with all extracted xpathes as String for the picture in the HTML-File
    :param html_text: List with all Texts of HTML-File as Strings
    :return: String which includes all extracted texts (separated with /n)
    """
    texts = list()
    final_text = str()

    for xpath in xpathes:

        doc = etree.HTML(str(doc))
        xpath = xpath.lower()
        xpath_original = xpath

        a_soup = get_image_soup(xpath, soup)
        a_soup.parent.parent.parent.parent.get_text(separator=' ', strip=True)

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
                    if text_lower in text_h or measure_correctness(text_h, text_lower) >= correctness_threshold:
                        texts.append(text)

    # combine texts to one string
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
    nodes_path = entry.pages[0].snp_nodes

    doc = read_html(doc_path)
    xpath = read_xpath(xpath_path)
    nodes = read_html_text(nodes_path)

    text = get_image_html_text(doc, xpath, nodes)

    return text


def html_test() -> dict:
    """
    Testing html_preprocessing
    :return: Dictionary dataset with extracted texts
    """
    data = DataEntry.get_image_ids(10)
    dataset = dict()

    for d in data:
        pathes = dict()
        pathes.setdefault("snp_dom", DataEntry.load(d).pages[0].snp_dom)
        pathes.setdefault("snp_xpath", DataEntry.load(d).pages[0].snp_image_xpath)
        pathes.setdefault("snp_text", DataEntry.load(d).pages[0].snp_text)

        dataset.setdefault(d, pathes)

    counter = int()
    for d in dataset:
        doc = read_html(dataset[d]["snp_dom"])
        xpath = read_xpath(dataset[d]["snp_xpath"])
        html_text = read_html_text(dataset[d]["snp_text"])
        text = get_image_html_text(doc, xpath, html_text)
        dataset[d].setdefault("text", text)
        print(text)

        if len(text) > 0:
            counter += 1

    print(str(counter) + " : " + str(len(data)))

    return dataset
