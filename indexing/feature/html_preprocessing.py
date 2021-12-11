from pathlib import Path

from bs4 import BeautifulSoup, Tag
from lxml import etree
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

from indexing import DataEntry

"""
Fix-Variables
"""
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

        #print(str(type(inner_soup)))
        try:
            for i in range(0, len(inner_soup.contents)):

                if type(inner_soup.contents[i]) is not Tag:
                    continue
                if inner_soup.contents[i].name.lower() == tag:
                    count += 1
                    if count == number:
                        return inner_soup.contents[i]

        except AttributeError:
            return True

    a_soup = html_soup
    for s in xpath.split('/'):
        error = bool
        if len(s) != 0 and a_soup is not True:
            inner = s.split('[')
            number = int(inner[1][:-1].replace(']', ''))
            tag = inner[0]
            #print(tag, number)
            a_soup = get_soup(a_soup, tag, number)

    #print(type(a_soup))
    return a_soup


def get_image_html_text(doc: BeautifulSoup, xpathes: List[str]) -> str:
    """
    Extract all texts connected to the picture from the HTML-File in a preprocessed form
    :param doc: BeautifulSoup-Object for the HTML-File
    :param xpathes: List with all extracted xpathes as String for the picture in the HTML-File
    :param html_text: List with all Texts of HTML-File as Strings
    :return: String which includes all extracted texts (separated with /n)
    """
    texts = list()
    final_text = str()
    errors = 0

    for xpath in xpathes:
        a_soup = get_image_soup(xpath, doc)

        if a_soup is not True and a_soup is not None:
            count_tags = xpath.count("/")
            text_range = round(count_tags * xpath_threshold)
            #print(text_range, count_tags)

            if text_range < 1 and count_tags > 1 and len(xpathes) < 2:
                text_range = 1

            for i in range(0, text_range):
                a_soup = a_soup.parent

            texts.append(a_soup.get_text(separator=' ', strip=True))

        else:
            errors += 1

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
        if text not in final_text and text >= min_len_texts:
            final_text += text + "\n"

    return final_text, errors


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

    text = get_image_html_text(doc, xpath)

    return text


def html_test() -> dict:
    """
    Testing html_preprocessing
    :return: Dictionary dataset with extracted texts
    """
    data = DataEntry.get_image_ids()
    dataset = dict()
    type_error = 0

    for d in data:
        pathes = dict()
        pathes.setdefault("snp_dom", DataEntry.load(d).pages[0].snp_dom)
        pathes.setdefault("snp_xpath", DataEntry.load(d).pages[0].snp_image_xpath)

        dataset.setdefault(d, pathes)

    counter = int()
    for d in dataset:
        doc = read_html(dataset[d]["snp_dom"])
        xpath = read_xpath(dataset[d]["snp_xpath"])
        print(d)
        text, error = get_image_html_text(doc, xpath)
        dataset[d].setdefault("text", text)
        print(text)

        type_error += error

        if len(text) > 0:
            counter += 1

    print(str(counter) + " : " + str(len(data)))

    return dataset, type_error
