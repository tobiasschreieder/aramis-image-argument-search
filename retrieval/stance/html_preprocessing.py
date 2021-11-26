# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:44:02 2021

@author: tobias
"""

from bs4 import BeautifulSoup
from lxml import etree
from indexing import DataEntry

# texts bellow images shorter than min_len_texts will be ignored
min_len_texts = 10


# open HTML-File
def read_html(path):
    with open(path, encoding="utf8") as f:
        doc = BeautifulSoup(f, "html.parser")
    f.close()

    return doc


# open xpath
def read_xpath(path):
    with open(path, encoding="utf8") as f:
        pathes = list()
        for line in f:
            pathes.append(str(line))
    f.close()

    # use first useful xpath
    xpath = str()
    for path in pathes:
        if ('"' not in path and ':' not in path and len(xpath) == 0):
            xpath = path

    return xpath


# extract all texts from the following tags: <figure>, <picture>, <img> in a preprocessed form
def get_image_html_text(doc, xpath):
    texts = list()

    # figure, picture, img in xpath
    f = False
    p = False
    i = False

    # position of figure, picture, img in html
    f_pos = int()
    p_pos = int()
    i_pos = int()

    doc = etree.HTML(str(doc))
    xpath = xpath.lower()

    # bring xpath to basic form (without figure, picture, img)
    if ("figure" in xpath):
        f = True
        for j in range(1, 5):
            if ("figure[" + str(j) in xpath):
                xpath = xpath.replace("/figure[" + str(j) + "]", "")
                f_pos = j
    if ("picture" in xpath):
        p = True
        for j in range(1, 5):
            if ("picture[" + str(j) in xpath):
                xpath = xpath.replace("/picture[" + str(j) + "]", "")
                p_pos = j
    if ("img" in xpath):
        i = True
        for j in range(1, 5):
            if ("img[" + str(j) in xpath):
                xpath = xpath.replace("/img[" + str(j) + "]", "")
                i_pos = j

    # create specific xpaths
    xpath_figure = str()
    xpath_picture = str()
    xpath_img = str()

    # receive texts from html for figure, picture, img
    if (f == True):
        xpath_figure = "normalize-space(" + xpath + "/figure[" + str(f_pos) + "])"
        text = str(doc.xpath(xpath_figure))
        if (len(text) >= min_len_texts):
            texts.append(text)
    if (p == True):
        xpath_picture = "normalize-space(" + xpath + "/picture[" + str(p_pos) + "])"
        text = str(doc.xpath(xpath_picture))
        if (len(text) >= min_len_texts):
            texts.append(text)
    if (i == True):
        xpath_img = "normalize-space(" + xpath + "/img[" + str(i_pos) + "])"
        text = str(doc.xpath(xpath_img))
        if (len(text) >= min_len_texts):
            texts.append(text)

    final_text = str()

    for text in texts:
        final_text += text + "\n"

    return final_text


# input image_id -> return extracted texts as string
def run_html_preprocessing(image_id):
    doc_path = DataEntry.load(image_id).pages[0].snp_dom
    xpath_path = DataEntry.load(image_id).pages[0].snp_image_xpath

    doc = read_html(doc_path)
    xpath = read_xpath(xpath_path)

    text = get_image_html_text(doc, xpath)
    print(text)

    return text


'''
# get test sample of data (just for testing)
def get_pathes():
    data = DataEntry.get_image_ids(50)

    dataset = dict()

    for d in data:
        pathes = dict()
        pathes.setdefault("snp_dom", DataEntry.load(d).pages[0].snp_dom)
        pathes.setdefault("snp_xpath", DataEntry.load(d).pages[0].snp_image_xpath)

        dataset.setdefault(d, pathes)

    return dataset


# method for testing
def html_test():
    data = get_pathes()
    counter = int()
    for d in data:
        print(d)
        doc = read_html(data[d]["snp_dom"])
        xpath = read_xpath(data[d]["snp_xpath"])
        text = get_image_html_text(doc, xpath)
        print(text)
        data[d].setdefault("text", text)

        if len(text) > 0:
            counter += 1
    print(str(counter) + " : " + str(len(data)))

    return data
'''
