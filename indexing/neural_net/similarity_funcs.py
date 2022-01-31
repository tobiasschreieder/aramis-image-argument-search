import math
from typing import List

from Bio import pairwise2

from indexing.feature import sentiment_detection


def query_frequency(query: List[str], text: List[str]) -> float:
    rho = 0
    doc_length = len(text)
    if doc_length == 0:
        return 0
    count = 0
    for term in query:
        frequency = text.count(term)
        if frequency == 0:
            continue
        try:
            rho += math.exp(math.log(frequency / doc_length))
            count += 1
        except ValueError:
            rho += 0
    if not count == 0:
        return float(rho / count)
    else:
        return 0


def context_sentiment(query: List[str], text: List[str]) -> float:
    val = 0
    scope = 5
    count = 0
    for term in query:
        last = 0
        try:
            while last < len(text):
                last = text.index(term, last + scope)
                context = text[last - scope:last + scope]
                val += sentiment_detection.sentiment_nltk(' '.join(context))
                count += 1
        except ValueError:
            pass
    if not count == 0:
        return val / count
    else:
        return 0


def alignment_query(query: List[str], text: List[str]) -> float:
    # looking for exact alignments of query in text
    a = pairwise2.align.localxx(text, query, gap_char=["-"])
    sum_score = 0
    number_alignments = 0
    for match in a:
        number_alignments += 1
        sum_score += match.score
    # normalize alignment-score
    if not number_alignments == 0:
        avg_score = sum_score / number_alignments
        return avg_score / len(query)
    else:
        return 0
