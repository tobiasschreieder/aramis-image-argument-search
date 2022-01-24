from typing import Tuple, Dict, List

from evaluation import evaluation

data = evaluation.get_df()
data = data.reset_index()


def exploratory_data_analysis() -> Tuple[Dict, Dict]:
    """
    Do Exploratory Data Analysis
    :return: Dictionary with results of EDA, Dictionary with average scores
    """
    analysis = dict()
    arguments = {"count_topic_relevance": 0, "count_images_in_topic": 0, "percentage_topic_relevance": 0.0,
                 "count_argumentative": 0, "percentage_argumentative": 0.0, "count_stance": 0, "percentage_stance": 0.0,
                 "count_relevant_images": 0, "percentage_relevant_images": 0.0, "count_relevant_images_strong": 0,
                 "percentage_relevant_images_strong": 0.0}

    for i in data.index:
        column = data.loc[i]

        topic = column.loc["Topic"]
        topic_relevance = column.loc["Topic_correct"]
        argument = column.loc["Argumentative"]
        stance = column.loc["Stance"]

        if topic not in analysis:
            analysis.setdefault(topic, arguments.copy())

        if topic_relevance:
            analysis[topic]["count_topic_relevance"] += 1

        if argument != "NONE":
            analysis[topic]["count_argumentative"] += 1

        if stance != "NEUTRAL":
            analysis[topic]["count_stance"] += 1

        if topic_relevance and argument != "NONE" and stance != "NEUTRAL":
            analysis[topic]["count_relevant_images"] += 1

        if topic_relevance and argument == "STRONG" and stance != "NEUTRAL":
            analysis[topic]["count_relevant_images_strong"] += 1

        analysis[topic]["count_images_in_topic"] += 1

    new_analysis = analysis.copy()

    for topic in analysis:
        if analysis[topic]["count_images_in_topic"] < 100:
            new_analysis.pop(topic)

    analysis = new_analysis

    for topic in analysis:
        analysis[topic]["percentage_topic_relevance"] = \
            analysis[topic]["count_topic_relevance"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_argumentative"] = \
            analysis[topic]["count_argumentative"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_stance"] = \
            analysis[topic]["count_stance"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_relevant_images"] = \
            analysis[topic]["count_relevant_images"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_relevant_images_strong"] = \
            analysis[topic]["count_relevant_images_strong"] / analysis[topic]["count_images_in_topic"]

    average_percentage_topic_relevance = 0.0
    average_percentage_argumentative = 0.0
    average_percentage_stance = 0.0
    average_percentage_relevant_images = 0.0
    average_percentage_relevant_images_strong = 0.0

    for topic in analysis:
        average_percentage_topic_relevance += analysis[topic]["percentage_topic_relevance"]
        average_percentage_argumentative += analysis[topic]["percentage_argumentative"]
        average_percentage_stance += analysis[topic]["percentage_stance"]
        average_percentage_relevant_images += analysis[topic]["percentage_relevant_images"]
        average_percentage_relevant_images_strong += analysis[topic]["percentage_relevant_images_strong"]

    average_percentage_topic_relevance = round(average_percentage_topic_relevance / len(analysis), 2)
    average_percentage_argumentative = round(average_percentage_argumentative / len(analysis), 2)
    average_percentage_stance = round(average_percentage_stance / len(analysis), 2)
    average_percentage_relevant_images = round(average_percentage_relevant_images / len(analysis), 2)
    average_percentage_relevant_images_strong = round(average_percentage_relevant_images_strong / len(analysis), 2)

    eda = {"average_percentage_topic_relevance": average_percentage_topic_relevance,
           "average_percentage_argumentative": average_percentage_argumentative,
           "average_percentage_stance": average_percentage_stance,
           "average_percentage_relevant_images": average_percentage_relevant_images,
           "average_percentage_relevant_images_strong": average_percentage_relevant_images_strong}

    return analysis, eda


def print_eda():
    """
    Print Exploratory Data Analysis
    """
    analysis, eda = exploratory_data_analysis()
    print("Average percentage of topic relevant images:", str(eda["average_percentage_topic_relevance"]))
    print("Average percentage of argumentative images:", str(eda["average_percentage_argumentative"]))
    print("Average percentage of stance not neutral images:", str(eda["average_percentage_stance"]))
    print("Average percentage of relevant images:", str(eda["average_percentage_relevant_images"]))
    print("Average percentage of relevant and strong argumentative images:",
          str(eda["average_percentage_relevant_images_strong"]))
