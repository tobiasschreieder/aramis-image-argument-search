from typing import Tuple, Dict

from evaluation import eval_data
from indexing import data_entry

data = eval_data.get_df()
data = data.reset_index()


def exploratory_data_analysis() -> Tuple[Dict, Dict]:
    """
    Do Exploratory Data Analysis
    :return: Dictionary with results of EDA, Dictionary with average scores
    """
    analysis = dict()
    arguments = {"topic_name": "", "count_topic_relevance": 0, "count_images_in_topic": 0,
                 "percentage_topic_relevance": 0.0, "count_argumentative": 0, "percentage_argumentative": 0.0,
                 "count_argumentative_strong": 0, "percentage_argumentative_strong": 0.0,
                 "count_stance": 0, "percentage_stance": 0.0, "count_stance_pro": 0, "percentage_stance_pro": 0.0,
                 "count_stance_con": 0, "percentage_stance_con": 0.0, "count_stance_neutral": 0,
                 "percentage_stance_neutral": 0.0, "count_relevant_images": 0, "percentage_relevant_images": 0.0,
                 "count_relevant_images_strong": 0, "percentage_relevant_images_strong": 0.0}

    for i in data.index:
        column = data.loc[i]

        topic = column.loc["Topic"]
        topic_relevance = column.loc["Topic_correct"]
        argument = column.loc["Argumentative"]
        stance = column.loc["Stance"]

        if topic not in analysis:
            analysis.setdefault(topic, arguments.copy())
            topic_name = data_entry.Topic.get(topic)
            topic_name = topic_name.title
            analysis[topic]["topic_name"] = topic_name

        if topic_relevance:
            analysis[topic]["count_topic_relevance"] += 1

        if argument != "NONE":
            analysis[topic]["count_argumentative"] += 1

        if argument == "STRONG":
            analysis[topic]["count_argumentative_strong"] += 1

        if stance != "NEUTRAL":
            analysis[topic]["count_stance"] += 1

        if stance == "PRO":
            analysis[topic]["count_stance_pro"] += 1

        if stance == "CON":
            analysis[topic]["count_stance_con"] += 1

        if stance == "NEUTRAL":
            analysis[topic]["count_stance_neutral"] += 1

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
        analysis[topic]["percentage_argumentative_strong"] = \
            analysis[topic]["count_argumentative_strong"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_stance"] = \
            analysis[topic]["count_stance"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_stance_pro"] = \
            analysis[topic]["count_stance_pro"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_stance_con"] = \
            analysis[topic]["count_stance_con"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_stance_neutral"] = \
            analysis[topic]["count_stance_neutral"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_relevant_images"] = \
            analysis[topic]["count_relevant_images"] / analysis[topic]["count_images_in_topic"]
        analysis[topic]["percentage_relevant_images_strong"] = \
            analysis[topic]["count_relevant_images_strong"] / analysis[topic]["count_images_in_topic"]

    average_percentage_topic_relevance = 0
    average_percentage_argumentative = 0
    average_percentage_argumentative_strong = 0
    average_percentage_stance = 0
    average_percentage_stance_pro = 0
    average_percentage_stance_con = 0
    average_percentage_stance_neutral = 0
    average_percentage_relevant_images = 0
    average_percentage_relevant_images_strong = 0

    counter = 0
    for topic in analysis:
        counter += analysis[topic]["count_images_in_topic"]
        average_percentage_topic_relevance += analysis[topic]["count_topic_relevance"]
        average_percentage_argumentative += analysis[topic]["count_argumentative"]
        average_percentage_argumentative_strong += analysis[topic]["count_argumentative_strong"]
        average_percentage_stance += analysis[topic]["count_stance"]
        average_percentage_stance_pro += analysis[topic]["count_stance_pro"]
        average_percentage_stance_con += analysis[topic]["count_stance_con"]
        average_percentage_stance_neutral += analysis[topic]["count_stance_neutral"]
        average_percentage_relevant_images += analysis[topic]["count_relevant_images"]
        average_percentage_relevant_images_strong += analysis[topic]["count_relevant_images_strong"]

    average_percentage_topic_relevance = round(average_percentage_topic_relevance / counter, 2)
    average_percentage_argumentative = round(average_percentage_argumentative / counter, 2)
    average_percentage_argumentative_strong = round(average_percentage_argumentative_strong / counter, 2)
    average_percentage_stance = round(average_percentage_stance / counter, 2)
    average_percentage_stance_pro = round(average_percentage_stance_pro / counter, 2)
    average_percentage_stance_con = round(average_percentage_stance_con / counter, 2)
    average_percentage_stance_neutral = round(average_percentage_stance_neutral / counter, 2)
    average_percentage_relevant_images = round(average_percentage_relevant_images / counter, 2)
    average_percentage_relevant_images_strong = round(average_percentage_relevant_images_strong / counter, 2)

    eda = {"average_percentage_topic_relevance": average_percentage_topic_relevance,
           "average_percentage_argumentative": average_percentage_argumentative,
           "average_percentage_argumentative_strong": average_percentage_argumentative_strong,
           "average_percentage_stance": average_percentage_stance,
           "average_percentage_stance_pro": average_percentage_stance_pro,
           "average_percentage_stance_con": average_percentage_stance_con,
           "average_percentage_stance_neutral": average_percentage_stance_neutral,
           "average_percentage_relevant_images": average_percentage_relevant_images,
           "average_percentage_relevant_images_strong": average_percentage_relevant_images_strong}

    # sort analysis by keys
    new_analysis = dict()
    for i in sorted(list(analysis.keys())):
        new_analysis.setdefault(i, analysis[i])
    analysis = new_analysis.copy()

    return analysis, eda


def print_eda():
    """
    Print Exploratory Data Analysis
    """
    analysis, eda = exploratory_data_analysis()
    print("Average percentage of topic relevant images:", str(eda["average_percentage_topic_relevance"]))
    print("Average percentage of argumentative images:", str(eda["average_percentage_argumentative"]))
    print("Average percentage of strong argumentative images:", str(eda["average_percentage_argumentative_strong"]))
    print("Average percentage of stance not neutral images:", str(eda["average_percentage_stance"]))
    print("Average percentage of stance pro images:", str(eda["average_percentage_stance_pro"]))
    print("Average percentage of stance con images:", str(eda["average_percentage_stance_con"]))
    print("Average percentage of stance neutral images:", str(eda["average_percentage_stance_neutral"]))
    print("Average percentage of relevant images:", str(eda["average_percentage_relevant_images"]))
    print("Average percentage of relevant and strong argumentative images:",
          str(eda["average_percentage_relevant_images_strong"]))


def preprocess_string(s: str) -> str:
    """
    Preprocess String
    :param s: str
    :return: s (str)
    """
    s = s.replace("_", " ")
    s = s.title()
    return s


def create_eda_md_table(analysis: Dict, eda: Dict):
    """
    Create Markdown File with Table of EDA
    :param eda: eda Dictionary
    :param analysis: Dictionary
    """
    text = list()
    text.append("#Analysis of labeled data")
    text.append("##Analysis per Topic")
    text.append("| Topic Number | Topic Name | Topic Relevance | Argumentative | Strong Argumentative | Stance Pro | "
                "Stance Con | Relevant | Strong Relevant |")
    text.append("|---|---|---|---|---|---|---|---|---|")
    for i in analysis:
        column = "| " + str(i) + " "
        column += "| " + str(analysis[i]["topic_name"]) + " "
        column += "| " + str(round(analysis[i]["percentage_topic_relevance"], 2)) + " " \
                  + " (" + str(analysis[i]["count_topic_relevance"]) + ") "
        column += "| " + str(round(analysis[i]["percentage_argumentative"], 2)) + " " \
                  + " (" + str(analysis[i]["count_argumentative"]) + ") "
        column += "| " + str(round(analysis[i]["percentage_argumentative_strong"], 2)) + " " \
                  + " (" + str(analysis[i]["count_argumentative_strong"]) + ") "
        column += "| " + str(round(analysis[i]["percentage_stance_pro"], 2)) + " " \
                  + " (" + str(analysis[i]["count_stance_pro"]) + ") "
        column += "| " + str(round(analysis[i]["percentage_stance_con"], 2)) + " " \
                  + " (" + str(analysis[i]["count_stance_con"]) + ") "
        column += "| " + str(round(analysis[i]["percentage_relevant_images"], 2)) \
                  + " (" + str(analysis[i]["count_relevant_images"]) + ") "
        column += "| " + str(round(analysis[i]["percentage_relevant_images_strong"], 2)) \
                  + " (" + str(analysis[i]["count_relevant_images_strong"]) + ") "
        column += "|"
        text.append(column)

    text.append("\n")
    text.append("##Analysis Overall")
    text.append("| Category | Value |")
    text.append("|---|---|")
    for i in eda:
        column = "| " + preprocess_string(i) + " "
        column += "| " + str(eda[i]) + " "
        column += "|"
        text.append(column)

    with open('analysis_labeled_data_table.md', 'w') as f:
        for item in text:
            f.write("%s\n" % item)






