import math

import backend.image_detection as image_detection
import backend.sentiment_detection as sentiment_detection


def log_normal_density_function(x):
    return (0.5 / (math.sqrt(2*math.pi) * 0.7 * x)) * math.e**(((math.log(x, math.e)+1.7)**2)/(0.98))


def calculate_argumentativeness(path):
    image = image_detection.read_image(path)
    text = image_detection.text_from_image(image)
    text_position = image_detection.textposition_from_image(image, plot=False)
    shapes = image_detection.shapes_from_image(image, plot=False)
    roi_area = image_detection.diagramms_from_image(image, plot=False)

    sentiment_score = sentiment_detection.sentiment_nltk(text)

    shape_factor = math.log(len(shapes) + 1) + 1
    print("shape factor: ", shape_factor)

    # use cazy function to get a score between 0 and 1
    diagramm_factor = log_normal_density_function(roi_area)
    print("diagramm factor: ", diagramm_factor)

    # getting max 3
    text_factor = (math.log(len(text.split(" ")) + 1) + 1) * (abs(sentiment_score) + 1)
    print("text factor: ", text_factor)

    score = (diagramm_factor / shape_factor) + text_factor
    print("final score: ", score)

    return score


def calculate_pro_con(path):
    # color analyze
    # smile recognition
    # image style

    sentiment_score = sentiment_detection.sentiment_nltk(text)

    score = sentiment_score

    return score


[]
scores = []
for i in range(1, 31):
    path = str("image_" + str(i) + ".png")
    s = round(calculate_argumentativeness(path), 4)
    scores.append(s)

print(scores)


    

