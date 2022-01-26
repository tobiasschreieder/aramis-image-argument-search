import math
import os

# to get no console-print from tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras

from indexing import FeatureIndex
from indexing import features_NN_argument
from indexing.feature import sentiment_detection

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
from pathlib import Path
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD

overfitCallback = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10)


def train_network(model_name: str, df: pd.DataFrame):
    '''
    model_argument = load_model("indexing/models/test_1_argument/model.hS")
    input_data = [row for index, row in df.iterrows()]
    results_argument = features_NN_argument.make_prediction(model=model_argument, input_data=input_data)
    df['argumentativeness'] = results_argument

    df = df.loc[df['argumentativeness'] > 0.4]
    '''
    model_name = model_name + "_stance"

    # --- handling the scaling, train and test-data
    print("start scaling dataframe")
    for index, row in df.iterrows():
        df.loc[index, :] = scale_data(row)
    print("finished scaling")

    column_list = ['image_percentage_green',
                   'image_percentage_red',
                   'image_percentage_blue',
                   'image_percentage_yellow',
                   'image_percentage_bright',
                   'image_percentage_dark',
                   'html_sentiment_score',
                   'text_len',
                   'text_sentiment_score',
                   'image_average_color_r',
                   'image_average_color_g',
                   'image_average_color_b',
                   'image_dominant_color_r',
                   'image_dominant_color_g',
                   'image_dominant_color_b',
                   'query_sentiment']

    split_data = split_train_test_data(df, column_list)
    x_train = split_data['x_train']
    x_test = split_data['x_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']

    model = Sequential([
        Dense(10, input_dim=len(column_list), activation='relu'),
        Dense(5, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    history = model.fit(x=x_train, y=y_train,
                        epochs=16, batch_size=5,
                        validation_data=(x_test, y_test))
                        #callbacks=[overfitCallback])

    Path("indexing/models/" + str(model_name)).mkdir(parents=True, exist_ok=True)

    # summarize history for accuracy
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('indexing/models/' + str(model_name) + '/accuracy_function.png')

    # summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('indexing/models/' + str(model_name) + '/loss_function.png')

    model.save('indexing/models/' + str(model_name) + '/model.hS')


def split_train_test_data(df: pd.DataFrame, column_list: list):
    # shuffle data-rows
    df.sample(frac=1)

    df_eval = df.loc[df['topic'].isin([27])].drop('topic', 1)
    # df = df.loc[df['topic'].isin([2, 8, 33, 40, 27])]

    df_len = len(df.index)
    split_index = round(df_len * 0.8)

    # df_train = df.iloc[:split_index, :]
    # df_test = df.iloc[split_index:, :]

    df_test = df.loc[df['topic'].isin([4, 21, 43, 48])]
    df_train = df.loc[df['topic'].isin([2, 8, 33, 40, 27])]

    y_train = df_train['stance_eval']
    y_train = eval_to_categorial(y_train)

    y_test = df_test['stance_eval']
    y_test = eval_to_categorial(y_test)

    x_train = df_train[column_list]
    x_train = np.asarray(x_train).astype('float32')

    x_test = df_test[column_list]
    x_test = np.asarray(x_test).astype('float32')

    return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test, 'df_eval': df_eval}


def eval_to_categorial(data):
    output_data = []
    for value in data:
        if value == 0.0:
            output_data.append([1, 0, 0])
        elif value == 1.0:
            output_data.append([0, 0, 1])
        else:
            output_data.append([0, 1, 0])
    return np.asarray(output_data)


def categorial_to_eval(data):
    output_data = []
    for value in data:
        value_0 = value[0]
        value_1 = value[1]
        value_2 = value[2]

        result = (value_0 * (-1)) + (value_2 * 1)
        output_data.append(result)

    return np.asarray(output_data)


def log_normal_density_function(x: float) -> float:
    if x == 0:
        return 0
    elif x == 1:
        return 0
    else:
        return ((1 / (math.sqrt(2 * math.pi) * 0.16 * (-x + 1))) * math.exp(
            ((math.log((-x + 1), 10) + 0.49) ** 2) / -0.0512) * 0.12)


def scale_data(df_row: pd.Series) -> pd.Series:
    df_row['html_sentiment_score'] = df_row['html_sentiment_score']
    df_row['text_len'] = (1 - (1 / (math.exp(0.01 * df_row['text_len'])))) * 3
    df_row['text_sentiment_score'] = df_row['text_sentiment_score']
    df_row['image_percentage_green'] = df_row['image_percentage_green'] / 100
    df_row['image_percentage_red'] = df_row['image_percentage_red'] / 100
    df_row['image_percentage_blue'] = df_row['image_percentage_blue'] / 100
    df_row['image_percentage_yellow'] = df_row['image_percentage_yellow'] / 100
    df_row['image_percentage_bright'] = df_row['image_percentage_bright'] / 100
    df_row['image_percentage_dark'] = df_row['image_percentage_dark'] / 100
    df_row['image_average_color_r'] = df_row['image_average_color_r'] / 360
    df_row['image_average_color_g'] = df_row['image_average_color_g'] / 360
    df_row['image_average_color_b'] = df_row['image_average_color_b'] / 360
    df_row['image_dominant_color_r'] = df_row['image_dominant_color_r'] / 360
    df_row['image_dominant_color_g'] = df_row['image_dominant_color_g'] / 360
    df_row['image_dominant_color_b'] = df_row['image_dominant_color_b'] / 360
    df_row['image_roi_area'] = log_normal_density_function(df_row['image_roi_area'])
    df_row['query_sentiment'] = sentiment_detection.sentiment_nltk(df_row['query_sentiment'])

    return df_row


def make_prediction(model: keras.Model, input_data: list) -> list:
    """
    Method to predict some steps for a runners level
    :param model_name: name of the trained network
    :return: a list of all predictions. Every prediction is a list of 40 values due to the 40-nodes-last-layer
    """

    tp_input_data = []
    color_input_data = []
    primary_features_input_data = []

    for row in input_data:
        scaled_row = scale_data(row)

        tp_str = scaled_row['text_position']
        list_from_str = FeatureIndex.convert_text_area_from_str(tp_str)
        dim_heatmap = int(math.sqrt(len(list_from_str)))
        chunked_list = list()
        for i in range(0, len(list_from_str), dim_heatmap):
            chunked_list.append(list_from_str[i:i + dim_heatmap])
        tp_input_data.append(chunked_list)

        color_input = scaled_row[['image_average_color_r',
                                  'image_average_color_g',
                                  'image_average_color_b',
                                  'image_dominant_color_r',
                                  'image_dominant_color_g',
                                  'image_dominant_color_b']]
        color_input_data.append(np.asarray(color_input).astype('float32'))

        primary_input = scaled_row[['image_percentage_green',
                                    'image_percentage_red',
                                    'image_percentage_blue',
                                    'image_percentage_yellow',
                                    'image_percentage_bright',
                                    'image_percentage_dark',
                                    'html_sentiment_score',
                                    'text_len',
                                    'text_sentiment_score',
                                    'image_type',
                                    'image_roi_area']]
        primary_features_input_data.append(np.asarray(primary_input).astype('float32'))

    tp_input_data = np.expand_dims(tp_input_data, axis=3)
    color_input_data = np.asarray(color_input_data)
    primary_features_input_data = np.asarray(primary_features_input_data)

    predictions = model.predict(x=[tp_input_data, color_input_data, primary_features_input_data])

    predictions = [value[0] for value in predictions]

    print(predictions)
    return predictions


def make_prediction_LorenzIdea(model: keras.Model, input_data: list):
    model_argument = load_model("indexing/models/test_1_argument/model.hS")
    results_argument = features_NN_argument.make_prediction(model=model_argument, input_data=input_data)

    column_list = ['image_percentage_green',
                   'image_percentage_red',
                   'image_percentage_blue',
                   'image_percentage_yellow',
                   'image_percentage_bright',
                   'image_percentage_dark',
                   'html_sentiment_score',
                   'text_len',
                   'text_sentiment_score',
                   'image_average_color_r',
                   'image_average_color_g',
                   'image_average_color_b',
                   'image_dominant_color_r',
                   'image_dominant_color_g',
                   'image_dominant_color_b',
                   'query_sentiment']

    non_argumentative_data = []
    input_data_network = []
    for i in range(len(input_data)):
        scaled_row = scale_data(input_data[i])
        row_input = scaled_row[column_list]
        # row_input['argumentativeness'] = results_argument[i]
        input_data_network.append(row_input)
        if results_argument[i] < 0.4:
            pass
            # non_argumentative_data.append(i)

    input_data_network = np.asarray(input_data_network).astype('float32')
    predictions = model.predict(x=input_data_network)

    predictions = categorial_to_eval(predictions)

    for i in non_argumentative_data:
        predictions[i] = 0

    return predictions
