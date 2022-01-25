import math
import os
# to get no console-print from tensorflow
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
from pathlib import Path
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model


overfitCallback = EarlyStopping(monitor='accuracy', min_delta=0, patience=5)


def train_network(model_name: str, df: pd.DataFrame):
    """
    Method to train a neural network
    :param model_name: the model_name for the folder to save the network
    :return: None
    """

    model_name = model_name + "_argument"

    print("start scaling dataframe")
    for index, row in df.iterrows():
        df.loc[index, :] = scale_data(row)
    print("finished scaling")

    # shuffle data-rows
    df.sample(frac=1)

    text_position = df['text_position']
    df = df.drop('text_position', 1)

    df_eval = df.loc[df['topic'].isin([2, 4])].drop('topic', 1)
    df = df.loc[~df['topic'].isin([2, 4])]

    '''
    df_len = len(df.index)
    split_index = round(df_len * 0.8)
    
    df_train = df.iloc[:split_index, :]
    df_test = df.iloc[split_index:, :]
    '''

    df_test = df.loc[df['topic'].isin([8, 21])]
    df_train = df.loc[~df['topic'].isin([8, 21])]

    print(df_test)
    print(df_train)

    df_test = df_test.drop('topic', 1)
    df_train = df_train.drop('topic', 1)

    x = df_train.loc[:, df_train.columns != 'arg_eval']
    x = x.loc[:, x.columns != 'image_id']
    x = np.asarray(x)

    y = df_train['arg_eval']
    y = np.asarray(y)

    x_test = df_test.loc[:, df_test.columns != 'arg_eval']
    x_test = x_test.loc[:, x_test.columns != 'image_id']
    x_test = np.asarray(x_test)

    y_test = df_test['arg_eval']
    y_test = np.asarray(y_test)

    input_dim = len(x[0])
    print("Current network ist trained with %s features." % (input_dim))

    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="mse", optimizer="Adam", metrics=["accuracy"])
    history = model.fit(x, y, epochs=100, batch_size=50, validation_data=(x_test, y_test), callbacks=[overfitCallback])

    Path("indexing/models/" + str(model_name)).mkdir(parents=True, exist_ok=True)

    # summarize history for accuracy
    print(history.history.keys())
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


def make_prediction(model: keras.Model, input_data: list) -> list:
    """
    Method to predict some steps for a runners level
    :param model_name: name of the trained network
    :return: a list of all predictions. Every prediction is a list of 40 values due to the 40-nodes-last-layer
    """

    input_data_scaled = []
    for row in input_data:
        row = row.drop('text_position')
        scaled_row = scale_data(row)
        input_data_scaled.append(scaled_row)

    x = np.array(input_data_scaled).astype('float32')

    predictions = model.predict(x)

    predictions = [value[0] for value in predictions]

    print(predictions)
    return predictions


def log_normal_density_function(x: float) -> float:
    if x == 0:
        return 0
    elif x == 1:
        return 0
    else:
        return ((1 / (math.sqrt(2 * math.pi) * 0.16 * (-x + 1))) * math.exp(
            ((math.log((-x + 1), 10) + 0.49) ** 2) / -0.0512) * 0.12)


def scale_data(df_row: pd.Series) -> pd.Series:

    df_row['html_sentiment_score'] = (df_row['html_sentiment_score'] + 1) / 2
    df_row['text_len'] = (1 - (1 / (math.exp(0.01 * df_row['text_len'])))) * 3
    df_row['text_sentiment_score'] = (df_row['text_sentiment_score'] + 1) / 2
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

    return df_row
