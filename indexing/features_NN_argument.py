import math
import os
# to get no console-print from tensorflow
import keras
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, concatenate

from indexing import FeatureIndex

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
from pathlib import Path
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model

overfitCallback = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10)


def train_network(model_name: str, df: pd.DataFrame):
    model_name = model_name + "_argument"

    df['html_sentiment_score_con'] = 0
    df['text_sentiment_score_con'] = 0
    df['query_sentiment_con'] = 0

    # --- handling the scaling, train and test-data
    print("start scaling dataframe")
    for index, row in df.iterrows():
        df.loc[index, :] = scale_data(row)
    print("finished scaling")

    split_data = split_train_test_data(df)
    df_train = split_data['df_train']
    df_test = split_data['df_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']

    diff_model_data = get_diff_model_data(df_train, df_test)
    tp_input_train = diff_model_data['tp_input_train']
    tp_input_test = diff_model_data['tp_input_test']
    color_input_train = diff_model_data['color_input_train']
    color_input_test = diff_model_data['color_input_test']
    primary_input_train = diff_model_data['primary_input_train']
    primary_input_test = diff_model_data['primary_input_test']

    tp_input_shape = (len(tp_input_train[0]), len(tp_input_train[0][0]), 1)
    tp_model = create_text_position_cnn(tp_input_shape)

    color_model = create_color_model()

    primary_inputs = Input(shape=len(primary_input_train[0]))

    combinedInput = concatenate([tp_model.output, color_model.output, primary_inputs])
    x = Dense(10, activation="relu")(combinedInput)
    x = Dense(5, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[tp_model.input, color_model.input, primary_inputs], outputs=x)

    model.compile(loss="mse", optimizer="Adam", metrics=["accuracy"])

    history = model.fit(x=[tp_input_train, color_input_train, primary_input_train], y=y_train,
                        epochs=100, batch_size=50,
                        validation_data=([tp_input_test, color_input_test, primary_input_test], y_test),
                        callbacks=[overfitCallback])

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


def create_color_model() -> Model:
    color_model = Sequential()
    color_model.add(Dense(3, input_dim=6, activation='relu'))
    return color_model


def create_text_position_cnn(tp_input_shape) -> Model:
    print(tp_input_shape)

    model = Sequential([
        Conv2D(filters=1, kernel_size=2, input_shape=tp_input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(6, activation='relu'),
        Dense(3, activation='relu')
    ])

    return model


def split_train_test_data(df: pd.DataFrame):
    # shuffle data-rows
    df.sample(frac=1)

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

    df_test = df_test.drop('topic', 1)
    df_train = df_train.drop('topic', 1)

    y_train = df_train['arg_eval']
    y_train = np.asarray(y_train)

    y_test = df_test['arg_eval']
    y_test = np.asarray(y_test)

    return {'df_train': df_train, 'df_test': df_test, 'y_train': y_train, 'y_test': y_test, 'df_eval': df_eval}


def get_diff_model_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    text_position_train = df_train['text_position']
    text_position_test = df_test['text_position']

    tp_input_train = []
    for str in text_position_train:
        list_from_str = FeatureIndex.convert_text_area_from_str(str)
        dim_heatmap = int(math.sqrt(len(list_from_str)))
        chunked_list = list()
        for i in range(0, len(list_from_str), dim_heatmap):
            chunked_list.append(list_from_str[i:i + dim_heatmap])
        tp_input_train.append(chunked_list)
    tp_input_train = np.asarray(tp_input_train)
    tp_input_train = np.expand_dims(tp_input_train, axis=3)

    tp_input_test = []
    for str in text_position_test:
        list_from_str = FeatureIndex.convert_text_area_from_str(str)
        dim_heatmap = int(math.sqrt(len(list_from_str)))
        chunked_list = list()
        for i in range(0, len(list_from_str), dim_heatmap):
            chunked_list.append(list_from_str[i:i + dim_heatmap])
        tp_input_test.append(chunked_list)
    tp_input_test = np.asarray(tp_input_test)
    tp_input_test = np.expand_dims(tp_input_test, axis=3)

    color_df_train = df_train[['image_average_color_r',
                               'image_average_color_g',
                               'image_average_color_b',
                               'image_dominant_color_r',
                               'image_dominant_color_g',
                               'image_dominant_color_b']]
    color_input_train = np.asarray(color_df_train)
    color_df_test = df_test[['image_average_color_r',
                             'image_average_color_g',
                             'image_average_color_b',
                             'image_dominant_color_r',
                             'image_dominant_color_g',
                             'image_dominant_color_b']]
    color_input_test = np.asarray(color_df_test)

    primary_features_train = df_train[['image_percentage_green',
                                       'image_percentage_red',
                                       'image_percentage_blue',
                                       'image_percentage_yellow',
                                       'image_percentage_bright',
                                       'image_percentage_dark',
                                       'html_sentiment_score',
                                       'html_sentiment_score_con',
                                       'text_len',
                                       'text_sentiment_score',
                                       'text_sentiment_score_con',
                                       'image_type',
                                       'image_roi_area']]
    primary_input_train = np.asarray(primary_features_train)
    primary_features_test = df_test[['image_percentage_green',
                                     'image_percentage_red',
                                     'image_percentage_blue',
                                     'image_percentage_yellow',
                                     'image_percentage_bright',
                                     'image_percentage_dark',
                                     'html_sentiment_score',
                                     'html_sentiment_score_con',
                                     'text_len',
                                     'text_sentiment_score',
                                     'text_sentiment_score_con',
                                     'image_type',
                                     'image_roi_area']]
    primary_input_test = np.asarray(primary_features_test)

    return {'tp_input_train': tp_input_train, 'tp_input_test': tp_input_test,
            'color_input_train': color_input_train, 'color_input_test': color_input_test,
            'primary_input_train': primary_input_train, 'primary_input_test': primary_input_test}


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
    if df_row['text_sentiment_score'] < 0:
        df_row['text_sentiment_score_con'] = df_row['text_sentiment_score'] * (-1)
        df_row['text_sentiment_score'] = 0
    if df_row['html_sentiment_score'] < 0:
        df_row['html_sentiment_score_con'] = df_row['html_sentiment_score'] * (-1)
        df_row['html_sentiment_score'] = 0

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

        row['html_sentiment_score_con'] = 0
        row['text_sentiment_score_con'] = 0
        row['query_sentiment_con'] = 0

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
                                    'html_sentiment_score_con',
                                    'text_len',
                                    'text_sentiment_score',
                                    'text_sentiment_score_con',
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
