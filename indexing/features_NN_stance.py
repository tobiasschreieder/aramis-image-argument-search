import math
import os

# to get no console-print from tensorflow
from keras.layers import BatchNormalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import string as tf_string
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Sequential, load_model, Model
import tensorflow_hub as hub
import tensorflow_text

overfitCallback = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15)


def train_network(model_name: str, df: pd.DataFrame):
    '''
    model_argument = load_model("indexing/models/test_1_argument/model.hS")
    input_data = [row for index, row in df.iterrows()]
    results_argument = features_NN_argument.make_prediction(model=model_argument, input_data=input_data)
    df['argumentativeness'] = results_argument

    df = df.loc[df['argumentativeness'] > 0.4]
    '''

    # --- Testing with ALBERT

    albert_url = 'https://tfhub.dev/tensorflow/albert_en_base/2'
    encoder = hub.KerasLayer(albert_url)
    preprocessor_url = "https://tfhub.dev/tensorflow/albert_en_preprocess/3"
    preprocessor = hub.KerasLayer(preprocessor_url)

    text_input_html = Input(shape=(), dtype=tf_string)
    encoder_inputs_html = preprocessor(text_input_html)
    outputs_html = encoder(encoder_inputs_html)
    pooled_output_html = outputs_html["pooled_output"]
    embedding_model_html = Model(text_input_html, pooled_output_html)

    model_html = Sequential()
    model_html.add(embedding_model_html)
    model_html.add(Dense(128, activation='relu'))
    model_html.add(BatchNormalization())
    model_html.add(Dense(30, activation='relu'))
    model_html.add(BatchNormalization())
    model_html.add(Dense(8, activation='relu'))

    # ---

    model_name = model_name + "_stance"

    df['html_sentiment_score_con'] = 0
    df['text_sentiment_score_con'] = 0
    df['query_sentiment_con'] = 0
    df['query_html_context_con'] = 0
    df['query_image_context_con'] = 0

    # --- handling the scaling, train and test-data
    print("start scaling dataframe")
    for index, row in df.iterrows():
        df.loc[index, :] = preprocess_data(row)
    print("finished scaling")

    column_list = ['image_percentage_bright',
                   'image_percentage_dark',
                   'html_sentiment_score',
                   'html_sentiment_score_con',
                   'text_len',
                   'text_sentiment_score',
                   'text_sentiment_score_con',
                   'query_sentiment',
                   'query_sentiment_con',
                   'query_html_eq',
                   'query_image_eq',
                   'query_html_context',
                   'query_html_context_con',
                   'query_image_context',
                   'query_image_context_con',
                   'query_image_align']

    split_data = split_train_test_data(df, column_list)
    x_train = split_data['x_train']
    x_test = split_data['x_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']

    x_train_query = split_data['x_train_query']
    x_train_html = split_data['x_train_html']
    x_test_query = split_data['x_test_query']
    x_test_html = split_data['x_test_html']

    primary_inputs = Input(shape=len(x_train[0]))

    combinedInput = concatenate([model_html.output, primary_inputs])
    x = Dense(15, activation="relu")(combinedInput)
    x = Dense(8, activation="relu")(x)
    x = Dense(3, activation="softmax")(x)

    model = Model(inputs=[model_html.input, primary_inputs], outputs=x)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    history = model.fit(x=[x_train_html, x_train], y=y_train,
                        epochs=10, batch_size=50,
                        validation_data=([x_test_html, x_test], y_test))
    # callbacks=[overfitCallback])

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

    df_test = df.loc[df['topic'].isin([33, 27, 31])]
    df_train = df.loc[~df['topic'].isin([33, 27, 31])]

    y_train = df_train['stance_eval']
    y_train = eval_to_categorial(y_train)

    y_test = df_test['stance_eval']
    y_test = eval_to_categorial(y_test)

    x_train = df_train[column_list]
    x_train = np.asarray(x_train).astype('float32')

    x_test = df_test[column_list]
    x_test = np.asarray(x_test).astype('float32')

    x_train_query = df_train['query_string']
    x_test_query = df_test['query_string']
    x_train_html = df_train['html_string']
    x_test_html = df_test['html_string']

    return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test, 'df_eval': df_eval,
            'x_train_query': x_train_query, 'x_train_html': x_train_html,
            'x_test_query': x_test_query, 'x_test_html': x_test_html}


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

        '''
        if value_1 > value_2*1.1 and value_1 > value_0*1.1:
            result = 0
        else:
            result = (value_0 * (-1)) + (value_2 * 1)
        output_data.append(result)
        '''

        if value_0 >= value_1 and value_0 > value_2:
            output_data.append(1)
        elif value_2 >= value_0 and value_2 > value_1:
            output_data.append(-1)
        else:
            output_data.append(0)

    return np.asarray(output_data)


def log_normal_density_function(x: float) -> float:
    if x == 0:
        return 0
    elif x == 1:
        return 0
    else:
        return ((1 / (math.sqrt(2 * math.pi) * 0.16 * (-x + 1))) * math.exp(
            ((math.log((-x + 1), 10) + 0.49) ** 2) / -0.0512) * 0.12)


def preprocess_data(df_row: pd.Series) -> pd.Series:
    df_row['text_len'] = (1 - (1 / (math.exp(0.01 * df_row['text_len'])))) * 3
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
    if df_row['query_sentiment'] < 0:
        df_row['query_sentiment_con'] = df_row['query_sentiment'] * (-1)
        df_row['query_sentiment'] = 0
    if df_row['text_sentiment_score'] < 0:
        df_row['text_sentiment_score_con'] = df_row['text_sentiment_score'] * (-1)
        df_row['text_sentiment_score'] = 0
    if df_row['html_sentiment_score'] < 0:
        df_row['html_sentiment_score_con'] = df_row['html_sentiment_score'] * (-1)
        df_row['html_sentiment_score'] = 0
    if df_row['query_html_context'] < 0:
        df_row['query_html_context_con'] = df_row['query_html_context'] * (-1)
        df_row['query_html_context'] = 0

    return df_row


def make_prediction(model: keras.Model, input_data: list):
    # model_argument = load_model("indexing/models/test_1_argument/model.hS")
    # results_argument = features_NN_argument.make_prediction(model=model_argument, input_data=input_data)

    column_list = ['image_percentage_bright',
                   'image_percentage_dark',
                   'html_sentiment_score',
                   'html_sentiment_score_con',
                   'text_len',
                   'text_sentiment_score',
                   'text_sentiment_score_con',
                   'query_sentiment',
                   'query_sentiment_con',
                   'query_html_eq',
                   'query_image_eq',
                   'query_html_context',
                   'query_html_context_con',
                   'query_image_context',
                   'query_image_context_con',
                   'query_image_align']

    input_data_network = []
    for i in range(len(input_data)):
        input_data[i]['html_sentiment_score_con'] = 0
        input_data[i]['text_sentiment_score_con'] = 0
        input_data[i]['query_sentiment_con'] = 0
        input_data[i]['query_html_context_con'] = 0
        input_data[i]['query_image_context_con'] = 0
        scaled_row = preprocess_data(input_data[i])
        row_input = scaled_row[column_list]
        input_data_network.append(row_input)

    input_data_network = np.asarray(input_data_network).astype('float32')
    predictions = model.predict(x=input_data_network)

    predictions = categorial_to_eval(predictions)

    return predictions
