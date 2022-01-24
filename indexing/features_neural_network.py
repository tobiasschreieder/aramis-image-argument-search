import math
import os
# to get no console-print from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model


overfitCallback = EarlyStopping(monitor='accuracy', min_delta=0, patience=30)


def train_network(model_name: str, df: pd.DataFrame):
    """
    Method to train a neural network
    :param model_name: the model_name for the folder to save the network
    :return: None
    """

    print("start scaling dataframe")
    for index, row in df.iterrows():
        df.loc[index, :] = scale_data(row)
    print("finished scaling")

    # shuffle data-rows
    df.sample(frac=1)

    df_len = len(df.index)
    split_index = round(df_len * 0.8)

    df_train = df.iloc[:split_index, :]
    df_test = df.iloc[split_index:, :]

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

    model = Sequential()
    model.add(Dense(40, input_dim=input_dim, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation="relu"))

    model.compile(loss="mse", optimizer="Adam", metrics=["accuracy"])
    history = model.fit(x, y, epochs=600, batch_size=10, validation_data=(x_test, y_test), callbacks=[overfitCallback])

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


def make_prediction(model_name: str, input_features: pd.Series) -> list:
    """
    Method to predict some steps for a runners level
    :param model_name: name of the trained network
    :return: a list of all predictions. Every prediction is a list of 40 values due to the 40-nodes-last-layer
    """

    my_model = load_model('indexing/models/' + str(model_name) + '/model.hS', compile=False)

    input_features = scale_data(input_features)
    x = np.array(input_features)

    predictions = my_model.predict(np.array([x, ]))

    print(predictions[0][0])
    return predictions[0][0]


def log_normal_density_function(x: float) -> float:
    if x == 0:
        return 0
    elif x == 1:
        return 0
    else:
        return ((1 / (math.sqrt(2 * math.pi) * 0.16 * (-x + 1))) * math.exp(
            ((math.log((-x + 1), 10) + 0.49) ** 2) / -0.0512) * 0.12)


def scale_data(df_row: pd.Series) -> pd.DataFrame:
    '''
    'image_id': pd.StringDtype(),
    'html_sentiment_score': np.float,
    'image_text_len': np.int,
    'image_text_sentiment_score': np.float,
    'image_percentage_green': np.float,
    'image_percentage_red': np.float,
    'image_percentage_bright': np.float,
    'image_percentage_dark': np.float,
    'image_average_color_r': np.float,
    'image_average_color_g': np.float,
    'image_average_color_b': np.float,
    'image_type': np.int8,
    'image_roi_area': np.float,
    '''

    df_row['html_sentiment_score'] = (df_row['html_sentiment_score'] + 1) / 2
    df_row['image_text_len'] = (1 - (1 / (math.exp(0.01 * df_row['image_text_len'])))) * 3
    df_row['image_text_sentiment_score'] = (df_row['image_text_sentiment_score'] + 1) / 2
    df_row['image_percentage_green'] = df_row['image_percentage_green'] / 100
    df_row['image_percentage_red'] = df_row['image_percentage_red'] / 100
    df_row['image_percentage_bright'] = df_row['image_percentage_bright'] / 100
    df_row['image_percentage_dark'] = df_row['image_percentage_dark'] / 100
    df_row['image_average_color_r'] = df_row['image_average_color_r'] / 360
    df_row['image_average_color_g'] = df_row['image_average_color_g'] / 360
    df_row['image_average_color_b'] = df_row['image_average_color_b'] / 360

    return df_row
