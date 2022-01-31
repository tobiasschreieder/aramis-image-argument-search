import pandas as pd
from indexing.neural_net.arg_network import NArgumentModel


def analyse_network_features_arg(data: pd.DataFrame):
    cols_to_use_color = [
        'image_average_color_r',
        'image_average_color_g',
        'image_average_color_b',
        'image_dominant_color_r',
        'image_dominant_color_g',
        'image_dominant_color_b'
    ]

    cols_to_use_primary = [
        'image_percentage_green',
        'image_percentage_red',
        'image_percentage_blue',
        'image_percentage_yellow',
        'image_percentage_bright',
        'image_percentage_dark',
        'text_len',
        'text_sentiment_score',
        'text_sentiment_score_con',
        'image_type',
        'image_roi_area'
    ]

    cols_to_exclude = []

    feature_to_remove = [
        [],
        ['image_average_color_r', 'image_average_color_g', 'image_average_color_b'],
        ['image_dominant_color_r', 'image_dominant_color_g', 'image_dominant_color_b'],
        ['image_percentage_green', 'image_percentage_red'],
        ['image_percentage_blue', 'image_percentage_yellow'],
        ['image_percentage_bright', 'image_percentage_dark'],
        ['text_len'],
        ['text_sentiment_score', 'text_sentiment_score_con'],
        ['image_type'],
        ['image_roi_area']
    ]

    arg_model = NArgumentModel.get('clean_test_1', version=3)

    val_accuracy = []

    for i in range(len(feature_to_remove) + 1):
        print("analyse %s/%s" % (i, len(feature_to_remove)+1))

        cols_to_use_primary_temp = cols_to_use_primary.copy()
        cols_to_use_color_temp = cols_to_use_color.copy()

        for el in cols_to_exclude:
            if el in cols_to_use_primary_temp:
                cols_to_use_primary_temp.remove(el)
            if el in cols_to_use_color_temp:
                cols_to_use_color_temp.remove(el)

        if i == len(feature_to_remove):
            arg_model.set_use_textposition(False)
        else:
            arg_model.set_use_textposition(True)
            for el in feature_to_remove[i]:
                if el in cols_to_use_color_temp:
                    cols_to_use_color_temp.remove(el)
                if el in cols_to_use_primary_temp:
                    cols_to_use_primary_temp.remove(el)

        print(cols_to_use_color_temp)
        print(cols_to_use_primary_temp)

        arg_model.set_cols_color(cols_to_use_color_temp)
        arg_model.set_cols_primary(cols_to_use_primary_temp)

        val_accuracy_model = []

        for j in range(20):
            val_accuracy_single = arg_model.train(data, test=[])
            val_accuracy_single_avg = (val_accuracy_single[-1] + val_accuracy_single[-2] + val_accuracy_single[-3]) / 3
            val_accuracy_model.append(val_accuracy_single_avg)

        val_accuracy_model.sort()
        print(val_accuracy_model)
        val_accuracy_model_avg = sum(val_accuracy_model[4:-2]) / len(val_accuracy_model[4:-2])
        val_accuracy.append(val_accuracy_model_avg)

    print("FINISHED: ", val_accuracy)