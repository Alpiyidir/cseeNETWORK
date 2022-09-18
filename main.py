import os

os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

videos = pd.read_csv("./data/final.csv")
videos_clean = videos.copy().dropna()
columns = list(videos_clean.columns.values)
for column in columns:
    videos_clean = videos_clean.astype({column: "int"})

videos_features = videos_clean.copy()
videos_labels = videos_features.pop("view_count")

inputs = dict()
for name, column in videos_features.items():
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype="int32")

x = layers.Concatenate()(list(inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(videos_clean[inputs.keys()]))

preprocessed_inputs = norm(x)

videos_preprocessing = keras.Model(inputs, preprocessed_inputs)

videos_features_dict = {name: np.array(value)
                        for name, value in videos_features.items()}

videos_dict = {name: values[:1] for name, values in videos_features_dict.items()}
videos_preprocessing(videos_dict)


def videos_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(1024),
        layers.Dense(1024),
        layers.Dense(1024),layers.Dense(1024),layers.Dense(1024),layers.Dense(1024),
        layers.Dense(1)
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    return model


videos_model = videos_model(videos_preprocessing, inputs)

videos_model.fit(x=videos_features_dict, y=videos_labels, epochs=1000)
