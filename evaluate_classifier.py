#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import wandb
import ray
import pdb
import pickle
import os

from dataset.cube_dataset import *
from tensorflow import keras
from models import *
from sklearn.model_selection import train_test_split

def main():
    ray.init(num_cpus=1)

    wandb.init(
        project="afa_sc",
        entity="ohs",
        job_type="evaluate_fs",
    )
    wandb.config = {
        "gamma": 50,
    }

    with open("./dataset/coviddata.pkl", "rb") as file:
        data = pickle.load(file)

    cl = {'healthy': 0, 'ward': 1, 'icu': 2}
    data["label"] = data["label"].map(cl)

    X = data.drop('label', axis=1).values
    y = data.values[:, -1]

    train_input, test_input, train_output, test_output = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

    num_train_samples = train_input.shape[0]
    num_features = train_input.shape[1]

    cla_artifact_dir = wandb.run.use_artifact('classifier:latest').download()
    class_model = keras.models.load_model(os.path.join(cla_artifact_dir, 'classifier.h5'))

    b = np.zeros_like(test_input)
    # b = np.ones_like(test_input)

    # PCA
    # b[:] = np.where(np.isin(data.drop('label', axis=1).columns,
    #                         ['IL-4', 'IL-10', 'CCR7', 'CTLA-4', 'GATA3', 'PD-1']), 1, 0)
    # b[:] = np.where(np.isin(data.drop('label', axis=1).columns,
    #                         ['CD8', 'IFNg', 'CTLA-4', 'TNFa', 'CCR7', 'IL-10']), 1, 0)

    # PREP
    # b[:] = np.where(np.isin(data.drop('label', axis=1).columns,
    #                         ['IL-4', 'CD25', 'IL-17a', 'HLA-DR', 'GATA3', 'CD14']), 1, 0)

    # Chi-Squared
    # b[:] = np.where(np.isin(data.drop('label', axis=1).columns,
    #                         ['IFNg', 'CD25', 'CD8', 'CTLA-4', 'IL-10', 'CCR7']), 1, 0)

    # Recursive Feature Elimination
    b[:] = np.where(np.isin(data.drop('label', axis=1).columns,
                            ['IL-17a', 'CCR7', 'GATA3', 'CD4', 'CD25', 'IL-4']), 1, 0)

    m = test_input * b
    mb = np.block([m, b])

    probs = class_model.predict(mb)
    pred = tf.math.argmax(probs, axis=1).numpy()
    accuracy = np.sum(pred == test_output)/test_output.size

    print(accuracy)


if __name__ == '__main__':
    main()
