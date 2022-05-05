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
        job_type="evaluate_mlp",
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

    input_shape = (2*num_features)
    output_shape = np.unique(y).size
    num_samples = 1

    cla_artifact_dir = wandb.run.use_artifact('classifier:latest').download()
    reg_artifact_dir = wandb.run.use_artifact('gains_net:latest').download()

    # class_model = keras.models.load_model('./pretraining/classifier.h5')
    class_model = keras.models.load_model(os.path.join(cla_artifact_dir, 'classifier.h5'))
    # gains_model = keras.models.load_model('./pretraining/gains_net.h5', custom_objects={'GainsModel': GainsModel})
    gains_model = keras.models.load_model(os.path.join(reg_artifact_dir, 'gains_net.h5'), custom_objects={'GainsModel': GainsModel})

    # test_input, test_output = ray.get(dataset.get_test_data.remote())
    # test_input, test_output = ray.get(dataset.sample_training_data.remote())

    num_acquisitions = []
    trajectories = []
    num_correct = 0

    gamma = 50

    # for i in range(len(test_input)):
    for i in range(10000):
        x = np.array([test_input[i]])
        y = np.array([test_output[i]])

        xb = np.block([x, np.ones_like(x)])

        b = np.zeros_like(x)
        m = x * b
        mb = np.block([m, b])

        actions = []
        for j in range(num_features):
            gains = gains_model.predict(mb)[0]
            if actions == []:
                stop_threshold = np.mean(gains)
            for action in actions:
                gains[action] = -99999
            # pdb.set_trace()
            p = tf.nn.softmax(gamma * np.append(gains, stop_threshold))

            # p = tf.nn.softmax(gamma * gains[:-1])
            # p = np.asarray(p).astype('float64')
            # p = (p / np.sum(p)) * (1 - tf.nn.sigmoid(gains[-1]))
            # a = np.random.choice(num_features+1, p=np.append(p, tf.nn.sigmoid(gains[-1])))

            # p = tf.nn.softmax(gamma * gains)
            p = np.asarray(p).astype('float64')
            p = p / np.sum(p)
            a = np.random.choice(num_features+1, p=p)

            a = np.argmax(tf.nn.softmax(np.append(gains, stop_threshold)))
            # print(f"action: {a}")
            probs = class_model.predict(mb)[0]
            prob = tf.reduce_max(probs)

            wandb.log({f"step_{j}_action": a,
                       f"step_{j}_max_prob": prob})

            if a == num_features or len(actions) == num_features:
                pred = tf.math.argmax(probs).numpy()
                # print(f"predicted class: {pred}")
                # print(probs)
                # print(f"actual class: {y[0]}")
                num_correct += (pred == int(y[0]))
                num_acquisitions.append(len(actions))
                break
            b[:, a] = np.ones(num_samples)
            m[:, a] = x[:, a]
            mb = np.block([m, b])
            actions.append(a)
        trajectories.append(actions)

        if (i+1) % 100 == 0:
            print(i+1)
            print(num_correct/(i+1))
            print(np.mean(num_acquisitions))

    with open(os.path.join(wandb.run.dir, "trajectories.pkl"), "wb") as fh:
        pickle.dump(trajectories, fh)

if __name__ == '__main__':
    main()
