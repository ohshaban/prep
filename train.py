import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import ray
import pdb
import sys
import os
import wandb

from masking import get_add_mask_fn, UniformMaskGenerator
from dataset.cube_dataset import *
from tensorflow import keras
from models import *
from sklearn.model_selection import train_test_split

def create_models(input_shape, output_shape, num_features):

    gains_net = keras.Sequential([
        keras.layers.Dense(128),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(64),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(32),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(num_features, activation='softplus')
    ], name="gains_net")

    classifier = keras.Sequential([
        keras.layers.Dense(128, input_shape=(input_shape,), activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(output_shape, activation='softmax'),
    ], name="classifier")

    obs = keras.layers.Input(shape=input_shape)
    info_gain_output = gains_net(obs)
    classifier_output = classifier(obs)

    gains_model = GainsModel(inputs = obs, outputs = info_gain_output)
    gains_model.compile(optimizer="adam")

    class_model = keras.models.Model(inputs = obs, outputs = classifier_output)
    class_model.compile("adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics="accuracy")

    return gains_model, class_model


def train_class_model(model, dataset, epochs):
    # train_input, train_output = ray.get(dataset.get_training_data.remote())
    # validation_input, validation_output = ray.get(dataset.get_validation_data.remote())

    train_input, train_output = dataset

    training_losses = []
    validation_losses = []
    validation_accuracies = []

    low_mask = get_add_mask_fn(UniformMaskGenerator(bounds=(0.0, 0.5)))
    mid_mask = get_add_mask_fn(UniformMaskGenerator(bounds=(0.2, 0.7)))
    high_mask = get_add_mask_fn(UniformMaskGenerator(bounds=(0.5, 1.0)))

    mask_fns = (low_mask, mid_mask, high_mask)

    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        # Sample features to be observed and apply appropriate mask (new features each training iter)
        mask_index = np.random.choice(3, p=[0.3, 0.5, 0.2])

        x, b = mask_fns[2](train_input)
        # v, vb = add_mask_fn(validation_input)

        m = x*b
        mb = np.block([m, b])

        # vm = v*vb
        # vmb =  np.block([vm, vb])

        training_history = model.fit(mb, train_output, callbacks=[wandb.keras.WandbCallback()])
        # print("#### Validation Results ####")
        # validation_loss = model.evaluate(vmb, validation_output, verbose=2)
        training_losses.append(training_history.history['loss'][0])
        # if type(validation_loss) == float:
        #     validation_losses.append(validation_loss)
        # else:
        #     validation_losses.append(validation_loss[0])
        #     validation_accuracies.append(validation_loss[1])

        print()

    return training_losses, validation_losses, validation_accuracies, model


def train_gains_model(gains_model, class_model, dataset, num_train_samples, num_features, epochs):
    # train_input, _ = ray.get(dataset.get_training_data.remote())
    train_input, _ = dataset
    # validation_input, _ = ray.get(dataset.get_validation_data.remote())

    training_losses = []
    validation_losses = []
    validation_accuracies = []

    rand_gen = np.random.RandomState()
    add_mask_fn = get_add_mask_fn(UniformMaskGenerator(bounds=(0.2, 0.7)))
    batch_size = num_train_samples // 4

    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        idx = rand_gen.randint(low=0, high=num_train_samples, size=batch_size)
        # batch = train_input[(i%4)*batch_size:((i%4)+1)*batch_size]
        batch = train_input[idx]

        # Sample features to be observed and apply appropriate mask (new features each training iter)
        # pdb.set_trace()
        x, b = add_mask_fn(batch)

        m = x*b
        mb = np.block([m, b])
        # vm = np.array([b]*len(validation_input))*validation_input

        y = np.zeros([batch_size, num_features])
        # y = np.zeros([num_train_samples, num_features+1])
        # validation_output = np.zeros([len(validation_input), num_features])

        for i in range(num_features):
            bb = np.copy(b)
            bb[:, i] = 1
            mm = np.copy(m)
            mm[:, i] = batch[:, i]
            mmbb = np.block([mm, bb])

            y[:, i] = info_gain(class_model.predict(mb), class_model.predict(mmbb))
            # validation_xi = np.copy(validation_masked)
            # validation_xi[:, i] = validation_input[:, 0]
            # validation_output[:, i] = kl_div(class_model.predict(validation_xi), class_model.predict(validation_masked))

        # pdb.set_trace()
        # gamma = 1.75
        # y[:, num_features] = tf.reduce_sum(tf.nn.softmax(gamma*y[:, :num_features]) * y[:, :num_features], axis=1)
        # y[:, num_features] = 0.135
        # y[:, num_features] = np.where(np.max(class_model.predict(mb), axis=1) > 0.9, 1, 0)
        # pdb.set_trace()

        training_history = gains_model.fit(mb, y, callbacks=[wandb.keras.WandbCallback()])
        # print("#### Validation Results ####")
        # validation_loss = gains_model.evaluate(validation_masked, validation_output, verbose=2)
        # pdb.set_trace()
        training_losses.append(training_history.history['loss'][0])
        # if type(validation_loss) == float:
        #     validation_losses.append(validation_loss)
        # else:
        #     validation_losses.append(validation_loss[0])
        #     validation_accuracies.append(validation_loss[1])

        print()

    return training_losses, validation_losses, validation_accuracies, gains_model


# Test trained model
def test_model(model, dataset, num_features):
    print()
    print("#### Final Testing Results ####")
    # x, y = ray.get(dataset.get_test_data.remote())
    x, y = dataset

    add_mask_fn = get_add_mask_fn(UniformMaskGenerator(bounds=(0.2, 0.7)))
    x, b = add_mask_fn(x)
    # p = np.random.uniform()
    # b = np.array([np.random.choice([1, 0], size=num_features, p=[p, 1-p])]*len(x))
    m = b*x
    mb = np.block([m, b])

    loss = model.evaluate(mb, y, verbose=2)
    print(loss)


# Calculate KL-Divergence between rows of p and q
# Each row corresponds to a probability distribution
def kl_div(dp, dq):
    return tf.reduce_sum(np.where((dp != 0) & (dq != 0), dp*(tf.math.log(dp) - tf.math.log(dq)), 0), axis=1)


def info_gain(dp, dq):
    return tf.reduce_sum(-dp * tf.math.log(dp) + dq * tf.math.log(dq), axis=1)


if __name__ == '__main__':
    ray.init(num_cpus=1)

    wandb.init(
        project="afa_sc",
        entity="ohs",
        job_type="train_mlp_classifier",
    )
    wandb.config = {
        "epochs": 150,
    }

    # dataset = CubeDataset.remote('../dataset/cube_20_0.3.pkl')
    # num_features, num_labels, num_train_samples = ray.get(dataset.get_attributes.remote())
    with open("./dataset/coviddata.pkl", "rb") as file:
        data = pickle.load(file)

    cl = {'healthy': 0, 'ward': 1, 'icu': 2}
    data["label"] = data["label"].map(cl)

    X = data.drop('label', axis=1).values
    y = data.values[:, -1]

    train_input, test_input, train_output, test_output = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    training = (train_input, train_output)
    testing = (test_input, test_output)

    num_train_samples = train_input.shape[0]
    num_features = train_input.shape[1]

    input_shape = (2*num_features)
    output_shape = np.unique(y).size

    gains_model, class_model = create_models(input_shape=input_shape, output_shape=output_shape, num_features=num_features)

    epochs = 250
    training_loss, validation_loss, validation_accuracy, class_model = train_class_model(class_model, training, epochs)
    # artifact = wandb.run.use_artifact('classifier:latest')
    # artifact_dir = artifact.download()
    # class_model = keras.models.load_model(os.path.join(artifact_dir, 'classifier.h5'))
    # training_loss, validation_loss, validation_accuracy, gains_model = train_gains_model(gains_model, class_model, training, num_train_samples, num_features, epochs)
    # gains_model = keras.models.load_model('../pretraining/gains_net.h5')

    # test_model(class_model, testing, num_features)
    # pdb.set_trace()

    # Save models
    class_model.save(os.path.join(wandb.run.dir, 'classifier.h5'))
    # gains_model.save(os.path.join(wandb.run.dir, 'gains_net.h5'))
