#!/usr/bin/env python3

import pdb
import tensorflow as tf
import numpy as np
from tensorflow import keras

loss_tracker = keras.metrics.Mean(name="loss")

class GainsModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        bce = keras.losses.BinaryCrossentropy(from_logits=True)
        x, y = data
        num_features = x.shape[-1]//2
        b = x[:, -num_features:]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # pdb.set_trace()
            # loss = (tf.reduce_sum((y_pred[:, :-1] - y[:, :-1]) ** 2 * (1 - b)) + tf.reduce_sum((y_pred[:, -1] - y[:, -1]) ** 2)) / (tf.reduce_sum(1 - b) + tf.math.count_nonzero(y[:, -1], dtype=tf.dtypes.float32))
            # loss = (tf.reduce_sum((y_pred[:, :-1] - y[:, :-1]) ** 2 * (1 - b)) / tf.reduce_sum(1 - b)) + (bce(y[:, -1], y_pred[:, -1]) / 1)
            loss = tf.reduce_sum((y_pred - y) ** 2 * (1 - b)) / tf.reduce_sum(1 - b)


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}
