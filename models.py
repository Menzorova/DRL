import tensorflow as tf

from keras import Sequential
from keras.layers import Dense, Activation, Flatten


class FCNN(Sequential):
    def __init__(self, n_inputs: int, n_outputs: int):
        layers = [
            Flatten(input_shape=(1, n_inputs)),
            Dense(512),
            Activation("relu"),
            Dense(512),
            Activation("relu"),
            Dense(256),
            Activation("relu"),
            Dense(64),
            Activation("relu"),
            Dense(n_outputs),
            Activation("linear"),
        ]
        super().__init__(layers=layers)


class CNN(tf.keras.layers.Layer):
    def __init__(self, n_outputs: int):
        super().__init__()

        self.n_outputs = n_outputs

        # normalize input
        self.bn0 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv1D(32, 4, strides=4, activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(32, 1, strides=1, activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.fc1 = tf.keras.layers.Dense(34, activation="relu")
        self.fc2 = tf.keras.layers.Dense(64, activation="relu")
        self.fc3 = tf.keras.layers.Dense(self.n_outputs, activation="linear")

    def call(self, input_tensor, training=False):

        full_inp = self.bn0(input_tensor)

        input_1 = full_inp[:, :, :2]
        input_2 = tf.transpose(full_inp[:, :, :8], perm=[0, 2, 1])

        x = self.conv1(input_2)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.expand_dims(x, axis=3)
        x = tf.nn.max_pool(x, [1, 2, 1, 1], 1, padding="VALID")
        x = tf.squeeze(x, axis=3)

        x = tf.concat(values=[input_1, x], axis=2)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
