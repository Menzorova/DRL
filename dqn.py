from agents import models
from agents import memory

import numpy as np
import pandas as pd

from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K

K.tensorflow_backend._get_available_gpus()


class DQN(object):
    def __init__(
        self,
        inputs,
        outputs,
        memory_size,
        discount_factor,
        learning_rate,
        model_name: str = "FCNN",
        memory_type: str = "ERM",
    ):
        assert hasattr(models, model_name)
        assert hasattr(memory, memory_type)

        self.inputs = inputs
        self.outputs = outputs
        self.memory_size = memory_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.memory_type = memory_type

        self.memory = getattr(memory, memory_type)(self.memory_size)

        self._initialize_nets()
        self._init_data_buffer()

    # Create nets
    def _build_net(self):
        model_name = self.model_name
        if model_name == "FCNN":
            return getattr(models, model_name)(self.inputs, self.outputs)
        if model_name == "CNN":
            return getattr(models, model_name)(self.outputs)

    def _initialize_nets(self):

        policy_net = self._build_net()
        policy_net.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        self.policy_net = policy_net

        target_net = self._build_net()
        target_net.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        self.target_net = target_net

        # initialize nets with the same weights
        self.backupNetwork(self.policy_net, self.target_net)

    # For data collecting
    def _init_data_buffer(self):

        df = pd.DataFrame(
            columns=[
                "Episode number",
                "Episode reward",
                "Number of steps",
                "Total number of steps",
                "Error",
            ]
        )
        self.df = df

    def SaveResults(self, episode, reward, step, decay_step, error):

        self.df = self.df.append(
            {
                "Episode number": episode,
                "Episode reward": reward,
                "Number of steps": step,
                "Total number of steps": decay_step,
                "Error": error,
            },
            ignore_index=True,
        )

        return self.df

    # Basic algorithm
    def select_action(self, state, exploration_rate):

        state = np.expand_dims(np.expand_dims(state, axis=0), axis=0)
        if np.random.random() > exploration_rate:
            action = np.argmax(self.policy_net.predict(state)[0])
        else:
            action = np.random.randint(0, self.outputs)

        return action

    def append_sample(self, state, action, reward, next_state, done):

        # calculate error
        state = np.expand_dims(np.expand_dims(state, axis=0), axis=0)
        next_state = np.expand_dims(np.expand_dims(next_state, axis=0), axis=0)
        current_prediction = self.policy_net.predict(state)
        current_value = current_prediction[0][action]
        target_value = self.target_net.predict(next_state)[0]

        if done:
            current_prediction[0][action] = reward

        else:
            current_prediction[0][action] = reward + self.discount_factor * np.max(
                target_value
            )

        error = abs(current_value - current_prediction[0][action])
        state = list(state.flatten())
        next_state = list(next_state.flatten())

        if self.memory_type == "ERM":
            self.memory.add(state, action, reward, next_state, done)
        elif self.memory_type == "PERM":
            self.memory.add(error, (state, action, reward, next_state, done))

    def train(self, batch_size):

        if self.memory_type == "ERM":
            transition_batch = self.memory.getMiniBatch(batch_size)
        elif self.memory_type == "PERM":
            transition_batch, idxs, is_weights = self.memory.sample(batch_size)

        state_batch = np.array(
            [np.expand_dims(transition[0], axis=0) for transition in transition_batch]
        )

        action_batch = np.array(
            [np.expand_dims(transition[1], axis=0) for transition in transition_batch]
        )
        reward_batch = np.array(
            [np.expand_dims(transition[2], axis=0) for transition in transition_batch]
        )
        next_state_batch = np.array(
            [np.expand_dims(transition[3], axis=0) for transition in transition_batch],
            ndmin=3,
        )
        done_batch = np.array(
            [np.expand_dims(transition[4], axis=0) for transition in transition_batch]
        )

        target_q_values = np.empty((0, self.outputs), dtype=np.float64)

        current_states = np.empty((0, 1, self.inputs), dtype=np.float64)

        errors = np.empty((0, len(transition_batch)), dtype=np.float64)

        for i in range(len(transition_batch)):

            q_values_state = self.policy_net.predict(
                np.expand_dims(state_batch[i], axis=0)
            )
            q_values_next_state = self.target_net.predict(
                np.expand_dims(next_state_batch[i], axis=0)
            )

            if done_batch[i]:
                q_value_expected = reward_batch[i]
            else:
                q_value_expected = reward_batch[i] + self.discount_factor * np.max(
                    q_values_next_state
                )

            # update inputs for model training
            current_states = np.append(current_states, [state_batch[i]].copy(), axis=0)

            q_values_new = q_values_state[0].copy()
            q_values_new[action_batch[i]] = q_value_expected
            q_values_new = np.expand_dims(q_values_new, axis=0)
            target_q_values = np.append(target_q_values, q_values_new.copy(), axis=0)

            error = abs(q_values_state[0][action_batch[i]] - q_value_expected)
            errors = np.append(errors, error)

        if self.memory_type == "ERM":
            train_error = self.policy_net.train_on_batch(
                current_states, target_q_values
            )
        elif self.memory_type == "PERM":

            for i in range(batch_size):
                idx = idxs[i]
                self.memory.update(idx, errors[i])

            train_error = self.policy_net.train_on_batch(
                current_states, target_q_values, sample_weight=is_weights
            )

        return train_error

    def updateTargetNetwork(self):
        self.backupNetwork(self.policy_net, self.target_net)

    def backupNetwork(self, policy_net, target_net):
        weight_matrix = []

        for layer in policy_net.layers:
            weights = layer.get_weights()
            weight_matrix.append(weights)

        for i, layer in enumerate(target_net.layers):
            weights = weight_matrix[i]
            layer.set_weights(weights)

    # save models
    def saveModel(self, path):
        self.policy_net.save(path)

    def saveTargetModel(self, path):
        self.target_net.save(path)

    # load models
    def loadWeights(self, path):
        self.policy_net.set_weights(load_model(path).get_weights())

    def loadWeightsTM(self, path):
        self.target_net.set_weights(load_model(path).get_weights())
