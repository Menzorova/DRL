"""Prioritized experience replay memory."""

import random
import numpy as np
from agents.data_structures import SumTree


class PERM:  # stored as ( s, a, r, s_ ) in SumTree

    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):

        self.tree = SumTree(capacity)
        self.capacity = capacity

    def get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, data):
        value = self.get_priority(error)
        self.tree.add(value, data)

    def sample(self, batch_size):

        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):

            left_border = segment * i
            right_border = segment * (i + 1)

            value = random.uniform(left_border, right_border)
            (idx, priority, data) = self.tree.get(value)
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):

        priority = self.get_priority(error)
        self.tree.update(idx, priority)
