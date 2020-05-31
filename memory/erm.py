"""Experience replay memory."""

import random


class ERM:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def getMiniBatch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def getCurrentSize(self):
        return len(self.memory)

    def getMemory(self, index):
        return self.memory[index]

    def make_transition_dict(self, state, action, reward, next_state, done):
        return [state, action, reward, next_state, done]

    def add(self, state, action, reward, next_state, done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.make_transition_dict(
            state, action, reward, next_state, done
        )
        self.position = (self.position + 1) % self.capacity
