import numpy as np


class SumTree(object):

    def __init__(self, capacity):
        self.capacity = capacity
        # number of nodes in the tree
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    # leaf node retrieval function
    # idx - index of the top-parent node as the first arguments
    # value - random sampled value
    def retrieve(self, idx, value):

        # left/right - left/right idx of elements following after idx
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self.retrieve(left, value)
        else:
            return self.retrieve(right, value - self.tree[left])

    # update priority
    def update(self, idx, new_value):

        change = new_value - self.tree[idx]
        self.tree[idx] = new_value
        self.propagate_changes(idx, change)

    def propagate_changes(self, idx, change):
 
        # calculate parent id
        parent = (idx - 1) // 2
        self.tree[parent] += change

        # if we not in root propogate changes
        if parent != 0:
            self.propagate_changes(parent, change)

    def add(self, value, data):

        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, value)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # get priority and sample
    def get(self, value):

        idx = self.retrieve(0, value)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def total(self):
        return self.tree[0]
