"""
Deep Q Network
"""
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))
#
#
# class ReplayMemory(object):
#
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0
#
#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)

class DQN:
    def __init__(self):
        pass

    def compile(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def summary(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def plot_curves(self):
        pass