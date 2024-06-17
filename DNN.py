import torch.nn as nn
import torch.nn.functional as F
import torch
"""the most common approach for creating a NN I've seen"""
class DeepNeuralNetwork(nn.Module):
    """Class definition for the Deep Neural network"""
    def __init__(self, n_observations, n_actions):
        # init the constructor of the inherited class
        super(DeepNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations*n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

        # NOTE: stimovanje hiperparametara, videti da li moze sa manjom mrezom da se progura
    
    def forward(self, x):

        # forward pass through the neural network
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
