import torch.nn as nn
import torch.nn.functional as F
import torch
"""the most common approach for creating a NN I've seen"""
class DeepNeuralNetwork(nn.Module):
    """Class definition for the Deep Neural network"""
    def __init__(self, n_observations, n_actions):
        # init the constructor of the inherited class
        super(DeepNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 128)
        self.layer6 = nn.Linear(128, n_actions)
    
    def forward(self, x):

        # forward pass through the neural network
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return F.relu(self.layer6(x))