import torch.nn as nn
import torch
import torch.nn.functional as F
from kan import KAN

class Net(nn.Module):
    """
    A class to represent a neural network model.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        #torch.set_default_dtype(torch.float64)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Network(nn.Module):
    """
    A class to represent a dynamic neural network model with dynamic number of layers based on the respective argument.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden = []
        self.hidden.append(nn.Linear(self.input_size, self.hidden_size))
        for i in range(self.num_layers):
            self.hidden.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        Forward pass of the dynamic neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        for i in range(self.num_layers):
            x = F.tanh(self.hidden[i](x))
        x = self.output(x)
        return x
    
class Kalm(nn.Module):
    """
    A class to represent a dynamic neural network model with dynamic number of layers based on the respective argument.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Kalm, self).__init__()
        #Fix the size of the NN
        self.size=[input_size]
        for _ in range(num_layers):
            self.size.append(hidden_size)
        self.size.append(output_size)

        self.ka=KAN(self.size,grid=10, k=3,noise_scale=0.25)#, grid_eps=1.0)
        self.ka.speed()

    def forward(self, x):
        return self.ka(x)
    def update_grid(self,x):
        self.ka.update_grid_from_samples(x)
        
class PinnA(nn.Module): # DISCARD IT, OUTPUT IS WRONG
    """
    A class to represent a Pinn model with adjusted output.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(PinnA, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden = []
        self.hidden.append(nn.Linear(self.input_size, self.hidden_size))
        for i in range(self.num_layers):
            self.hidden.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(self.hidden_size, self.output_size)
        #self.shortcut = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        """
        Forward pass of the PinnA model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        y = torch.tanh(self.hidden[0](x))
        for i in range(self.num_layers-1):
            y = torch.tanh(self.hidden[i+1](y))
        y = self.output(y)
        time = x[:,0].view(-1,1)
        #time = torch.where(time < 0.5, time, 0.5*torch.ones_like(time))
        y = x[:,1:] + y*time
        return y
    

class ResidualBlock(nn.Module):
    """
    A class to represent a residual block in a fully connected ResNet model.
    """
    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.activation = activation
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out += identity
        out = self.activation(out)
        return out

class FullyConnectedResNet(nn.Module):
    """
    A class to represent a fully connected ResNet model.
    """
    def __init__(self, input_size, hidden_size, output_size, num_blocks, num_layers_per_block, activation=nn.ReLU()):
        super(FullyConnectedResNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.activation = activation

        self.fc_input = nn.Linear(input_size, hidden_size)
        self.blocks = self._make_blocks()
        self.fc_output = nn.Linear(hidden_size, output_size)

    def _make_blocks(self):
        blocks = []
        for _ in range(self.num_blocks):
            block_layers = []
            in_features = self.hidden_size
            for _ in range(self.num_layers_per_block):
                block_layers.append(ResidualBlock(in_features, self.hidden_size, self.activation))
                in_features = self.hidden_size
            blocks.append(nn.Sequential(*block_layers))
        return nn.ModuleList(blocks)

    def forward(self, x):
        x = self.fc_input(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc_output(x)
        return x




