import torch
from torch import nn
from torch.autograd import Variable

class LSTM(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out

class forecasterModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_lyrs = 1, do = .05, device = "cpu"):
        """Initialize the network architecture

        Args:
            input_dim ([int]): [Number of time lags to look at for current prediction]
            hidden_dim ([int]): [The dimension of RNN output]
            n_lyrs (int, optional): [Number of stacked RNN layers]. Defaults to 1.
            do (float, optional): [Dropout for regularization]. Defaults to .05.
        """
        super(forecasterModel, self).__init__()

        self.ip_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_lyrs
        self.dropout = do
        self.device = device

        self.rnn = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = n_lyrs, dropout = do)
        self.fc1 = nn.Linear(in_features = hidden_dim, out_features = int(hidden_dim / 2))
        self.act1 = nn.ReLU(inplace = True)
        self.bn1 = nn.BatchNorm1d(num_features = int(hidden_dim / 2))

        self.estimator = nn.Linear(in_features = int(hidden_dim / 2), out_features = 1)
        
    
    def init_hiddenState(self, bs):
        """Initialize the hidden state of RNN to all zeros

        Args:
            bs ([int]): [Batch size during training]
        """
        return torch.zeros(self.n_layers, bs, self.hidden_dim)

    def forward(self, input):
        """Define the forward propogation logic here

        Args:
            input ([Tensor]): [A 3-dimensional float tensor containing parameters]

        """
        bs = input.shape[1]
        hidden_state = self.init_hiddenState(bs).to(self.device)
        # out , _ = self.rnn(input, hidden_state)

        cell_state = hidden_state
        out, _ = self.rnn(input, (hidden_state, cell_state))

        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.act1(self.bn1(self.fc1(out)))
        out = self.estimator(out)
        
        return out
    
    def predict(self, input):
        """Makes prediction for the set of inputs provided and returns the same

        Args:
            input ([torch.Tensor]): [A tensor of inputs]
        """
        with torch.no_grad():
            predictions = self.forward(input)
        
        return predictions