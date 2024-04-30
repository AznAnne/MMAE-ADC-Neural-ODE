import torch
import torch.nn as nn
import torch.nn.functional


INIT_STD = 0.001  # standard deviation of the initial weights
CLASSIFIER_HIDDEN_DIM = 32  # classifier's hidden dimension
ODE_CONSTANT_VAL = 0.5  # ODE function's linear components use a different init for their constant offsets


def reverse(tensor):
    idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
    return tensor[idx]


def init_network_weights(net, std):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=INIT_STD)
                nn.init.constant_(m.bias, val=ODE_CONSTANT_VAL)

    def forward(self, t, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.hiddens_to_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        init_network_weights(self.hiddens_to_output, std=INIT_STD)

        self.rnn = nn.GRU(self.input_dim, self.hidden_dim)

    def forward(self, data):
        data = data.permute(1, 0, 2)
        data = reverse(data)
        output_rnn, _ = self.rnn(data)
        outputs = self.hiddens_to_output(output_rnn[-1])
        return outputs


class Classifier(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 20, CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(CLASSIFIER_HIDDEN_DIM, output_dim),
        )
        init_network_weights(self.net, std=INIT_STD)

    def forward(self, z, cmax_time):
        cmax_time = cmax_time.repeat(z.size(0), 1, 1)
        z = torch.cat([z, cmax_time], 2)
        return self.net(z)
