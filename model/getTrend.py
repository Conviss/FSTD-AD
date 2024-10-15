import torch
from torch import nn

from model.embedding import DataEmbedding
import torch.nn.functional as F


class GetTrend(nn.Module):
    def __init__(self, device, window_size, feature_num, model_dim, hidden_dim, ff_dim):
        super(GetTrend, self).__init__()

        self.data_embedding = DataEmbedding(device, window_size, model_dim, feature_num)
        self.device = device
        self.model_dim = model_dim
        self.num_layers = 2
        self.num_directions = 1

        self.lstm = nn.LSTM(model_dim, model_dim, self.num_layers, batch_first=True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        self.conv3 = nn.Conv1d(in_channels=model_dim, out_channels=hidden_dim, kernel_size=(1,))
        self.conv4 = nn.Conv1d(in_channels=hidden_dim, out_channels=model_dim, kernel_size=(1,))
        self.conv5 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv6 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        self.conv7 = nn.Conv1d(in_channels=model_dim, out_channels=feature_num, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv3.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv4.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv5.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv6.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv7.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.ae = nn.Sequential(
            self.conv1,
            nn.GELU(),
            self.conv2,
            nn.GELU(),
            self.conv3,
            nn.GELU(),
            self.conv4,
            nn.GELU(),
            self.conv5,
            nn.GELU(),
            self.conv6,
            nn.GELU(),
            self.conv7,
        )

    def forward(self, data, xt):

        x = data - xt
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.model_dim).to(self.device)
        c_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.model_dim).to(self.device)
        x, _ = self.lstm(x, (h_0, c_0))

        x = self.ae(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x