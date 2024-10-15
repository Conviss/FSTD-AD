import torch
import torch.nn as nn

from model.FDA import FDA
from model.getTrend import GetTrend
from model.vae import Encoder, Decoder, Embedding


class TEST(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim,
                 feature_num, time_num, block_num, head_num, dropout, device, hidden_dim, kernel_size, stride, epsilon):
        super(TEST, self).__init__()

        self.device = device
        self.window_size = window_size
        self.feature_num = feature_num
        self.epsilon = epsilon
        self.hidden_dim = hidden_dim

        self.fda = FDA(
            window_size=window_size,
            feature_num=feature_num,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.x_embedding = Embedding(
            device=device,
            window_size=window_size,
            feature_num=3*feature_num,
            # feature_num=feature_num,
            model_dim=model_dim,
            time_num=time_num
        )

        self.y_embedding = Embedding(
            device=device,
            window_size=window_size,
            feature_num=3*feature_num,
            model_dim=model_dim,
            time_num=time_num
        )

        self.encoder = Encoder(
            device=device,
            window_size=window_size,
            feature_num=3*feature_num,
            model_dim=model_dim,
            ff_dim=ff_dim,
            atten_dim=atten_dim,
            hidden_dim=hidden_dim,
            block_num=block_num,
            head_num=head_num,
            time_num=time_num,
            dropout=dropout,
            epsilon=epsilon
        )

        self.decoder = Decoder(
            device=device,
            window_size=window_size,
            feature_num=feature_num,
            model_dim=model_dim,
            ff_dim=ff_dim,
            hidden_dim=hidden_dim,
        )

        self.GetTrend = GetTrend(
            device=device,
            window_size=window_size,
            feature_num=feature_num,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            ff_dim=ff_dim
        )

    def forward(self, data, time, p):

        disturb_data = data.clone()
        disturb = (torch.randn(data.shape[0], data.shape[2]) * p).to(self.device)
        disturb = disturb.unsqueeze(1).repeat(1, self.window_size, 1).float().to(self.device)
        disturb_data = disturb_data + disturb

        sample_noise = (self.epsilon * torch.rand(disturb_data.shape)).float().to(self.device)
        noise_data = disturb_data + sample_noise

        f_x, f_y = self.fda(noise_data, data)

        x, xt, xs = self.x_embedding(f_x, time)
        y, yt, ys = self.y_embedding(f_y, time)

        mu_z_y, logvar_z_y, _, _ = self.encoder(yt, ys)
        mu_z_x, logvar_z_x, z_x, xt = self.encoder(xt, xs)

        trend = self.GetTrend(x, xt)

        stable = self.decoder(z_x)

        trend = trend - disturb
        recon = stable + trend

        return mu_z_x, mu_z_y, logvar_z_x, logvar_z_y, z_x, stable, trend, recon
