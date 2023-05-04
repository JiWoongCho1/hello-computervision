import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VanilaVAE(nn.Module):
    def __init__(self, in_channels : int, latent_dim : int, hidden_dims :List = None, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim

        modules = []

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels = h_dim, kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i+1],
                                       kernel_size= 3,
                                       stride = 2,
                                       padding = 1,
                                       output_padding = 1),
                    nn.BatchNorm2d(hidden_dims[i+1],
                                   nn.LeakyReLU())
            )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size = 3,
                stride = 2,
                padding = 1,
                output_padding =1
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels = 3, kernel_size = 3, padding = 1),
            nn.Tanh()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim = 1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z:Tensor) -> Tensor:
        result = self.decode_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu:Tensor, logvar : Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return eps * std + mu #decoder에 들어갈 latent vector


    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]


    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1+log_var - mu**2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss':loss, 'Recons Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples:int, current_device:int):
        z = torch.rand(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x:Tensor, **kwargs):
        return self.forward(x)[0]


