import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):

        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']

        @torch.no_grad()
    def _update_target_network_parameters(self):
            """

            Momentum update of the key encoder
            """
        for param_q , param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1-self.m)

    @staticmethod #클래스 선언없이 사용가능
    def regression_loss(x, y):
        x = F.normalize(x, dim = 1)
        y = F.normalize(y, dim = 1)
        return 2 - 2*(x*y).sum(dim = -1)


    def initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size,
                                  num_workers = self.num_workers, drop_last = False, shuffle = True)
        niter = 0
        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):

            for (batch_view_1, batch_view_2), _ in train_loader:
                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                loss = self.update(batch_view_1,batch_view_2)
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                self.update_target_network_parameters()
                niter += 1

        self.save_models(os.path.join(model_checkpoints_folder, 'model.pth'))
    def update(self, batch_view_1, batch_view_2):
        predictions_from_view_1 = self.predictior(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictior(self.online_network(batch_view_2))

        with torch.no_grad():
            targets_to_view_1 = self.target_network(batch_view_1)
            targets_to_view_2 = self.target_network(batch_view_1)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):
        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'tagret_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)