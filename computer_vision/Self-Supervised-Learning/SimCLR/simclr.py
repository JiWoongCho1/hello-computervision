import torch
from utils import accuracy, save_checkpoint, save_config_file
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm

import os


## 출처 : https://github.com/iamchosenlee/SimCLR-1/blob/24fe33dc547928f2be0d4a294b77caf246ac1eb8/simclr.py#L26
class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)])
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim = 1) #L2 norm , feature : 128

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype = torch.bool).to(self.args.devcie)
        labels = labels[~mask].view(labels.shape[0], -1) #except diagonal
        similarity_matrix = similarity_matrix[~mask].view(labels.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim = 1)

        labels = torch.zeros(logits.shape[0], dtype = torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels


    def train(self, train_loader):

        scaler = GradScaler(enabled = self.args.fp16_precision)
        n_iter = 0
        logging.info(f'start SimCLR training for {self.args.epochs} epochs')

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim = 0)
                images = images.to(self.args.device)

                with autocast(enabled = self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk =(1,5))
                    self.writer.add_scaler('loss', loss, global_step = n_iter)
                    self.writer.add_scaler('acc / top1', top1[0], global_step = n_iter)
                    self.writer.add_scaler('acc / top5', top5[0], global_step=n_iter)
                    self.writer.add_scaler('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                n_iter += 1

            if epoch_counter >= 10:
                self.scheduler.step()
        logging.info("Training has finished")

        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)

        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict()

        }, is_best = False, filename = os.path.join(self.writer.log_dir, checkpoint_name))



