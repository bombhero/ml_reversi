import numpy as np
import os
import torch
import copy
from torch.utils.data import DataLoader
from comm_utils import model_path
from nn_dataloader import ReversiDataSet
from nn_player import ReversiCriticNet

cnn_model_path = model_path + '/' + 'cnn'
critic_model_file = 'model_critic.pkl'


class NetTrain:
    def __init__(self, reload=False):
        if not os.path.exists(cnn_model_path):
            os.makedirs(cnn_model_path)
        if torch.cuda.is_available():
            self.calc_device = torch.device('cuda')
        else:
            self.calc_device = torch.device('cpu')
        self.model_file = cnn_model_path + '/' + critic_model_file
        if reload and os.path.exists(self.model_file):
            print('Reload {}'.format(self.model_file))
            self.model = torch.load(self.model_file)
        else:
            self.model = ReversiCriticNet().to(self.calc_device)
        self.backup_model = None

    def train(self, epoch):
        loss_record = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_func = torch.nn.L1Loss(reduction='mean')
        dataset = ReversiDataSet(10000)
        dataloader = DataLoader(dataset=dataset, batch_size=4098, shuffle=True)
        for e in range(epoch):
            self.model.train()
            current_loss = 0
            for i, data in enumerate(dataloader):
                print('Epoch {}: load {}.'.format(e, i), end='\r')
                tensor_x, tensor_y = data
                train_x = torch.autograd.Variable(tensor_x).to(self.calc_device)
                train_y = torch.autograd.Variable(tensor_y.view(tensor_y.size(0), -1)).to(self.calc_device)
                optimizer.zero_grad()
                output_y = self.model(train_x)
                loss = loss_func(output_y, train_y)
                current_loss += loss.data.cpu()
                loss.backward()
                optimizer.step()
            print('Epoch {}: loss = {}'.format(e, current_loss))
            loss_record.append(current_loss)
            if len(loss_record) > 10:
                del loss_record[0]
                if current_loss > (sum(loss_record) / len(loss_record)):
                    print('Current loss({}) is higher than ave loss({})'.format(current_loss,
                                                                                (sum(loss_record) / len(loss_record))))
                    break
                else:
                    self.model.eval()
                    self.save_model()

    def save_model(self):
        torch.save(self.model, self.model_file)
        print('Save model to {}'.format(self.model_file))


if __name__ == '__main__':
    for tst_i in range(2):
        if tst_i == 0:
            net_train = NetTrain(reload=False)
        else:
            net_train = NetTrain(reload=True)
        print('-------------------------------------------------------------------Round {}'.format(tst_i))
        net_train.train(200)
