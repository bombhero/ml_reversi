import datetime
import os
import torch
import time
from torch.utils.data import DataLoader
from comm_utils import model_path
from nn_dataloader_rot import ReversiDataSet
from nn_player_s import ReversiCriticNetS

model_label = 'players'
cnn_model_path = model_path + '/{}'.format(model_label)
backup_model_path = model_path + '/{}_back'.format(model_label)
critic_model_file = 'model_{}.pkl'.format(model_label)


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
            self.model = ReversiCriticNetS().to(self.calc_device)
        self.backup_model = None

    def train(self, epoch):
        loss_record = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_func = torch.nn.L1Loss(reduction='mean')
        dataset = ReversiDataSet(10000)
        dataloader = DataLoader(dataset=dataset, batch_size=int(len(dataset) / 100), shuffle=True, num_workers=4)
        for e in range(epoch):
            self.model.train()
            current_loss = 0
            start_ts = time.time()
            for i, data in enumerate(dataloader):
                middle_ts = time.time()
                print('Epoch {}: Spent {:.2f}, load {}.'.format(e, (middle_ts-start_ts), i), end='\r')
                tensor_x, tensor_y = data
                train_x = torch.autograd.Variable(tensor_x).to(self.calc_device)
                train_y = torch.autograd.Variable(tensor_y.view(tensor_y.size(0), -1)).to(self.calc_device)
                optimizer.zero_grad()
                output_y = self.model(train_x)
                loss = loss_func(output_y, train_y)
                current_loss += loss.data.cpu()
                loss.backward()
                optimizer.step()
            end_ts = time.time()
            print('Epoch {}: Spent {:.2f}sec, loss = {}'.format(e, (end_ts - start_ts), current_loss))
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
        dt = datetime.datetime.now()
        backup_path = '{}/{}'.format(backup_model_path, dt.strftime('%Y%m%d%H%M%S'))
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        backup_file = '{}/{}'.format(backup_path, critic_model_file)
        torch.save(self.model, self.model_file)
        torch.save(self.model, backup_file)
        print('Save {}, backup {}'.format(self.model_file, backup_file))


if __name__ == '__main__':
    for tst_i in range(3):
        print('-------------------------------------------------------------------Round {}'.format(tst_i))
        if tst_i == 0:
            net_train = NetTrain(reload=False)
        else:
            net_train = NetTrain(reload=True)
        net_train.train(200)
