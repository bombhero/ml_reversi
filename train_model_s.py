import datetime
import os
import torch
import time
from train_utils import TrainParam
from torch.utils.data import DataLoader
from nn_dataloader_rot import ReversiDataSet
from nn_player_s import ReversiCriticNetS


class NetTrain:
    def __init__(self, train_param):
        self.model_folder = train_param.models_path + train_param.model_sub_path
        self.model_backup_folder = train_param.models_path + train_param.model_backup_path
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        if not os.path.exists(self.model_backup_folder):
            os.makedirs(self.model_backup_folder)
        if torch.cuda.is_available():
            self.calc_device = torch.device('cuda')
        else:
            self.calc_device = torch.device('cpu')
        self.model_filename = train_param.model_filename
        self.model_file = self.model_folder + '/' + self.model_filename
        if os.path.exists(self.model_file):
            print('Reload {}'.format(self.model_file))
            self.model = torch.load(self.model_file).to(self.calc_device)
            self.reload = True
        else:
            self.model = ReversiCriticNetS().to(self.calc_device)
            self.reload = False
        self.backup_model = None
        self.example_file_count = train_param.round_count
        self.examples_full_path = train_param.examples_path + train_param.examples_sub_path
        self.model_label = train_param.model_label

    def train(self, epoch):
        loss_record = []
        if self.reload:
            start_save_round = 5
        else:
            start_save_round = 10
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_func = torch.nn.L1Loss(reduction='mean')
        dataset = ReversiDataSet(self.example_file_count, self.examples_full_path)
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
            if len(loss_record) > start_save_round:
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
        backup_path = '{}/{}_{}'.format(self.model_backup_folder, self.model_label, dt.strftime('%Y%m%d%H%M%S'))
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        backup_file = '{}/{}'.format(backup_path, self.model_filename)
        torch.save(self.model, self.model_file)
        torch.save(self.model, backup_file)
        print('Save {}, backup {}'.format(self.model_file, backup_file))


def train_model(train_param):
    net_train = NetTrain(train_param)
    net_train.train(200)


if __name__ == '__main__':
    tst_param = TrainParam()
    train_model(tst_param)
