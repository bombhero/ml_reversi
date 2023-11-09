import os
import random
import numpy as np
import torch
import __main__
from comm_utils import model_path

cnn_model_path = model_path + '/' + 'playerj'
critic_model_file = 'model_critic.pkl'


class ReversiCriticNetJ(torch.nn.Module):
    def __init__(self):
        super(ReversiCriticNetJ, self).__init__()
        self.conv_l1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5, padding=0),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.Tanh()
        )
        # 64*4*4
        self.conv_l2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4, padding=0),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.Tanh()
        )
        self.linear_l1 = torch.nn.Linear(in_features=256, out_features=128)
        self.linear_l2 = torch.nn.Linear(in_features=128, out_features=1)
        self.output = torch.nn.Tanh()

    def forward(self, x):
        """
        layer0: my side
        layer1: oppo side
        layer2: empty place
        layer3: the point which I want to put
        """
        x = self.conv_l1(x)
        x = self.conv_l2(x)
        x = x.view(x.size(0), -1)
        x = self.linear_l1(x)
        x = self.linear_l2(x)
        x = self.output(x)
        return x


setattr(__main__, 'ReversiCriticNetJ', ReversiCriticNetJ)


class AIPlayerJ:
    def __init__(self, player_name, color, oppo_color, model_file_path=None, shadow=False):
        self.player_name = player_name
        self.color = color
        self.oppo_color = oppo_color
        self.position_count = 0
        self.position_list = []
        self.result = None
        if model_file_path is None:
            self.model_file = cnn_model_path + '/' + critic_model_file
        else:
            self.model_file = model_file_path
        if torch.cuda.is_available():
            self.calc_device = torch.device('cuda')
        else:
            self.calc_device = torch.device('cpu')
        if os.path.exists(self.model_file):
            print('{} is loaded {}'.format(player_name, self.model_file))
            if self.calc_device == torch.device('cpu'):
                self.model = torch.load(self.model_file, map_location=torch.device('cpu')).to(self.calc_device)
            else:
                self.model = torch.load(self.model_file).to(self.calc_device)
        else:
            raise Exception('Cannot open {}'.format(self.model_file))
        if not shadow:
            self.emulate_oppo = AIPlayerJ('Shadow', oppo_color, color, model_file_path=self.model_file, shadow=True)
        else:
            self.emulate_oppo = None

    def transfer_board(self, board):
        """
        layer0: my side
        layer1: oppo side
        layer2: empty place
        """
        layer0 = (board == self.color) + 0
        layer1 = (board == self.oppo_color) + 0
        layer2 = (board == 0) + 0
        return np.array([[layer0, layer1, layer2]])

    def predict_score(self, board, position_list):
        """
        layer3: the point which I want to put
        """
        score_list = []
        current_board = self.transfer_board(board)
        for position in position_list:
            layer3 = np.zeros([1, 1, 8, 8])
            layer3[0][0][position[0]][position[1]] = 1.0
            data_x = np.concatenate((current_board, layer3), axis=1)
            tensor_x = torch.from_numpy(np.float32(data_x)).to(self.calc_device)
            data_y = self.model(tensor_x).cpu().detach().numpy()[0][0]
            score_list.append(data_y)
        return score_list

    def get_action(self, game_board, position_list):
        max_idx = 0
        if (random.random() > 0.99) or (sum(sum(game_board.base_board == 0)) > 58):
            r_idx = random.randint(0, (len(position_list)-1))
            print('Random step {}'.format(position_list[r_idx]))
            return position_list[r_idx]
        score_list = self.predict_score(game_board.base_board, position_list)
        print(score_list, end='')
        for idx in range(1, len(score_list)):
            if score_list[idx] >= score_list[max_idx]:
                max_idx = idx
        print(position_list[max_idx])
        return position_list[max_idx]


if __name__ == '__main__':
    tst_player = AIPlayerJ('AIPlayer', 1, -1)
