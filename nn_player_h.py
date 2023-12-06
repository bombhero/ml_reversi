import os
import copy
import random
import time
import numpy as np
import torch
import __main__


class ReversiCriticNetH(torch.nn.Module):
    def __init__(self):
        super(ReversiCriticNetH, self).__init__()
        self.conv_l1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Tanh(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.Tanh()
        )
        # 32*8*8
        self.conv_highway = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.Tanh(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.Tanh()
        )
        self.conv_l2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=0),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.Tanh(),
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
        for _ in range(2):
            tmp_x = self.conv_highway(x)
            x = tmp_x + x
        x = self.conv_l2(x)
        x = x.view(x.size(0), -1)
        x = self.linear_l1(x)
        x = self.linear_l2(x)
        x = self.output(x)
        return x


setattr(__main__, 'ReversiCriticNetH', ReversiCriticNetH)


class EmulateGame:
    def __init__(self, game_board, color_list):
        self.gb = copy.deepcopy(game_board)
        self.color_list = color_list
        self.direct_list = [[-1, -1], [1, 1], [0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]]
        self.dim = 2
        self.side_length = 8

    def _is_in_board(self, position):
        in_board = [0 <= position[d] < self.side_length for d in range(self.dim)]
        if sum(in_board) == self.dim:
            return True
        else:
            return False

    def _is_legal_position(self, position, color, need_full=False):
        if color == self.color_list[0]:
            oppo_color = self.color_list[1]
        else:
            oppo_color = self.color_list[0]

        if self.gb.base_board[position[0]][position[1]] != 0:
            return False

        # Legal checking
        is_legal = [False for _ in range(len(self.direct_list))]
        for idx in range(len(self.direct_list)):
            for i in range(1, 8, 1):
                check_position = [position[d] + self.direct_list[idx][d] * i for d in range(self.dim)]
                if not self._is_in_board(check_position):
                    break

                if (i == 1) and (self.gb.base_board[check_position[0]][check_position[1]] != oppo_color):
                    break

                if self.gb.base_board[check_position[0]][check_position[1]] == 0:
                    break

                if self.gb.base_board[check_position[0]][check_position[1]] == color:
                    is_legal[idx] = True
                    break

            if is_legal[idx] and need_full:
                self.gb.base_board[position[0]][position[1]] = color
                for i in range(1, 8, 1):
                    full_position = [position[d] + self.direct_list[idx][d] * i for d in range(self.dim)]
                    if self.gb.base_board[full_position[0]][full_position[1]] == oppo_color:
                        self.gb.base_board[full_position[0]][full_position[1]] = color
                    else:
                        break

        return sum(is_legal) > 0

    def detect_position(self, color):
        position_list = []
        for row_idx in range(self.side_length):
            for col_idx in range(self.side_length):
                if self.gb.base_board[row_idx][col_idx] != 0:
                    continue
                if self._is_legal_position([row_idx, col_idx], color):
                    position_list.append([row_idx, col_idx])
        return position_list

    def step(self, color, position):
        self._is_legal_position(position, color, need_full=True)


class AIPlayerH:
    def __init__(self, player_name, color, oppo_color, model_file_path, train_mode=False, verbose=False,
                 shadow=False, deep_all=True):
        self.player_name = player_name
        self.color = color
        self.oppo_color = oppo_color
        self.position_count = 0
        self.position_list = []
        self.result = None
        self.model_file = model_file_path
        if torch.cuda.is_available():
            self.calc_device = torch.device('cuda')
        else:
            self.calc_device = torch.device('cpu')
        if os.path.exists(self.model_file):
            if verbose:
                print('{} is loaded {}'.format(player_name, self.model_file))
            if self.calc_device == torch.device('cpu'):
                self.model = torch.load(self.model_file, map_location=torch.device('cpu')).to(self.calc_device)
            else:
                self.model = torch.load(self.model_file).to(self.calc_device)
        else:
            raise Exception('Cannot open {}'.format(self.model_file))
        if not shadow:
            self.emulate_oppo = AIPlayerH('Shadow', oppo_color, color, model_file_path=self.model_file, shadow=True,
                                          train_mode=train_mode, verbose=verbose)
        else:
            self.emulate_oppo = None
        self.verbose = verbose
        # If train_mode is true, need to provide some random step
        self.train_mode = train_mode
        self.deep_all = deep_all

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

    def deep_analysis(self, game_board, position_list, timeout=6):
        start_ts = time.time()
        score_list = []
        for idx in range(len(position_list)):
            first_position = position_list[idx]
            emulate_game = EmulateGame(game_board, [self.color, self.oppo_color])
            emulate_game.step(self.color, first_position)
            no_option_count = 0
            player_list = [self.emulate_oppo, self]
            current_id = 0
            while True:
                p_list = emulate_game.detect_position(player_list[current_id].color)
                if len(p_list) == 0:
                    no_option_count += 1
                    if no_option_count >= 2:
                        break
                    else:
                        current_id = (current_id + 1) % 2
                        continue
                position = player_list[current_id].get_action(emulate_game.gb, p_list, deep_analysis=False,
                                                              verbose=False)
                emulate_game.step(player_list[current_id].color, position)
                current_id = (current_id + 1) % 2
            score_list.append(sum(sum(emulate_game.gb.base_board == self.color)))
            spent_time = time.time() - start_ts
            if (timeout - spent_time) < (spent_time / (idx+1.0)):
                break
        return score_list

    @staticmethod
    def sort_position_by_score(position_list, score_list):
        sorted_score = sorted(score_list, reverse=True)
        sorted_position_list = []
        for idx in range(len(score_list)):
            for position_idx in range(len(score_list)):
                if sorted_score[idx] == score_list[position_idx]:
                    sorted_position_list.append(position_list[position_idx])
                    break
        return sorted_position_list

    def random_step(self, board, position_list):
        if random.random() > 0.7:
            r_idx = random.randint(0, len(position_list)-1)
            if self.verbose:
                print('Full random step {}'.format(position_list))
        else:
            pre_score = self.predict_score(board, position_list)
            position_list = self.sort_position_by_score(position_list, pre_score)
            if len(position_list) >= 4:
                length = int(len(position_list) / 2)
            else:
                length = len(position_list)
            r_idx = random.randint(0, length-1)
            if self.verbose:
                print('Half random step {}'.format(position_list[r_idx]))
        return position_list[r_idx]

    @staticmethod
    def check_highest_position(position_list):
        highest_position_list = [[0, 0], [0, 7], [7, 0], [7, 7]]
        for highest_position in highest_position_list:
            if highest_position in position_list:
                return highest_position
        return None

    def get_action(self, game_board, position_list, deep_analysis=True, verbose=True):
        start_ts = time.time()
        max_idx = 0
        rand_flag = False
        position = self.check_highest_position(position_list)
        if position is not None:
            if self.verbose and verbose:
                print('Highest step: {}'.format(position))
            return position
        if len(position_list) == 1:
            if self.verbose and verbose:
                print('Only Step: {}'.format(position_list[0]))
            return position_list[0]
        if self.train_mode:
            if sum(sum(game_board.base_board == 0)) > 2:
                if random.random() > 0.7:
                    rand_flag = True
        if sum(sum(game_board.base_board == 0)) > 59:
            rand_flag = True
        if rand_flag:
            return self.random_step(game_board.base_board, position_list)
        if deep_analysis:
            org_len = len(position_list)
            pre_scores = self.predict_score(game_board.base_board, position_list)
            position_list = self.sort_position_by_score(position_list, pre_scores)
            if self.deep_all:
                timeout = 1000
            else:
                timeout = 6
            score_list = self.deep_analysis(game_board, position_list, timeout=timeout)
            if self.verbose and verbose:
                print('{} --> {}'.format(org_len, len(score_list)), end='')
        else:
            score_list = self.predict_score(game_board.base_board, position_list)
        if self.verbose and verbose:
            print(score_list, end=' ')
        for idx in range(1, len(score_list)):
            if score_list[idx] >= score_list[max_idx]:
                max_idx = idx
        if self.verbose and verbose:
            end_ts = time.time()
            print('Step: {} Spent {:.2f}'.format(position_list[max_idx], (end_ts - start_ts)))
        return position_list[max_idx]


if __name__ == '__main__':
    tst_player = AIPlayerH('AIPlayer', 1, -1)
