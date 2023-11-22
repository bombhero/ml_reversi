import os
import datetime
import numpy as np
import pandas as pd
import random
import time
from reversi import ReversiGame
from human_player import HumanPlayer
from random_player import RandomPlayer
from nn_player import AIPlayer
from nn_player_j import AIPlayerJ
from nn_player_s import AIPlayerS
from calc_player import CalcPlayer
from train_utils import TrainParam


class ExecuteReversi:
    def __init__(self, reverse=False, game_path=None, player_list=None):
        self.reversi_game = ReversiGame()
        self.player_list = player_list
        if reverse:
            self.player_list = [self.player_list[1], self.player_list[0]]
        # col 0-63: board situation
        # col 64: player color
        # col 65: position id (row_id*8+col_idx)
        self.game_record = None
        self.show = False
        self.game_path = game_path

    def execute(self):
        skip_count = 0
        self.reversi_game.prepare_game()
        if self.show:
            self.reversi_game.gb.show_text()
        for term_id in range(int(self.reversi_game.gb.base_board.shape[0] * self.reversi_game.gb.base_board.shape[1])
                             * 2):
            side_id = term_id % 2
            position_list = self.reversi_game.detect_position(self.player_list[side_id].color)
            if self.show:
                print('Term {}: '.format(term_id), end='')
            if len(position_list) == 0:
                skip_count += 1
                if self.show:
                    print('{} has no position.'.format(self.player_list[side_id].player_name))
                if skip_count > 1:
                    break
                continue
            else:
                if self.show:
                    for position in position_list:
                        print('{} '.format(position), end='')
                skip_count = 0
            new_row = self.reversi_game.gb.base_board.reshape([1, int(self.reversi_game.side_length ** 2)])
            new_row = np.concatenate((new_row, np.array([[self.player_list[side_id].color]])), axis=1)
            while True:
                position = self.player_list[side_id].get_action(self.reversi_game.gb, position_list)
                ret = self.reversi_game.step(position, self.player_list[side_id].color, show=self.show)
                if ret == 0:
                    position_id = position[0] * self.reversi_game.side_length + position[1]
                    new_row = np.concatenate((new_row, np.array([[position_id]])), axis=1)
                    if self.game_record is None:
                        self.game_record = new_row
                    else:
                        self.game_record = np.concatenate((self.game_record, new_row), axis=0)
                    break

        winner_color, score = self.judge_winner()
        new_row = np.zeros([1, int(self.reversi_game.side_length ** 2)], dtype='int')
        new_row = np.concatenate((new_row, np.array([[winner_color, score]])), axis=1)
        self.game_record = np.concatenate((new_row, self.game_record), axis=0)
        if winner_color != 0:
            self.save_record()

    def judge_winner(self):
        for player_id in range(len(self.player_list)):
            self.player_list[player_id].result = 0
            for row_idx in range(self.reversi_game.side_length):
                for col_idx in range(self.reversi_game.side_length):
                    if self.reversi_game.gb.base_board[row_idx][col_idx] == self.player_list[player_id].color:
                        self.player_list[player_id].result += 1
        if self.show:
            print('{}: {}'.format(self.player_list[0].player_name, self.player_list[0].result))
            print('{}: {}'.format(self.player_list[1].player_name, self.player_list[1].result))
        if self.player_list[0].result > self.player_list[1].result:
            return self.player_list[0].color, (self.player_list[0].result - self.player_list[1].result)
        elif self.player_list[0].result < self.player_list[1].result:
            return self.player_list[1].color, (self.player_list[1].result - self.player_list[0].result)
        else:
            return 0, 0

    def save_record(self):
        dt = datetime.datetime.now()
        ms = random.randint(1, 9999)
        if not os.path.exists(self.game_path):
            os.makedirs(self.game_path)
        file_path = '{}/reversi_{}{:>04}.csv'.format(self.game_path, dt.strftime('%Y%m%d%H%M%S'), ms)
        np.savetxt(file_path, self.game_record, delimiter=',', fmt='%.0f')


def train_game_play(train_param):
    df = None
    game_summary = {}
    if not os.path.exists(train_param.models_path + train_param.model_sub_path):
        os.makedirs(train_param.models_path + train_param.model_sub_path)
    if not os.path.exists(train_param.models_path + train_param.model_backup_path):
        os.makedirs(train_param.models_path + train_param.model_backup_path)
    model_file = train_param.models_path + train_param.model_sub_path + '/' + train_param.model_filename
    game_path = train_param.examples_path + train_param.examples_sub_path
    if not os.path.exists(model_file):
        print('Random vs Random')
        player_list = [RandomPlayer('Random', train_param.color_list[0], train_param.color_list[1], verbose=False),
                       RandomPlayer('Dice', train_param.color_list[1], train_param.color_list[0], verbose=False)]
        round_count = int(train_param.round_count * 1.2)
    else:
        r_value = random.randint(0, 1)
        if r_value == 0:
            if random.random() > 0.5:
                print('Calc vs AI')
                player_list = [CalcPlayer('Calc', train_param.color_list[0], train_param.color_list[1], verbose=False),
                               AIPlayerS('NNTrain', train_param.color_list[1], train_param.color_list[0],
                                         model_file_path=model_file, train_mode=True, verbose=False)]
            else:
                print('Random vs AI')
                player_list = [RandomPlayer('Random', train_param.color_list[0], train_param.color_list[1],
                                            verbose=False),
                               AIPlayerS('NNTrain', train_param.color_list[1], train_param.color_list[0],
                                         model_file_path=model_file, train_mode=True, verbose=False)]
        else:
            print('AI vs AI')
            player_list = [AIPlayerS('NNTrainer', train_param.color_list[0], train_param.color_list[1],
                                     model_file_path=model_file, train_mode=True, verbose=False),
                           AIPlayerS('NNTeacher', train_param.color_list[1], train_param.color_list[0],
                                     model_file_path=model_file, train_mode=True, verbose=False)]
        round_count = int(train_param.round_count * 0.1)
    start_ts = time.time()
    for i in range(round_count):
        if i % 2 == 1:
            reverse = True
        else:
            reverse = False
        exe_reversi = ExecuteReversi(reverse, game_path=game_path, player_list=player_list)
        exe_reversi.execute()
        result = {exe_reversi.player_list[0].player_name: exe_reversi.player_list[0].result,
                  exe_reversi.player_list[1].player_name: exe_reversi.player_list[1].result}
        if exe_reversi.player_list[0].result > exe_reversi.player_list[1].result:
            result['winner'] = exe_reversi.player_list[0].player_name
        elif exe_reversi.player_list[0].result < exe_reversi.player_list[1].result:
            result['winner'] = exe_reversi.player_list[1].player_name
        else:
            result['winner'] = 'Both'
        if result['winner'] not in list(game_summary.keys()):
            game_summary[result['winner']] = 1
        else:
            game_summary[result['winner']] += 1
        if df is None:
            df = pd.DataFrame(result, index=[0])
        else:
            df = pd.concat((df, pd.DataFrame(result, index=[0])), ignore_index=True)
        end_ts = time.time()
        print('Playing round {}, spent {:.2f}'.format(i, (end_ts-start_ts)), end='\r')
    print('\nSummary {}'.format(game_summary))
    dt = datetime.datetime.now()
    if not os.path.exists(train_param.record_path):
        os.makedirs(train_param.record_path)
    result_path = '{}/result_{}.csv'.format(train_param.record_path, dt.strftime('%Y%m%d%H%M%S'))
    df.to_csv(path_or_buf=result_path, index=False, sep=',')


if __name__ == '__main__':
    tst_param = TrainParam()
    train_game_play(tst_param)
