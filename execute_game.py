import os
import datetime
import numpy as np
import pandas as pd
import random
import torch
from reversi import ReversiGame
from human_player import HumanPlayer
from random_player import RandomPlayer
from nn_player import AIPlayer
from nn_player_j import AIPlayerJ
from comm_utils import example_path
from comm_utils import records_path
from nn_player import ReversiCriticNet
from nn_player_j import ReversiCriticNetJ

# game_path = example_path + '/random_v_random'
# game_path = example_path + '/random_v_ai'
game_path = example_path + '/ai_v_ai'
backup_model = './models/cnn_backup/model_critic.pkl'
jay_model = './models/playerj_backup/model_critic.pkl'


class ExecuteReversi:
    def __init__(self):
        self.reversi_game = ReversiGame()
        self.player_list = []
        # self.player_list.append(AIPlayer('Backup', list(self.reversi_game.gb.color_dict.keys())[1],
        #                                  list(self.reversi_game.gb.color_dict.keys())[0],
        #                                  model_file_path=backup_model))
        # self.player_list.append(AIPlayerJ('NNJay', list(self.reversi_game.gb.color_dict.keys())[0],
        #                                   list(self.reversi_game.gb.color_dict.keys())[1],
        #                                   model_file_path=jay_model))
        # self.player_list.append(RandomPlayer('Dice', list(self.reversi_game.gb.color_dict.keys())[0],
        #                                      list(self.reversi_game.gb.color_dict.keys())[1]))
        # self.player_list.append(RandomPlayer('Random', list(self.reversi_game.gb.color_dict.keys())[1],
        #                                      list(self.reversi_game.gb.color_dict.keys())[0]))
        self.player_list.append(AIPlayer('Backup', list(self.reversi_game.gb.color_dict.keys())[1],
                                         list(self.reversi_game.gb.color_dict.keys())[0],
                                         model_file_path=backup_model))
        self.player_list.append(HumanPlayer('Bomb', list(self.reversi_game.gb.color_dict.keys())[0],
                                            list(self.reversi_game.gb.color_dict.keys())[1]))
        # col 0-63: board situation
        # col 64: player color
        # col 65: position id (row_id*8+col_idx)
        self.game_record = None
        self.show = True

    def execute(self):
        skip_count = 0
        self.reversi_game.prepare_game()
        if self.show:
            self.reversi_game.gb.show_text()
        for term_id in range(int(self.reversi_game.gb.base_board.shape[0] * self.reversi_game.gb.base_board.shape[1])
                             * 2):
            side_id = term_id % 2
            position_list = self.reversi_game.detect_position(self.player_list[side_id].color)
            print('Term {}: '.format(term_id), end='')
            if len(position_list) == 0:
                skip_count += 1
                print('{} has no position.'.format(self.player_list[side_id].player_name))
                if skip_count > 1:
                    break
                continue
            else:
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
        print('{}: {}'.format(self.player_list[0].player_name, self.player_list[0].result))
        print('{}: {}'.format(self.player_list[1].player_name, self.player_list[1].result))
        if self.player_list[0].result > self.player_list[1].result:
            print('{} win, color = {}'.format(self.player_list[0].player_name, self.player_list[0].color))
            return self.player_list[0].color, (self.player_list[0].result - self.player_list[1].result)
        elif self.player_list[0].result < self.player_list[1].result:
            print('{} win, color = {}'.format(self.player_list[1].player_name, self.player_list[1].color))
            return self.player_list[1].color, (self.player_list[1].result - self.player_list[0].result)
        else:
            print('Both win.')
            return 0, 0

    def save_record(self):
        dt = datetime.datetime.now()
        ms = random.randint(1, 9999)
        if not os.path.exists(game_path):
            os.makedirs(game_path)
        file_path = '{}/reversi_{}{:>04}.csv'.format(game_path, dt.strftime('%Y%m%d%H%M%S'), ms)
        np.savetxt(file_path, self.game_record, delimiter=',', fmt='%.0f')


def main():
    df = None
    for _ in range(3000):
        exe_reversi = ExecuteReversi()
        exe_reversi.execute()
        result = {exe_reversi.player_list[0].player_name: exe_reversi.player_list[0].result,
                  exe_reversi.player_list[1].player_name: exe_reversi.player_list[1].result}
        if exe_reversi.player_list[0].result > exe_reversi.player_list[1].result:
            result['winner'] = exe_reversi.player_list[0].player_name
        elif exe_reversi.player_list[0].result < exe_reversi.player_list[1].result:
            result['winner'] = exe_reversi.player_list[1].player_name
        else:
            result['winner'] = 'Both'
        if df is None:
            df = pd.DataFrame(result, index=[0])
        else:
            df = pd.concat((df, pd.DataFrame(result, index=[0])), ignore_index=True)
    dt = datetime.datetime.now()
    if not os.path.exists(records_path):
        os.makedirs(records_path)
    result_path = '{}/result_{}.csv'.format(records_path, dt.strftime('%Y%m%d%H%M%S'))
    df.to_csv(path_or_buf=result_path, index=False, sep=',')


if __name__ == '__main__':
    main()
