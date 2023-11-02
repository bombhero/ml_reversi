from reversi import ReversiGame
from human_player import HumanPlayer
from random_player import RandomPlayer


class ExecuteReversi:
    def __init__(self):
        self.reversi_game = ReversiGame()
        self.player_list = []
        self.player_list.append(RandomPlayer('bomb', list(self.reversi_game.gb.color_dict.keys())[0],
                                            list(self.reversi_game.gb.color_dict.keys())[1]))
        self.player_list.append(RandomPlayer('random', list(self.reversi_game.gb.color_dict.keys())[1],
                                list(self.reversi_game.gb.color_dict.keys())[0]))
        self.game_record = None

    def execute(self):
        skip_count = 0
        self.reversi_game.prepare_game()
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
            while True:
                position = self.player_list[side_id].get_action(self.reversi_game.gb, position_list)
                ret = self.reversi_game.step(position, self.player_list[side_id].color, show=True)
                if ret == 0:
                    break

        self.judge_winner()

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
            print('{} win'.format(self.player_list[0].player_name))
            return self.player_list[0].color
        else:
            print('{} win'.format(self.player_list[1].player_name))
            return self.player_list[1].color




if __name__ == '__main__':
    exe_reversi = ExecuteReversi()
    exe_reversi.execute()