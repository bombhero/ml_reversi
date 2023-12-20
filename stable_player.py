import copy
import random
import time


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


class StablePlayer:
    def __init__(self, player_name, color, oppo_color, verbose=True):
        self.player_name = player_name
        self.color = color
        self.oppo_color = oppo_color
        self.position_count = 0
        self.position_list = []
        self.result = None
        self.verbose = verbose

    @staticmethod
    def _is_in_board(position):
        in_board = [0 <= position[d] < 8 for d in range(2)]
        if sum(in_board) == 2:
            return True
        else:
            return False

    def calc_stable_rate(self, game_board, position, color):
        direct_group_list = [[[-1, -1], [1, 1]], [[0, -1], [0, 1]], [[-1, 0], [1, 0]], [[-1, 1], [1, -1]]]
        stable_rate = 0
        for direct_list in direct_group_list:
            out_band = False
            only_oppo = 0
            for direct in direct_list:
                d_step = 0
                while True:
                    d_step += 1
                    d_position = [position[idx] + direct[idx] * d_step for idx in range(2)]
                    if not self._is_in_board(d_position):
                        out_band = True
                        break
                    if game_board.base_board[d_position[0]][d_position[1]] == 0:
                        break
                    elif game_board.base_board[d_position[0]][d_position[1]] == color:
                        continue
                    else:
                        only_oppo += 1
                        break
                if out_band:
                    break
            if out_band:
                stable_rate += 1
                continue
            if only_oppo == 2:
                stable_rate += 1
        return stable_rate

    def predict_score(self, game_board, position, color):
        emulate_game = EmulateGame(game_board, [self.color, self.oppo_color])
        emulate_game.step(color, position)
        total_rate = 0
        for row_idx in range(8):
            for col_idx in range(8):
                if emulate_game.gb.base_board[row_idx][col_idx] == color:
                    total_rate += self.calc_stable_rate(emulate_game.gb, [row_idx, col_idx], color)
        return total_rate

    def _get_action(self, game_board, position_list, color, verbose=False):
        score_list = []
        max_score = -1
        max_list = []
        for position in position_list:
            score = self.predict_score(game_board, position, color)
            if score > max_score:
                max_score = score
                max_list = [len(score_list)]
            elif score == max_score:
                max_list.append(len(score_list))
            score_list.append(score)
        final_id = max_list[random.randint(0, len(max_list)-1)]
        if self.verbose and verbose:
            print('rate {} step {}'.format(score_list, position_list[final_id]))
        return position_list[final_id]

    def _switch_color(self, color):
        if color == self.color:
            return self.oppo_color
        else:
            return self.color

    def deep_analysis(self, game_board, position_list, color, timeout=6, verbose=True):
        start_ts = time.time()
        score_list = []
        for idx in range(len(position_list)):
            first_position = position_list[idx]
            emulate_game = EmulateGame(game_board, [self.color, self.oppo_color])
            emulate_game.step(color, first_position)
            no_option_count = 0
            current_color = self._switch_color(color)
            while True:
                p_list = emulate_game.detect_position(current_color)
                if len(p_list) == 0:
                    no_option_count += 1
                    if no_option_count >= 2:
                        break
                    else:
                        current_color = self._switch_color(current_color)
                        continue
                else:
                    no_option_count = 0
                position = self._get_action(emulate_game.gb, p_list, current_color, verbose=False)
                emulate_game.step(current_color, position)
                current_color = self._switch_color(current_color)
            score_list.append(sum(sum(emulate_game.gb.base_board == self.color)))
        max_score = -1
        max_idx_list = []
        for idx in range(len(score_list)):
            if score_list[idx] > max_score:
                max_idx_list = [idx]
                max_score = score_list[idx]
            elif score_list[idx] == max_score:
                max_idx_list.append(idx)
        final_idx = random.randint(0, len(max_idx_list)-1)
        if self.verbose:
            end_ts = time.time()
            print('score {}, step {}, spent {:.2f}'.format(score_list, position_list[max_idx_list[final_idx]],
                                                           (end_ts-start_ts)))
        return position_list[max_idx_list[final_idx]]

    def get_action(self, game_board, position_list, deep_analysis=False):
        if deep_analysis:
            return self.deep_analysis(game_board, position_list, self.color)
        else:
            return self._get_action(game_board, position_list, self.color, verbose=True)
