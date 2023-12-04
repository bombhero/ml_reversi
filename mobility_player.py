import copy
import random


class MobilityPlayer:
    def __init__(self, player_name, color, oppo_color, verbose=True):
        self.player_name = player_name
        self.color = color
        self.oppo_color = oppo_color
        self.position_count = 0
        self.position_list = []
        self.result = None
        self.verbose = verbose
        self.direct_list = [[-1, -1], [1, 1], [0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]]

    def get_action(self, game_board, position_list, deep_analysis=True):
        min_count = 64
        final_position = []
        for position in position_list:
            count = self.predict_result(game_board.base_board, position)
            if count < 0:
                continue
            if count < min_count:
                min_count = count
                final_position = [position]
            elif count == min_count:
                final_position.append(position)
        if len(final_position) == 1:
            select_id = 0
        else:
            select_id = random.randint(0, len(final_position)-1)
        if self.verbose:
            print('Term to {}({}): '.format(self.player_name, game_board.color_dict[self.color]), end='')
            print('{},{}'.format(len(final_position), final_position[select_id]))
        return final_position[select_id]

    @staticmethod
    def _is_in_board(position):
        in_board = [0 <= position[d] < 8 for d in range(2)]
        if sum(in_board) == 2:
            return True
        else:
            return False

    def _is_legal_position(self, base_board, position, color, oppo_color, need_full=False):
        if base_board[position[0]][position[1]] != 0:
            return False

        # Legal checking
        is_legal = [False for _ in range(len(self.direct_list))]
        for idx in range(len(self.direct_list)):
            for i in range(1, 8, 1):
                check_position = [position[d] + self.direct_list[idx][d] * i for d in range(2)]
                if not self._is_in_board(check_position):
                    break

                if (i == 1) and (base_board[check_position[0]][check_position[1]] != oppo_color):
                    break

                if base_board[check_position[0]][check_position[1]] == 0:
                    break

                if base_board[check_position[0]][check_position[1]] == color:
                    is_legal[idx] = True
                    break

            if is_legal[idx] and need_full:
                base_board[position[0]][position[1]] = color
                for i in range(1, 8, 1):
                    full_position = [position[d] + self.direct_list[idx][d] * i for d in range(2)]
                    if base_board[full_position[0]][full_position[1]] == oppo_color:
                        base_board[full_position[0]][full_position[1]] = color
                    else:
                        break

        return sum(is_legal) > 0

    def detect_position(self, base_board, color, oppo_color):
        position_list = []
        for row_idx in range(8):
            for col_idx in range(8):
                if base_board[row_idx][col_idx] != 0:
                    continue
                if self._is_legal_position(base_board, [row_idx, col_idx], color, oppo_color):
                    position_list.append([row_idx, col_idx])
        return position_list

    def predict_result(self, base_board, position):
        if base_board[position[0]][position[1]] != 0:
            return -1

        # Legal checking
        current_board = copy.copy(base_board)
        if self._is_legal_position(current_board, position, self.color, self.oppo_color, need_full=True):
            position_list = self.detect_position(current_board, self.oppo_color, self.color)
            return len(position_list)
        else:
            return -1

