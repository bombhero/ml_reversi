import copy
import random


class CalcPlayer:
    def __init__(self, player_name, color, oppo_color, verbose=True):
        self.player_name = player_name
        self.color = color
        self.oppo_color = oppo_color
        self.position_count = 0
        self.position_list = []
        self.result = None
        self.verbose = verbose

    def get_action(self, game_board, position_list, deep_analysis=True):
        max_count = 0
        final_position = []
        for position in position_list:
            count = self.predict_result(game_board.base_board, position)
            if count > max_count:
                max_count = count
                final_position = [position]
            elif count == max_count:
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

    def predict_result(self, base_board, position):
        direct_list = [[-1, -1], [1, 1], [0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]]
        if base_board[position[0]][position[1]] != 0:
            return 0

        # Legal checking
        is_legal = [False for _ in range(len(direct_list))]
        current_board = copy.copy(base_board)
        for idx in range(len(direct_list)):
            for i in range(1, 8, 1):
                check_position = [position[d] + direct_list[idx][d] * i for d in range(2)]
                if not self._is_in_board(check_position):
                    break

                if (i == 1) and (current_board[check_position[0]][check_position[1]] != self.oppo_color):
                    break

                if current_board[check_position[0]][check_position[1]] == 0:
                    break

                if current_board[check_position[0]][check_position[1]] == self.color:
                    is_legal[idx] = True
                    break

            if is_legal[idx]:
                current_board[position[0]][position[1]] = self.color
                for i in range(1, 8, 1):
                    full_position = [position[d] + direct_list[idx][d] * i for d in range(2)]
                    if current_board[full_position[0]][full_position[1]] == self.oppo_color:
                        current_board[full_position[0]][full_position[1]] = self.color
                    else:
                        break

        return sum(sum(current_board == self.color))

