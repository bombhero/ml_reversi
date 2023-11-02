from board import GameBoard


class ReversiGame:
    def __init__(self):
        self.side_length = 8
        self.dim = 2
        self.color_list = [-1, 1]
        self.gb = GameBoard([self.side_length for _ in range(self.dim)])
        self.direct_list = [[-1, -1], [1, 1], [0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]]

    def prepare_game(self):
        self.gb.base_board[int(self.side_length/2-1)][int(self.side_length/2-1)] = self.color_list[0]
        self.gb.base_board[int(self.side_length/2-1)][int(self.side_length/2)] = self.color_list[1]
        self.gb.base_board[int(self.side_length/2)][int(self.side_length/2-1)] = self.color_list[1]
        self.gb.base_board[int(self.side_length/2)][int(self.side_length/2)] = self.color_list[0]

    def _is_in_board(self, position):
        in_board = [0 <= position[d] < self.side_length for d in range(self.dim)]
        if sum(in_board) == self.dim:
            return True
        else:
            return False

    def step(self, position, color, show=False):
        if not self._is_legal_position(position, color, need_full=True):
            return -1
        if show:
            self.gb.show_text()
            print('\r')

        return 0

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


if __name__ == '__main__':
    rg = ReversiGame()
    rg.prepare_game()
    rg.gb.show_text()
