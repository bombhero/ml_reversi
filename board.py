import numpy as np


class GameBoard:
    def __init__(self, size):
        self.base_board = np.zeros(size)

    def show_text(self):
        for row_idx in range(self.base_board.shape[0]):
            for col_idx in range(self.base_board.shape[1]):
                if self.base_board[row_idx][col_idx] == -1:
                    print('x ', end='')
                elif self.base_board[row_idx][col_idx] == 1:
                    print('o ', end='')
                else:
                    print('- ', end='')
            print('\r')


if __name__ == '__main__':
    gb = GameBoard([8, 8])
    gb.base_board[4][4] = 1
    gb.base_board[3][4] = -1
    gb.show_text()
