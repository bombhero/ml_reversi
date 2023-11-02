import numpy as np


class GameBoard:
    def __init__(self, size):
        self.base_board = np.zeros(size)
        self.color_dict = {-1: 'x', 1: 'o', 0: '-'}

    def show_text(self):
        print('  ', end='')
        for col_idx in range(self.base_board.shape[0]):
            print('{} '.format(col_idx), end='')
        print('\r')
        for row_idx in range(self.base_board.shape[0]):
            print('{} '.format(row_idx), end='')
            for col_idx in range(self.base_board.shape[1]):
                if self.base_board[row_idx][col_idx] in self.color_dict.keys():
                    print('{} '.format(self.color_dict[self.base_board[row_idx][col_idx]]), end='')
                else:
                    print('- ', end='')
            print('\r')


if __name__ == '__main__':
    gb = GameBoard([8, 8])
    gb.base_board[4][4] = 1
    gb.base_board[3][4] = -1
    gb.show_text()
