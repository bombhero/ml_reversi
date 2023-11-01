from board import GameBoard


class ReversiGame:
    def __init__(self):
        self.shape = [8, 8]
        self.color_list = [-1, 1]
        self.gb = GameBoard(self.shape)
        self.direct_list = [[-1, -1], [1, 1], [0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]]

    def prepare_game(self):
        self.gb.base_board[int(self.shape[0]/2-1)][int(self.shape[1]/2-1)] = self.color_list[0]
        self.gb.base_board[int(self.shape[0]/2-1)][int(self.shape[1]/2)] = self.color_list[1]
        self.gb.base_board[int(self.shape[0]/2)][int(self.shape[1]/2-1)] = self.color_list[1]
        self.gb.base_board[int(self.shape[0]/2)][int(self.shape[1]/2)] = self.color_list[0]

    def step(self, position, color):
        if color == self.color_list[0]:
            oppo_color = self.color_list[1]
        else:
            oppo_color = self.color_list[0]

        for direct in self.direct_list:
            checking_position = [position[i]+direct[i] for i in range(len(position))]
            while (0<=checking_position[0]<self.shape[0]) and (0<=checking_position[1]<self.shape[1]):
                pass


if __name__ == '__main__':
    rg = ReversiGame()
    rg.prepare_game()
    rg.gb.show_text()
