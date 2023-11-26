import random


class RandomPlayer:
    def __init__(self, player_name, color, oppo_color, verbose=True):
        self.player_name = player_name
        self.color = color
        self.oppo_color = oppo_color
        self.position_count = 0
        self.position_list = []
        self.result = None
        self.verbose = verbose

    def get_action(self, game_board, position_list, deep_analysis=True):
        idx = random.randint(0, len(position_list)-1)
        if self.verbose:
            print('Term to {}({}): '.format(self.player_name, game_board.color_dict[self.color]), end='')
            print('{}'.format(position_list[idx]))
        return position_list[idx]
