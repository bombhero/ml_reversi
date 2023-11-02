import random


class RandomPlayer:
    def __init__(self, player_name, color, oppo_color):
        self.player_name = player_name
        self.color = color
        self.oppo_color = oppo_color
        self.position_count = 0
        self.position_list = []
        self.result = None

    def get_action(self, game_board, position_list):
        print('Term to {}({}): '.format(self.player_name, game_board.color_dict[self.color]), end='')
        idx = random.randint(0, len(position_list)-1)
        print('{}'.format(position_list[idx]))
        return position_list[idx]
