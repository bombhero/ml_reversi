class HumanPlayer:
    def __init__(self, player_name, color, oppo_color):
        self.player_name = player_name
        self.color = color
        self.oppo_color = oppo_color
        self.position_count = 0
        self.position_list = []
        self.result = None

    def get_action(self, game_board, position_list):
        print('Term to {}({}): '.format(self.player_name, game_board.color_dict[self.color]), end='')
        step_str = input()
        li = step_str.split(' ')
        position = [int(li[idx]) for idx in range(2)]
        return position

