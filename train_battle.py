"""
对一个目录中所有的NN进行两两对弈10局.
统计数据: 针对每个模型统计其 胜/平/负/净胜子
"""
import os
import datetime
import time
import pandas as pd
from train_game import ExecuteReversi
from nn_player_s import AIPlayerS
from train_utils import train_root_path


class BattleParam:
    def __init__(self):
        self.battle_path = None
        self.nn_path = None
        self.nn_label = None
        self.model_filename = 'players_model.pkl'
        self.color_list = [-1, 1]


def get_model_list(nn_path, label):
    dir_list = os.listdir(nn_path)
    out_list = []
    for dir_name in dir_list:
        if dir_name.find(label) == 0:
            out_list.append(dir_name)
    return out_list


def trans_result(result):
    name_list = list(result.keys())
    new_line = [{'name': name_list[0]}, {'name': name_list[1]}]
    if result[name_list[0]] > result[name_list[1]]:
        new_line[0]['win'] = 1
        new_line[1]['win'] = 0
        new_line[0]['draw'] = 0
        new_line[1]['draw'] = 0
        new_line[0]['loss'] = 0
        new_line[1]['loss'] = 1
    elif result[name_list[0]] < result[name_list[1]]:
        new_line[0]['win'] = 0
        new_line[1]['win'] = 1
        new_line[0]['draw'] = 0
        new_line[1]['draw'] = 0
        new_line[0]['loss'] = 1
        new_line[1]['loss'] = 0
    else:
        new_line[0]['win'] = 0
        new_line[1]['win'] = 0
        new_line[0]['draw'] = 1
        new_line[1]['draw'] = 1
        new_line[0]['loss'] = 0
        new_line[1]['loss'] = 0
    new_line[0]['score'] = result[name_list[0]] - result[name_list[1]]
    new_line[1]['score'] = result[name_list[1]] - result[name_list[0]]
    return new_line


def execute_battle(battle_param):
    df = pd.DataFrame(columns=('name', 'win', 'draw', 'loss', 'score'))
    model_list = get_model_list(battle_param.nn_path, battle_param.nn_label)
    for current_idx in range(len(model_list)-1):
        for oppo_idx in range(current_idx+1, len(model_list)):
            model_file_0 = battle_param.nn_path + '/' + model_list[current_idx] + '/' + battle_param.model_filename
            model_file_1 = battle_param.nn_path + '/' + model_list[oppo_idx] + '/' + battle_param.model_filename
            player_list = [AIPlayerS(model_list[current_idx], battle_param.color_list[0], battle_param.color_list[1],
                                     model_file_path=model_file_0, train_mode=False, verbose=False),
                           AIPlayerS(model_list[oppo_idx], battle_param.color_list[1], battle_param.color_list[0],
                                     model_file_path=model_file_1, train_mode=False, verbose=False)]
            start_ts = time.time()
            for i in range(10):
                if i % 2 == 1:
                    reverse = True
                else:
                    reverse = False
                exe_reversi = ExecuteReversi(reverse, game_path=battle_param.game_record_path, player_list=player_list)
                exe_reversi.execute()
                result = {exe_reversi.player_list[0].player_name: exe_reversi.player_list[0].result,
                          exe_reversi.player_list[1].player_name: exe_reversi.player_list[1].result}
                new_line = trans_result(result)
                for j in range(len(new_line)):
                    if new_line[j]['name'] in list(df['name']):
                        df.loc[df['name'] == new_line[j]['name'], 'win'] += new_line[j]['win']
                        df.loc[df['name'] == new_line[j]['name'], 'draw'] += new_line[j]['draw']
                        df.loc[df['name'] == new_line[j]['name'], 'loss'] += new_line[j]['loss']
                        df.loc[df['name'] == new_line[j]['name'], 'score'] += new_line[j]['score']
                    else:
                        tmp_df = pd.DataFrame(new_line[j], index=[0])
                        df = pd.concat((df, tmp_df), ignore_index=True)
                end_ts = time.time()
                print('{:0>3d}:{:0>3d} {} vs {} Round {} Spent {:.2f}'.
                      format(current_idx, oppo_idx, model_list[current_idx], model_list[oppo_idx], i,
                             (end_ts - start_ts)),
                      end='\r')
    print('')
    dt = datetime.datetime.now()
    result_file_path = '{}/battle_result_{}.csv'.format(battle_param.battle_path, dt.strftime('%Y%m%d%H%M%S'))
    df.to_csv(path_or_buf=result_file_path, sep=',', float_format='%.0f')
    return result_file_path


def run_battle(root_path=None):
    if root_path is None:
        root_path = train_root_path
    battle_param = BattleParam()
    battle_param.nn_path = root_path + '/models/players_backup'
    battle_param.nn_label = 'players'
    battle_param.battle_path = root_path + '/battle_record'
    battle_param.game_record_path = root_path + '/examples/battle_data'
    if not os.path.exists(battle_param.battle_path):
        os.makedirs(battle_param.battle_path)
    return execute_battle(battle_param)


if __name__ == '__main__':
    run_battle()
