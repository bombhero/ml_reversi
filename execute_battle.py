"""
对一个目录中所有的NN进行两两对弈10局.
统计数据: 针对每个模型统计其 胜/平/负/净胜子
"""
import os
import datetime
import time
import pandas as pd
from train_game import ExecuteReversi
from nn_player_ss import AIPlayerSS
from nn_player_r import AIPlayerR
from train_utils import train_root_path

default_models_root_path = 'c:/bomb/proj/ml_reversi_train'
default_models_group_path = [default_models_root_path + '/model_20231123',
                             default_models_root_path + '/model_20231207',
                             default_models_root_path + '/model_20231209',
                             default_models_root_path + '/model_20231211']
default_label_list = ['playerR', 'players', 'playerss']


class BattleParam:
    def __init__(self):
        self.battle_path = None
        self.models_group_path = None
        self.nn_label_list = None
        self.game_record_path = None
        self.model_filename = None
        self.color_list = [-1, 1]


def get_model_list(models_group_path, label_list):
    model_dict_list = []
    for group_path in models_group_path:
        if not os.path.isdir(group_path):
            continue
        folder_list = os.listdir(group_path)
        for folder_name in folder_list:
            model_type = folder_name.split('_')[0]
            if model_type not in label_list:
                continue
            model_parent_path = group_path + '/' + folder_name
            file_list = os.listdir(model_parent_path)
            model_full_path = ''
            for filename in file_list:
                if filename.split('.')[-1] == 'pkl':
                    model_full_path = model_parent_path + '/' + filename
                    break
            if len(model_full_path) == 0:
                continue
            model_dict_list.append({'type': model_type, 'full_path': model_full_path, 'name': folder_name})
    return model_dict_list


def create_player(model_dict, color, oppo_color):
    if model_dict['type'] in ['players', 'playerss']:
        return AIPlayerSS(player_name=model_dict['name'], color=color, oppo_color=oppo_color,
                          model_file_path=model_dict['full_path'], train_mode=False, deep_all=False, verbose=False)
    elif model_dict['type'] in ['playerR']:
        return AIPlayerR(player_name=model_dict['name'], color=color, oppo_color=oppo_color,
                         model_file_path=model_dict['full_path'], train_mode=False, deep_all=False, verbose=False)


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
    per_round = 6
    df = pd.DataFrame(columns=('name', 'win', 'draw', 'loss', 'score'))
    model_dict_list = get_model_list(battle_param.models_group_path, battle_param.nn_label_list)
    total_round = len(model_dict_list) * (len(model_dict_list) - 1) * per_round
    print('Total {} round, might spend {} sec.'.format(total_round, 120*total_round))
    for current_idx in range(len(model_dict_list)-1):
        for oppo_idx in range(current_idx+1, len(model_dict_list)):
            player_list = [create_player(model_dict_list[current_idx], battle_param.color_list[0],
                                         battle_param.color_list[1]),
                           create_player(model_dict_list[oppo_idx], battle_param.color_list[1],
                                         battle_param.color_list[0])
                           ]
            start_ts = time.time()
            for i in range(per_round):
                if i % 2 == 1:
                    reverse = True
                else:
                    reverse = False
                exe_reversi = ExecuteReversi(reverse, game_path=battle_param.game_record_path, player_list=player_list)
                exe_reversi.execute(deep_analysis=True)
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
                      format(current_idx, oppo_idx, model_dict_list[current_idx]['name'],
                             model_dict_list[oppo_idx]['name'], i, (end_ts - start_ts)), end='\r')
    print('')
    dt = datetime.datetime.now()
    result_file_path = '{}/battle_result_{}.csv'.format(battle_param.battle_path, dt.strftime('%Y%m%d%H%M%S'))
    df.to_csv(path_or_buf=result_file_path, sep=',', float_format='%.0f')
    return result_file_path


def run_battle(root_path=None, models_group_path=None, label_list=None):
    if root_path is None:
        root_path = train_root_path
    if models_group_path is None:
        models_group_path = default_models_group_path
    if label_list is None:
        label_list = default_label_list
    battle_param = BattleParam()
    battle_param.nn_label_list = label_list
    battle_param.models_group_path = models_group_path
    battle_param.battle_path = root_path + '/battle_record'
    battle_param.game_record_path = root_path + '/examples/battle_data'
    if not os.path.exists(battle_param.battle_path):
        os.makedirs(battle_param.battle_path)
    return execute_battle(battle_param)


if __name__ == '__main__':
    run_battle()
