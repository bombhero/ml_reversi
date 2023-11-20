# -- coding: utf-8 --
"""
训练数据添加棋盘旋转数据
"""
import numpy as np
import os
import shutil
import random
import torch
import hashlib
import time
from comm_utils import example_path
from torch.utils.data import Dataset

# data_path = example_path + '/random_v_random'
# data_path = example_path + '/ai_v_ai'
default_data_path = example_path + '/training_data'


def read_one_file(file_path):
    """
    Read one file.
    文件第一行:col0-63为0, col64为获胜的棋手(-1或者1), col65为-1
    其他行: col0-63为当前局面(-1, 0, 1), col64为当前棋手, col65为当前棋手的落子.
    :param file_path:
    :return:
    example_x: 结构为 N*4*8*8
        其中N为样本数, 每个样本为3层8*8的数据.
        第0层为当前棋手的落子,1为有子,0为无子.
        第1层为对方棋手的落子,1为有子,0为无子.
        第2层为空白区域的信息,1为空白,0为非空.
        第3层为当前棋手的落子,1为落子,0为不落子.因此该层只有一个1.
    example_y: 结构为 N*1 [-1, 1]
        为当前棋手在该局的获胜情况, 正数为赢, 数字越大说明最终差值越大, 负数为输, 0为平局.
        注意:由于当前棋手是轮流下棋,所以一个文件中的样本为一半赢,一半输
    """
    example_x = None
    example_y = None
    rot_list = [0, -1, 1, 2]
    org_x = np.loadtxt(fname=file_path, delimiter=',', dtype='int')
    winner_color = int(org_x[0][64])
    if winner_color == 0:
        return None, None, None
    score = (float(org_x[0][65]) / 64) * winner_color
    for row_id in range(1, org_x.shape[0]):
        new_row = org_x[row_id]
        current_player = int(new_row[64])
        oppo_player = current_player * -1
        layer0 = ((new_row[0:64] == current_player) + 0).reshape([8, 8])
        layer1 = ((new_row[0:64] == oppo_player) + 0).reshape([8, 8])
        layer2 = ((new_row[0:64] == 0) + 0).reshape([8, 8])
        step_row_id = int(int(new_row[65]) / 8)
        step_col_id = int(int(new_row[65]) % 8)
        layer3 = np.zeros([8, 8])
        layer3[step_row_id][step_col_id] = 1
        for rot in rot_list:
            new_x = np.array([[np.rot90(layer0, rot),
                               np.rot90(layer1, rot),
                               np.rot90(layer2, rot),
                               np.rot90(layer3, rot)]])
            if example_x is None:
                example_x = new_x
            else:
                example_x = np.concatenate((example_x, new_x), axis=0)

            if example_y is None:
                example_y = np.array([new_row[64] * score])
            else:
                example_y = np.concatenate((example_y, np.array([new_row[64] * score])), axis=0)
    return example_x, example_y


def read_all_files(file_count, data_path=default_data_path):
    enhance_x = np.zeros([60*4*file_count, 4, 8, 8])
    enhance_y = np.zeros([60*4*file_count])
    file_list = os.listdir(data_path)
    idx_list = [x for x in range(len(file_list))]
    if file_count < len(idx_list):
        idx_list = random.sample(idx_list, file_count)
    count = 0
    total_row = 0
    print('Loading data from {}'.format(data_path))
    total_start_ts = time.time()
    for idx in idx_list:
        full_path = '{}/{}'.format(data_path, file_list[idx])
        count += 1
        start_ts = time.time()
        print('Reading {}, '.format(count), end='')
        # print('.', end='')
        new_x, new_y = read_one_file(full_path)
        middle_ts = time.time()
        print('load {:.4f}, '.format(middle_ts-start_ts), end='')
        if new_x is None:
            # print(' Both winner. Skip it.')
            continue
        # print(' Got {}'.format(new_x.shape[0]))

        current_len = new_y.shape[0]
        enhance_x[total_row:(total_row+current_len), :, :, :] = new_x
        enhance_y[total_row:(total_row+current_len)] = new_y
        total_row += current_len
        end_ts = time.time()
        print('merge {:.4f}'.format(end_ts-middle_ts), end='\r')
    print('')
    total_end_ts = time.time()
    print('Loaded {} lines. Spent {:.2f} seconds.'.format(total_row, (total_end_ts - total_start_ts)))
    return enhance_x[:total_row, :, :, :], enhance_y[:total_row]


def duplicate_example_checking(remove_dup=False, data_path=default_data_path):
    file_list = os.listdir(data_path)
    md5_dict = {}
    dup_list = []
    for filename in file_list:
        file_path = data_path + '/' + filename
        print('Reading {}'.format(file_path), end='\r')
        with open(file_path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            if md5 in list(md5_dict.keys()):
                if remove_dup:
                    dup_list.append(file_path)
                else:
                    print('Duplicate {}'.format(filename))
                    md5_dict[md5].append(filename)
            else:
                md5_dict[md5] = [filename]
    print('')
    remove_count = 0
    for file_path in dup_list:
        print('Remove {}'.format(file_path))
        os.remove(file_path)
        remove_count += 1
    print('Remained {}, Removed {}'.format(len(md5_dict), remove_count))


def remove_old_examples(remained, data_path=default_data_path):
    file_list = os.listdir(data_path)
    file_list.sort(reverse=True)
    if len(file_list) > remained:
        for filename in file_list[remained:]:
            full_path = data_path + '/' + filename
            print('Remove {}'.format(full_path))
            os.remove(full_path)


class ReversiDataSet(Dataset):
    def __init__(self, file_count, examples_path=None):
        if examples_path is None:
            x, y = read_all_files(file_count)
        else:
            x, y = read_all_files(file_count, data_path=examples_path)
        self.data_x = torch.from_numpy(np.float32(x))
        self.data_y = torch.from_numpy(np.float32(y))

    def __getitem__(self, item):
        return self.data_x[item, :], self.data_y[item]

    def __len__(self):
        return self.data_y.shape[0]


if __name__ == '__main__':
    duplicate_example_checking(remove_dup=True)
