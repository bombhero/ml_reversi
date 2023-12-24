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


def read_files_from_list(file_list):
    enhance_x = np.zeros([60*4*len(file_list), 4, 8, 8])
    enhance_y = np.zeros([60*4*len(file_list)])
    file_list = random.sample(file_list, len(file_list))
    count = 0
    total_row = 0
    total_start_ts = time.time()
    for file_path in file_list:
        count += 1
        if not os.path.exists(file_path):
            continue
        new_x, new_y = read_one_file(file_path)
        if new_x is None:
            continue

        current_len = new_y.shape[0]
        enhance_x[total_row:(total_row+current_len), :, :, :] = new_x
        enhance_y[total_row:(total_row+current_len)] = new_y
        total_row += current_len
        end_ts = time.time()
        print('Loading {}/{}: {}, spent {:.2f}'.
              format(count, len(file_list), file_path.split('/')[-1], end_ts-total_start_ts), end='\r')
    total_end_ts = time.time()
    print('\nLoaded {} lines. Spent {:.2f} seconds.'.format(total_row, (total_end_ts - total_start_ts)))
    return enhance_x[:total_row, :, :, :], enhance_y[:total_row]


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
        new_x, new_y = read_one_file(full_path)
        if new_x is None:
            continue

        current_len = new_y.shape[0]
        enhance_x[total_row:(total_row+current_len), :, :, :] = new_x
        enhance_y[total_row:(total_row+current_len)] = new_y
        total_row += current_len
        if count % 10 == 0:
            end_ts = time.time()
            print('Reading {}, spent {:.4f}'.format(count, end_ts-total_start_ts), end='\r')
    end_ts = time.time()
    print('Reading {}, spent {:.4f}'.format(count, end_ts - total_start_ts))
    total_end_ts = time.time()
    print('Loaded {} lines. Spent {:.2f} seconds.'.format(total_row, (total_end_ts - total_start_ts)))
    return enhance_x[:total_row, :, :, :], enhance_y[:total_row]


def merge_and_average(x, y):
    start_ts = time.time()
    result_x = np.zeros([x.shape[0], 4, 8, 8])
    result_y = np.zeros([y.shape[0]])
    total_row = 0
    result_dict = {}
    for row_idx in range(x.shape[0]):
        row_key = x[row_idx].tobytes()
        if row_key in result_dict.keys():
            result_dict[row_key]['row'].append(row_idx)
            result_dict[row_key]['y'].append(y[row_idx])
        else:
            result_dict[row_key] = {'row': [row_idx], 'y': [y[row_idx]]}
        if row_idx % 1000 == 0:
            end_ts = time.time()
            print('Analyzing {}, total={}, spent {:.3f}'.format(row_idx, len(result_dict.keys()),
                                                                (end_ts-start_ts)), end='\r')
    end_ts = time.time()
    print('Analyzing {}, total={}, spent {:.3f}'.format(x.shape[0], len(result_dict.keys()), (end_ts - start_ts)))
    start_ts = time.time()
    for row_key in list(result_dict.keys()):
        new_li = result_dict[row_key]['y']
        if len(new_li) % 2 == 0:
            left_value = new_li[int(len(new_li) / 2) - 1]
            right_value = new_li[int(len(new_li) / 2)]
            new_y = (left_value + right_value) / 2
        else:
            new_y = new_li[int(len(new_li) / 2)]
        result_x[total_row, :] = x[result_dict[row_key]['row'][0], :]
        result_y[total_row] = new_y
        total_row += 1
        if total_row % 1000 == 0:
            end_ts = time.time()
            print('Merged {}, spent {:.3f}'.format(total_row, (end_ts-start_ts)), end='\r')
    end_ts = time.time()
    print('Merged {}, spent {:.3f}'.format(total_row, (end_ts-start_ts)))
    return result_x[0:total_row, :], result_y[0:total_row]


def duplicate_example_checking(remove_dup=False, data_path=default_data_path):
    file_list = os.listdir(data_path)
    md5_dict = {}
    dup_list = []
    file_count = 0
    for filename in file_list:
        file_count += 1
        file_path = data_path + '/' + filename
        print('Reading {}/{} {}'.format(file_count, len(file_list), file_path), end='\r')
        with open(file_path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            if md5 in list(md5_dict.keys()):
                dup_list.append(file_path)
                md5_dict[md5].append(filename)
            else:
                md5_dict[md5] = [filename]
    print('')
    if remove_dup:
        remove_count = 0
        for file_path in dup_list:
            # print('Remove {}'.format(file_path))
            os.remove(file_path)
            remove_count += 1
        print('Remained {}, Removed duplicated {}'.format(len(md5_dict), remove_count))
    else:
        print('Remained {}, Duplicate {}'.format(len(md5_dict), len(dup_list)))


def remove_old_examples(remained, data_path=default_data_path):
    file_list = os.listdir(data_path)
    file_list.sort(reverse=True)
    if len(file_list) > remained:
        print('Removing {} old files...'.format(len(file_list) - remained))
        for filename in file_list[remained:]:
            full_path = data_path + '/' + filename
            os.remove(full_path)


class ReversiDataSet(Dataset):
    def __init__(self, file_count, examples_path=None):
        if examples_path is None:
            x, y = read_all_files(file_count)
        else:
            x, y = read_all_files(file_count, data_path=examples_path)
        x, y = merge_and_average(x, y)
        self.data_x = torch.from_numpy(np.float32(x))
        self.data_y = torch.from_numpy(np.float32(y))

    def __getitem__(self, item):
        return self.data_x[item, :], self.data_y[item]

    def __len__(self):
        return self.data_y.shape[0]


class ReversiDataSetS(Dataset):
    def __init__(self, file_list):
        x, y = read_files_from_list(file_list)
        x, y = merge_and_average(x, y)
        self.data_x = torch.from_numpy(np.float32(x))
        self.data_y = torch.from_numpy(np.float32(y))

    def __getitem__(self, item):
        return self.data_x[item, :], self.data_y[item]

    def __len__(self):
        return self.data_y.shape[0]


def get_reversi_dataset(train_file_count, examples_path):
    test_file_count = int(train_file_count / 10)
    example_list = os.listdir(examples_path)
    example_list = random.sample(example_list, len(example_list))
    if len(example_list) < (train_file_count + test_file_count):
        train_count = int(float(train_file_count) * len(example_list) / float(train_file_count + test_file_count))
        test_count = int(float(test_file_count) * len(example_list) / float(train_file_count + test_file_count))
    else:
        train_count = int(train_file_count)
        test_count = int(test_file_count)
    train_file_list = []
    test_file_list = []
    for idx in range(train_count):
        train_file_list.append(examples_path + '/' + example_list[idx])
    for idx in range(test_count):
        test_file_list.append(examples_path + '/' + example_list[train_count + idx])
    print('Prepare train dataset by {} files from {}'.format(train_count, examples_path))
    train_dataset = ReversiDataSetS(train_file_list)
    print('Prepare test dataset by {} files from {}'.format(test_count, examples_path))
    test_dataset = ReversiDataSetS(test_file_list)
    return train_dataset, test_dataset


if __name__ == '__main__':
    tst_train, tst_test = get_reversi_dataset(3000, 'e:/bomb/proj/ml_reversi_train/examples/deep_v_deep')
    print('{}, {}'.format(len(tst_train), len(tst_test)))
