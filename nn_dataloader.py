import numpy as np
import os
import random
from comm_utils import example_path


def read_one_file(file_path):
    """
    Read one file.
    文件第一行:col0-63为0, col64为获胜的棋手(-1或者1), col65为-1
    其他行: col0-63为当前局面(-1, 0, 1), col64为当前棋手, col65为当前棋手的落子.
    :param file_path:
    :return:
    example_x: 结构为 N*3*8*8
        其中N为样本数, 每个样本为3层8*8的数据.
        第0层为当前棋手的落子,1为有子,0为无子.
        第1层为对方棋手的落子,1为有子,0为无子.
        第2层为空白区域的信息,1为空白,0为非空.
    step_x: 结构为 N*1 (0-63)
        为当前棋手在example_x局面下的落子. example_x局面不包括这颗落子
        step_x = row_idx*8+col_idx
    example_y: 结构为 N*1 (-1, 1)
        为当前棋手在该局的获胜情况, -1为输, 1为赢.
        注意:由于当前棋手是轮流下棋,所以一个文件中的样本为一半赢,一半输
    """
    example_x = None
    example_y = None
    org_x = np.loadtxt(fname=file_path, delimiter=',', dtype='int')
    winner_color = int(org_x[0][64])
    for row_id in range(1, org_x.shape[0]):
        new_row = org_x[row_id]
        current_player = new_row[64]
        oppo_player = current_player * -1
        layer0 = ((new_row[0:64] == current_player) + 0).reshape([8, 8])
        layer1 = ((new_row[0:64] == oppo_player) + 0).reshape([8, 8])
        layer2 = ((new_row[0:64] == 0) + 0).reshape([8, 8])
        new_x = np.array([[layer0, layer1, layer2]])
        if example_x is None:
            example_x = new_x
        else:
            example_x = np.concatenate((example_x, new_x), axis=0)

    step_x = org_x[1:, 65]
    example_y = org_x[1:, 64] * winner_color

    return example_x, step_x, example_y


def read_all_files(file_count):
    example_x = None
    step_x = None
    example_y = None
    file_list = os.listdir(example_path)
    idx_list = [x for x in range(len(file_list))]
    if file_count < len(idx_list):
        idx_list = random.sample(idx_list, file_count)
    for idx in idx_list:
        full_path = '{}/{}'.format(example_path, file_list[idx])
        print('Reading {}'.format(full_path))
        new_x, new_step, new_y = read_one_file(full_path)

        if example_x is None:
            example_x = new_x
        else:
            example_x = np.concatenate((example_x, new_x), axis=0)

        if step_x is None:
            step_x = new_step
        else:
            step_x = np.concatenate((step_x, new_step), axis=0)

        if example_y is None:
            example_y = new_y
        else:
            example_y = np.concatenate((example_y, new_y), axis=0)

    return example_x, step_x, example_y


if __name__ == '__main__':
    tst_file = '{}/reversi_202311031538331993.csv'.format(example_path)
    tst_x, tst_step, tst_y = read_all_files(10000)
    print(tst_x.shape[0])
