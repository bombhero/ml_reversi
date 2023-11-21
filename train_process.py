import pandas as pd
from train_game import train_game_play
from train_utils import TrainParam
from nn_dataloader_rot import duplicate_example_checking
from nn_dataloader_rot import remove_old_examples
from train_model_s import train_model


def main():
    train_param = TrainParam()
    for idx in range(20):
        print('--------------------------------------------------- Train Round {}'.format(idx))
        train_game_play(train_param)
        duplicate_example_checking(data_path=(train_param.examples_path + train_param.examples_sub_path))
        remove_old_examples(int(train_param.round_count * 1.2),
                            data_path=(train_param.examples_path + train_param.examples_sub_path))
        train_model(train_param)


if __name__ == '__main__':
    main()
