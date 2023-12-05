import shutil
from train_game import train_game_play
from train_utils import TrainParam
from nn_dataloader_rot import duplicate_example_checking
from nn_dataloader_rot import remove_old_examples
from train_model_h import train_model
from train_battle import run_battle
from train_identify import identify_best_model


def main():
    train_param = TrainParam()
    for idx in range(20):
        print('--------------------------------------------------- Train Round {}'.format(idx))
        train_game_play(train_param)
        duplicate_example_checking(remove_dup=False,
                                   data_path=(train_param.examples_path + train_param.examples_sub_path))
        remove_old_examples(int(train_param.round_count * 1.2),
                            data_path=(train_param.examples_path + train_param.examples_sub_path))
        train_model(train_param)
        result_file_path = run_battle()
        best_model_name = identify_best_model(train_param, result_file_path, remain_count=10)
        best_model_file = (train_param.models_path + train_param.model_backup_path + '/' + best_model_name + '/'
                           + train_param.model_filename)
        target_model_file = train_param.models_path + train_param.model_sub_path + '/' + train_param.model_filename
        print('Copy {} --> {}'.format(best_model_file, target_model_file))
        shutil.copy(best_model_file, target_model_file)


if __name__ == '__main__':
    main()
