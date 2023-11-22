import pandas as pd
import shutil
from train_utils import TrainParam


def identify_best_model(param, result_file_path, remain_count=0):
    df = pd.read_csv(filepath_or_buffer=result_file_path, sep=',', index_col=0)
    df = df.sort_values(by=['win', 'score'], ascending=False)
    winner = str(df.iloc[0]['name'])
    if (len(df) > remain_count) and (remain_count > 0):
        for row_id in range(remain_count, len(df)):
            model_name = str(df.iloc[row_id]['name'])
            model_path = param.models_path + param.model_backup_path + '/' + model_name
            print('Remove model {}'.format(model_path))
            shutil.rmtree(model_path)
    return winner


if __name__ == '__main__':
    tst_param = TrainParam()
    print(identify_best_model(tst_param,
                              'c:/bomb/proj/ml_reversi_train/battle_record/battle_result_20231122141018.csv',
                              remain_count=10))
