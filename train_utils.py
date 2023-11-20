train_root_path = 'c:/bomb/proj/ml_reversi_train'

train_example_path = train_root_path + '/examples'
train_models_path = train_root_path + '/models'
train_record_path = train_root_path + '/records'


class TrainParam:
    def __init__(self, examples_path=None, models_path=None, record_path=None):
        if examples_path is None:
            self.examples_path = train_example_path
        else:
            self.examples_path = examples_path
        if models_path is None:
            self.models_path = train_models_path
        else:
            self.models_path = train_models_path
        if record_path is None:
            self.record_path = train_record_path
        else:
            self.record_path = train_record_path
        self.color_list = [-1, 1]
        self.round_count = 100
        self.model_label = 'players'
        self.model_sub_path = '/{}_model'.format(self.model_label)
        self.model_backup_path = '/{}_backup'.format(self.model_label)
        self.examples_sub_path = '/training_data'
        self.model_filename = '{}_model.pkl'.format(self.model_label)

