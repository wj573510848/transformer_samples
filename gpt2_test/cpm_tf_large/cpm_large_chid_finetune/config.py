import os

class basic_config:
    def __init__(self):
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))

        
        
        self.max_length = 230 - 64
        self.batch_size = 8
        self.learning_rate = 1e-5
        self.epoch = 4
        self.print_batch = 2

        self.pretrained_model_path = ''
        self.raw_data_dir = ''
        self.data_save_path = os.path.join(self.cur_dir,'data_{}'.format(self.max_length))
        self.model_save_path = os.path.join(self.cur_dir,'model_{}'.format(self.max_length))
        self.check_folder(self.data_save_path)
        self.check_folder(self.model_save_path)

    def check_folder(self,path):
        # 检查path是否存在，不存在就创建
        if not os.path.isdir(path):
            os.makedirs(path)