import os

class basic_config:
    def __init__(self):
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.pretrained_model_path = ''
        
    def check_folder(self,path):
        # 检查path是否存在，不存在就创建
        if not os.path.isdir(path):
            os.makedirs(path)

class config_cpm_large_zero(basic_config):
    def __init__(self):
        super(config_cpm_large_zero,self).__init__()
        self.raw_data_dir = ''
        self.data_save_path = os.path.join(self.cur_dir,'data')
        self.check_folder(self.data_save_path)
        self.model_save_path = os.path.join(self.cur_dir,'zero_model')
        self.check_folder((self.model_save_path))

        self.batch_size=64

class config_cpm_large_zero_fixed_length(basic_config):
    def __init__(self):
        super(config_cpm_large_zero_fixed_length,self).__init__()
        self.raw_data_dir = ''
        self.data_save_path = os.path.join(self.cur_dir,'data_fixed_length')
        self.check_folder(self.data_save_path)
        self.model_save_path = os.path.join(self.cur_dir,'zero_model')
        self.check_folder((self.model_save_path))

        self.batch_size = 128
        self.max_length = 128
