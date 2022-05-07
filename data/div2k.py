import os
from data import srdata


class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(DIV2K, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')


