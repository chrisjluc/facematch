import consts

from models import AutoEncoderModel
from storage import Writer

class GPUTask(object):
    """
    A task that is to be run on a GPU
    and is scheduled by the TaskManager
    """
    def run(self):
        raise NotImplementedError

class TrainingTask(GPUTask):

    def __init__(self, model_cls, X_train, Y_train, user_ids, model_name):
        self.model_cls = model_cls
        self.X_train = X_train
        self.Y_train = Y_train
        self.user_ids = user_ids
        self.model_name = model_name

    def run(self):
        model = self.model_cls(self.model_name, self.user_ids, self.X_train, self.Y_train)
        model.train()
        model.save()


class ActivationExtractionTask(GPUTask):

    def __init__(self, model_cls, model_name, X_train, user_ids):
        self.model_cls = model_cls
        self.model_name = model_name
        self.X_train = X_train
        self.user_ids = user_ids

    def run(self):
        model = self.model_cls(self.model_name, self.user_ids, self.X_train, self.X_train)
        model.load()
        model.save_activations()


class TrainingAutoEncoderTask(GPUTask):

    def __init__(self, model_name, X_train, input_size, encoding_size):
        self.model_name = model_name
        self.X_train = X_train
        self.input_size = input_size
        self.encoding_size = encoding_size

    def run(self):
        model = AutoEncoderModel(self.model_name, self.input_size, self.encoding_size, self.X_train)
        model.train()
        model.save()
        model.save_activations()

