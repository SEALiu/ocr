import os
import os.path as osp
import datetime
import json

class Config(object):
    LOG_DIR = 'logs'
    CHECKPOINT_DIR = 'checkpoints'

    def __init__(self):
        """Set values of computed attributes."""

        if not osp.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)

        self.CHECKPOINT_DIR = osp.join(self.CHECKPOINT_DIR, str(datetime.date.today()))
        if not osp.exists(self.CHECKPOINT_DIR):
            os.makedirs(self.CHECKPOINT_DIR)


class DBConfig(Config):

    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: ResNet50
    BACKBONE = "ResNet50"


    # train
    EPOCHS = 1000
    INITIAL_EPOCH = 0
    # PRETRAINED_MODEL_PATH = 'checkpoints/ckpt/db_173_2.0138_2.0660.h5'
    PRETRAINED_MODEL_PATH = 'checkpoints/db_167_1.9499_1.9947.h5'

    CHECKPOINT_DIR = 'checkpoints/dbnet'

    LEARNING_RATE = 1e-4

    # dataset
    IGNORE_TEXT = ["*", "###"]

    TRAIN_DATA_PATH = '/hd2/zonas/data/text_detection/merge/train.json'
    VAL_DATA_PATH = '/hd2/zonas/data/text_detection/merge/val.json'

    IMAGE_SIZE = 640
    BATCH_SIZE = 8

    MIN_TEXT_SIZE = 8
    SHRINK_RATIO = 0.4

    THRESH_MIN = 0.3
    THRESH_MAX = 0.7


    def __init__(self):
        super().__init__()


class CRNNConfig(Config):
    EPOCHS = 20

    BATCH_SIZE = 256

    BUFFER_SIZE = 10000

    LEARNING_RATE = 1e-3

    INPUT_SHAPE = (32, 320, 3)

    TRAIN_DATA_PATH= '/Users/yang/PycharmProjects/ocr-data/ocr_train/'

    CHAR_PATH = './data/char.json'

    TABLE_PATH = './data/table.txt'

    with open(CHAR_PATH, 'r') as f:
        char_dic = json.load(f)

    NUM_CLASSES = len(char_dic) + 1

    with open(TABLE_PATH, 'w') as fw:
        for char in char_dic:
            fw.write(char + '\n')

    CHECKPOINT_DIR = 'checkpoints/crnn'

    def __init__(self):
        super().__init__()

