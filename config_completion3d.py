from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Dataset Config
#
__C.DATASETS = edict()
__C.DATASETS.COMPLETION3D = edict()
__C.DATASETS.COMPLETION3D.ROOT                  = 'R:/dataset2019/shapenet'
__C.DATASETS.COMPLETION3D.N_POINTS = 2048

#
# Dataset
#
__C.DATASET = edict()
__C.DATASET.TRAIN_DATASET = 'Completion3D'
__C.DATASET.TEST_DATASET = 'Completion3D'
__C.DATASET.VAL_DATASET = 'Completion3D'

#
# Constants
#
__C.CONST = edict()
__C.CONST.NUM_WORKERS                            = 2
__C.CONST.N_INPUT_POINTS = 2048

#
# Directories
#
__C.DIR = edict()
__C.DIR.OUT_PATH = 'SVDFormer_Comp3D'
__C.CONST.DEVICE = '0'

#
# Memcached
#
__C.MEMCACHED = edict()
__C.MEMCACHED.ENABLED = False
__C.MEMCACHED.LIBRARY_PATH = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG = '/mnt/lustre/share/memcached_server/client_list.conf'
__C.MEMCACHED.CLIENT_CONFIG = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK = edict()
__C.NETWORK.N_SAMPLING_POINTS = 2048
__C.NETWORK.step1 = 4
__C.NETWORK.step2 = 8
__C.NETWORK.merge_points = 256
__C.NETWORK.DILATION = 1
__C.NETWORK.local_points = 512
__C.NETWORK.view_distance = 0.7

#
# Train
#
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE                             = 4
__C.TRAIN.N_EPOCHS = 2
__C.TRAIN.SAVE_FREQ = 50
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.LR_MILESTONES = [50, 100, 150, 200, 250]
__C.TRAIN.LR_DECAY_STEP = [40, 80, 120, 160, 200, 240, 280, 320, 360]
__C.TRAIN.WARMUP_STEPS = 300
__C.TRAIN.GAMMA = 0.7
__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.WEIGHT_DECAY = 0

#
# Test
#
__C.TEST = edict()
__C.TEST.METRIC_NAME = 'ChamferDistance'
