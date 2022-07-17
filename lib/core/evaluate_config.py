import argparse
from yacs.config import CfgNode as CN

# CONSTANTS
# You may modify them at will
BASE_DATA_DIR = 'data/poses'  # data dir

# Configuration variables
cfg = CN()
cfg.DEVICE = 'cuda'  # training device 'cuda' | 'cpu'
cfg.SEED_VALUE = 4321  # random seed
cfg.LOGDIR = ''  # log dir
cfg.EXP_NAME = 'default'  # experiment name
cfg.DEBUG = True  # debug
cfg.OUTPUT_DIR = 'results'  # output folder

cfg.DATASET_NAME = ''  # dataset name
cfg.ESTIMATOR = ''  # backbone estimator name
cfg.BODY_REPRESENTATION = ''  # 3D | 2D | smpl

cfg.SMPL_MODEL_DIR = "data/smpl/"  # smpl model dir

# CUDNN config
cfg.CUDNN = CN()  # cudnn config
cfg.CUDNN.BENCHMARK = True  # cudnn config
cfg.CUDNN.DETERMINISTIC = False  # cudnn config
cfg.CUDNN.ENABLED = True  # cudnn config

# dataset config
cfg.DATASET = CN()
cfg.DATASET.BASE_DIR=BASE_DATA_DIR
cfg.DATASET.ROOT_AIST_SPIN_3D=[2,3]
cfg.DATASET.ROOT_AIST_TCMR_3D=[2,3]
cfg.DATASET.ROOT_AIST_VIBE_3D=[2,3]
cfg.DATASET.ROOT_H36M_FCN_3D=[0]
cfg.DATASET.ROOT_H36M_RLE_3D=[0]
cfg.DATASET.ROOT_H36M_TCMR_3D=[2,3]
cfg.DATASET.ROOT_H36M_VIBE_3D=[2,3]
cfg.DATASET.ROOT_H36M_VIDEOPOSET27_3D=[0]
cfg.DATASET.ROOT_H36M_VIDEOPOSET81_3D=[0]
cfg.DATASET.ROOT_H36M_VIDEOPOSET243_3D=[0]
cfg.DATASET.ROOT_MPIINF3DHP_SPIN_3D=[14]
cfg.DATASET.ROOT_MPIINF3DHP_TCMR_3D=[14]
cfg.DATASET.ROOT_MPIINF3DHP_VIBE_3D=[14]
cfg.DATASET.ROOT_MUPOTS_TPOSENET_3D=[14]
cfg.DATASET.ROOT_MUPOTS_TPOSENETREFINENET_3D=[14]
cfg.DATASET.ROOT_PW3D_EFT_3D=[2,3]
cfg.DATASET.ROOT_PW3D_PARE_3D=[2,3]
cfg.DATASET.ROOT_PW3D_SPIN_3D=[2,3]
cfg.DATASET.ROOT_PW3D_TCMR_3D=[2,3]
cfg.DATASET.ROOT_PW3D_VIBE_3D=[2,3]
cfg.DATASET.ROOT_H36M_MIX_3D=[0]


# model config
cfg.MODEL = CN()
cfg.MODEL.SLIDE_WINDOW_SIZE = 100  # slide window size

cfg.MODEL.HIDDEN_SIZE=512 # hidden size
cfg.MODEL.RES_HIDDEN_SIZE=256 # res hidden size
cfg.MODEL.NUM_BLOCK=3 # block number
cfg.MODEL.DROPOUT=0.5 # dropout

# training config
cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE = 1024  # batch size
cfg.TRAIN.WORKERS_NUM = 0  # workers number
cfg.TRAIN.EPOCH = 70  # epoch number
cfg.TRAIN.LR = 0.001  # learning rate
cfg.TRAIN.LRDECAY = 0.95  # learning rate decay rate
cfg.TRAIN.RESUME = None  # resume training checkpoint path
cfg.TRAIN.VALIDATE = True  # validate while training
cfg.TRAIN.USE_6D_SMPL = True  # True: use 6D rotation | False: use Rotation Vectors (only take effect when cfg.TRAIN.USE_SMPL_LOSS=False )

# test config
cfg.EVALUATE = CN()
cfg.EVALUATE.PRETRAINED = ''  # evaluation checkpoint
cfg.EVALUATE.ROOT_RELATIVE = True  # root relative represntation in error caculation
cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE = 1  # slide window step size
cfg.EVALUATE.TRADITION='' # traditional filter for comparison
cfg.EVALUATE.TRADITION_SAVGOL=CN()
cfg.EVALUATE.TRADITION_SAVGOL.WINDOW_SIZE=31
cfg.EVALUATE.TRADITION_SAVGOL.POLYORDER=2
cfg.EVALUATE.TRADITION_GAUS1D=CN()
cfg.EVALUATE.TRADITION_GAUS1D.WINDOW_SIZE=31
cfg.EVALUATE.TRADITION_GAUS1D.SIGMA=3
cfg.EVALUATE.TRADITION_ONEEURO=CN()
cfg.EVALUATE.TRADITION_ONEEURO.MIN_CUTOFF=0.04
cfg.EVALUATE.TRADITION_ONEEURO.BETA=0.7


# loss config
cfg.LOSS = CN()
cfg.LOSS.W_ACCEL = 1.0  # loss w accel
cfg.LOSS.W_POS = 1.0  # loss w position

# log config
cfg.LOG = CN()
cfg.LOG.NAME = ''  # log name


def get_cfg_defaults():
    """Get yacs CfgNode object with default values"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--checkpoint', type=str, help='pretrained checkpoint file path')
    parser.add_argument('--dataset_name',
                        type=str,
                        help='dataset name [pw3d, h36m, jhmdb, pw3d]')
    parser.add_argument(
        '--estimator',
        type=str,
        help='backbone estimator name [spin, eft, pare, pw3d, fcn, simplepose]'
    )
    parser.add_argument('--body_representation',
                        type=str,
                        help='human body representation [2D, 3D, smpl]')
    parser.add_argument('--slide_window_size',
                        type=int,
                        help='slide window size')
    parser.add_argument('--tradition',
                        type=str,
                        default="",
                        help='traditional filters [savgol,oneeuro,gaus1d]')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    cfg.DATASET_NAME = args.dataset_name
    cfg.ESTIMATOR = args.estimator
    cfg.BODY_REPRESENTATION = args.body_representation
    cfg.MODEL.SLIDE_WINDOW_SIZE=args.slide_window_size

    cfg.EVALUATE.PRETRAINED = args.checkpoint
    cfg.EVALUATE.TRADITION = args.tradition

    return cfg, cfg_file