import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from lib.dataset import find_dataset_using_name
from lib.utils.utils import create_logger, prepare_output_dir, worker_init_fn
from lib.core.train_config import parse_args
from lib.core.loss import SmoothNetLoss
from lib.models.smoothnet import SmoothNet
from lib.core.trainer import Trainer
import torch.optim as optim


def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment is {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)

    logger = create_logger(cfg.LOGDIR, phase='train')

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    # ========= Dataloaders ========= #
    train_datasets=[]
    test_datasets=[]

    all_estimator=cfg.ESTIMATOR.split(",")
    all_body_representation=cfg.BODY_REPRESENTATION.split(",")
    all_dataset=cfg.DATASET_NAME.split(",")

    for training_dataset_index in range(len(all_dataset)):
        estimator=all_estimator[training_dataset_index]
        body_representation=all_body_representation[training_dataset_index]
        dataset=all_dataset[training_dataset_index]

        dataset_class = find_dataset_using_name(dataset)

        print("Loading dataset ("+str(training_dataset_index)+")......")

        train_datasets.append(dataset_class(cfg,
                                    estimator=estimator,
                                    return_type=body_representation,
                                    phase='train'))

        test_datasets.append(dataset_class(cfg,
                                    estimator=estimator,
                                    return_type=body_representation,
                                    phase='test'))
    train_loader=[]
    test_loader=[]

    for train_dataset in  train_datasets:

        train_loader.append(DataLoader(dataset=train_dataset,
                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                shuffle=True,
                                num_workers=cfg.TRAIN.WORKERS_NUM,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn))

    for test_dataset in test_datasets:
        test_loader.append(DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=cfg.TRAIN.WORKERS_NUM,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn))

    # # ========= Compile Loss ========= #
    loss = SmoothNetLoss(w_accel=cfg.LOSS.W_ACCEL, w_pos=cfg.LOSS.W_POS)

    # # ========= Initialize networks ========= #
    model = SmoothNet(window_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                    output_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                    hidden_size=cfg.MODEL.HIDDEN_SIZE,
                    res_hidden_size=cfg.MODEL.RES_HIDDEN_SIZE,
                    num_blocks=cfg.MODEL.NUM_BLOCK,
                    dropout=cfg.MODEL.DROPOUT).to(cfg.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, amsgrad=True)

    # ========= Start Training ========= #
    Trainer(train_dataloader=train_loader,
            test_dataloader=test_loader,
            model=model,
            loss=loss,
            writer=writer,
            optimizer=optimizer,
            cfg=cfg).run()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)











