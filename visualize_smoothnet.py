import os
import torch
from lib.dataset import find_dataset_using_name
from lib.models.smoothnet import SmoothNet
from lib.core.visualize_config import parse_args
from lib.visualize.visualize import Visualize


def main(cfg):

    dataset_class = find_dataset_using_name(cfg.DATASET_NAME)
    test_dataset = dataset_class(cfg,
                                 estimator=cfg.ESTIMATOR,
                                 return_type=cfg.BODY_REPRESENTATION,
                                 phase='test')

    model = SmoothNet(window_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                    output_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                    hidden_size=cfg.MODEL.HIDDEN_SIZE,
                    res_hidden_size=cfg.MODEL.RES_HIDDEN_SIZE,
                    num_blocks=cfg.MODEL.NUM_BLOCK,
                    dropout=cfg.MODEL.DROPOUT).to(cfg.DEVICE)

    visualizer = Visualize(test_dataset,cfg)

    if cfg.EVALUATE.PRETRAINED != '' and os.path.isfile(
            cfg.EVALUATE.PRETRAINED):
        checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'==> Loaded pretrained model from {cfg.EVALUATE.PRETRAINED}...')
    else:
        print(f'{cfg.EVALUATE.PRETRAINED} is not a pretrained model!!!!')
        exit()
    
    visualizer.visualize(model)


if __name__ == '__main__':
    cfg, cfg_file = parse_args()

    main(cfg)