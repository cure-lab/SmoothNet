import os
import torch
from lib.dataset import find_dataset_using_name
from lib.models.smoothnet import SmoothNet
from lib.core.evaluate import Evaluator
from torch.utils.data import DataLoader
from lib.utils.utils import prepare_output_dir, worker_init_fn
from lib.core.evaluate_config import parse_args


def main(cfg):
    test_datasets=[]

    all_estimator=cfg.ESTIMATOR.split(",")
    all_body_representation=cfg.BODY_REPRESENTATION.split(",")
    all_dataset=cfg.DATASET_NAME.split(",")

    for dataset_index in range(len(all_dataset)):
        estimator=all_estimator[dataset_index]
        body_representation=all_body_representation[dataset_index]
        dataset=all_dataset[dataset_index]

        dataset_class = find_dataset_using_name(dataset)

        print("Loading dataset ("+str(dataset_index)+")......")

        test_datasets.append(dataset_class(cfg,
                                    estimator=estimator,
                                    return_type=body_representation,
                                    phase='test'))
    test_loader=[]

    for test_dataset in test_datasets:
        test_loader.append(DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=cfg.TRAIN.WORKERS_NUM,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn))

    model = SmoothNet(window_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                    output_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                    hidden_size=cfg.MODEL.HIDDEN_SIZE,
                    res_hidden_size=cfg.MODEL.RES_HIDDEN_SIZE,
                    num_blocks=cfg.MODEL.NUM_BLOCK,
                    dropout=cfg.MODEL.DROPOUT).to(cfg.DEVICE)

    if cfg.EVALUATE.PRETRAINED != '' and os.path.isfile(
            cfg.EVALUATE.PRETRAINED):
        checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
        performance = checkpoint['performance']
        model.load_state_dict(checkpoint['state_dict'])
        print(f'==> Loaded pretrained model from {cfg.EVALUATE.PRETRAINED}...')
    else:
        print(f'{cfg.EVALUATE.PRETRAINED} is not a pretrained model!!!!')
        exit()

    evaluator = Evaluator(model=model, test_loader=test_loader, cfg=cfg)
    evaluator.calculate_flops()
    evaluator.calculate_parameter_number()
    evaluator.run()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)