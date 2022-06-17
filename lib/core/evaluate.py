import torch
from lib.utils.eval_metrics import *
from lib.utils.geometry_utils import *
from thop import profile


class Evaluator():

    def __init__(
        self,
        test_loader,
        model,
        cfg,
    ):
        self.test_dataloader = test_loader
        self.model = model
        self.device = cfg.DEVICE
        self.cfg = cfg

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self):
        self.evaluate()

    def calculate_parameter_number(self):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters()
                            if p.requires_grad)
        log_str = f'Total Parameters: {total_num/(1000 ** 2)} M, Trainable Parameters: {trainable_num /(1000 ** 2)} M'
        print(log_str)
        return {'Total': total_num, 'Trainable': trainable_num}

    def calculate_flops(self):
        data = torch.randn(
            (10,17*3, self.cfg.MODEL.SLIDE_WINDOW_SIZE)).to(self.device)
        flops, _ = profile(self.model, inputs=(data,))
        log_str = f'Flops Per Frame: {flops/self.cfg.MODEL.SLIDE_WINDOW_SIZE/10/(1000 ** 3)} G'
        print(log_str)
        return {'Flops': flops}

    def evaluate_3d(self,dataset_index,dataset,estimator):
        eval_dict = evaluate_smoothnet_3D(self.model, self.test_dataloader[dataset_index],
                                          self.device, dataset,estimator,self.cfg)

        log_str = ' '.join(
            [f'{k.upper()}: {v:.2f},' for k, v in eval_dict.items()])
        print(log_str)

        return eval_dict

    def evaluate_smpl(self,dataset_index,dataset):

        eval_dict = evaluate_smoothnet_smpl(self.model, self.test_dataloader[dataset_index],
                                            self.device, self.cfg,dataset)

        log_str = ' '.join(
            [f'{k.upper()}: {v:.2f},' for k, v in eval_dict.items()])
        print(log_str)

        return eval_dict

    def evaluate_2d(self,dataset_index,dataset):
        eval_dict = evaluate_smoothnet_2D(self.model, self.test_dataloader[dataset_index],
                                          self.device, self.cfg,dataset_index,dataset)

        if dataset == "jhmdb":
            log_str = "" + ' '.join(
                [f'{k.upper()}: {v*100:.2f}%,' for k, v in eval_dict.items()])
        elif dataset == "h36m":
            log_str = ' '.join([f'{k.upper()}: {v:.2f},' for k, v in eval_dict.items()])
            
        print(log_str)

        return eval_dict

    def evaluate(self):
        self.model.eval()

        performance=[]
        all_dataset=self.cfg.DATASET_NAME.split(",")
        all_body_representation=self.cfg.BODY_REPRESENTATION.split(",")
        all_estimator=self.cfg.ESTIMATOR.split(",")

        for dataset_index in range(len(all_dataset)):
            present_representation= all_body_representation[dataset_index]
            present_dataset=all_dataset[dataset_index]
            present_estimator=all_estimator[dataset_index]
            print("=======================================================")
            print("evaluate on dataset: "+present_dataset+", estimator: "+present_estimator+", body representation: "+present_representation)
            if present_representation == "3D":
                performance.append(self.evaluate_3d(dataset_index,present_dataset,present_estimator))

            elif present_representation == "smpl":
                performance.append(self.evaluate_smpl(dataset_index,present_dataset))

            elif present_representation == "2D":
                performance.append(self.evaluate_2d(dataset_index,present_dataset))

        return performance

