import torch
from lib.utils.eval_metrics import *
from lib.utils.geometry_utils import *
import os
import cv2
from lib.visualize.visualize_3d import visualize_3d
from lib.visualize.visualize_smpl import visualize_smpl
from lib.visualize.visualize_2d import visualize_2d
import sys


class Visualize():

    def __init__(self,test_dataset, cfg):

        self.cfg = cfg
        self.device = cfg.DEVICE

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.estimator = self.cfg.ESTIMATOR
        self.dataset_name = self.cfg.DATASET_NAME
        self.body_representation = self.cfg.BODY_REPRESENTATION

        self.vis_seq_index = self.cfg.VIS.INPUT_VIDEO_NUMBER
        self.vis_output_video_path = self.cfg.VIS.OUTPUT_VIDEO_PATH

        self.slide_window_size = self.cfg.MODEL.SLIDE_WINDOW_SIZE
        self.slide_window_step = self.cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE

        self.base_data_path = self.cfg.DATASET.BASE_DIR

        self.phase="test"

        try:
            self.ground_truth_data = np.load(os.path.join(
                self.base_data_path,
                self.dataset_name+"_"+self.estimator+"_"+self.body_representation,
                "groundtruth",
                self.dataset_name + "_" + "gt"+"_"+self.body_representation + "_" + self.phase + ".npz"),
                                        allow_pickle=True)
        except:
            raise ImportError("Ground-truth data do not exist!")

        try:
            self.detected_data = np.load(os.path.join(
                self.base_data_path, 
                self.dataset_name+"_"+self.estimator+"_"+self.body_representation,
                "detected",
                self.dataset_name + "_" + self.estimator+"_"+self.body_representation + "_" + self.phase + ".npz"),
                                        allow_pickle=True)
        except:
            raise ImportError("Detected data do not exist!")

        self.device = self.cfg.DEVICE

        if self.body_representation == '3D':
            self.input_dimension = self.ground_truth_data["joints_3d"][0].shape[-1]
        elif self.body_representation == 'smpl':
            if cfg.TRAIN.USE_6D_SMPL:
                self.input_dimension = 6 * 24
            else:
                self.input_dimension = 3 * 24
        elif self.body_representation == '2D':
            self.input_dimension = self.ground_truth_data["joints_2d"][0].shape[-1]

    def visualize_3d(self, model):
        keypoint_number =self.input_dimension//3
        keypoint_root = eval("self.cfg.DATASET." +
                               "ROOT_"+self.cfg.DATASET_NAME.upper() +"_"+ self.cfg.ESTIMATOR.upper()+"_3D")

        data_gt = self.ground_truth_data["joints_3d"][self.vis_seq_index].reshape(-1,keypoint_number,3)
        data_pred = self.detected_data["joints_3d"][self.vis_seq_index].reshape(-1,keypoint_number,3)

        data_gt = data_gt - data_gt[:, keypoint_root, :].mean(axis=1).reshape(-1, 1, 3)
        data_pred = data_pred - data_pred[:, keypoint_root, :].mean(axis=1).reshape(-1, 1, 3)

        data_gt = torch.tensor(data_gt).to(self.device)
        data_pred = torch.tensor(data_pred).to(self.device)

        data_len = data_pred.shape[0]
        data_pred_window = torch.as_strided(
            data_pred, ((data_len - self.slide_window_size) // self.slide_window_step+1,
                        self.slide_window_size, keypoint_number, 3),
            (self.slide_window_step * keypoint_number * 3,
             keypoint_number * 3, 3, 1),
            storage_offset=0).reshape(-1, self.slide_window_size,
                                      self.input_dimension)

        with torch.no_grad():
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos= model(data_pred_window)
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos=predicted_pos.permute(0,2,1)

        predicted_pos = slide_window_to_sequence(predicted_pos,self.slide_window_step,self.slide_window_size).reshape(-1, keypoint_number, 3)

        data_len = predicted_pos.shape[0]
        data_pred = data_pred[:data_len, :].reshape(-1, keypoint_number, 3)
        data_gt = data_gt[:data_len, :].reshape(-1, keypoint_number, 3)

        if self.dataset_name in ["aist","h36m","mpiinf3dhp","mupots","pw3d"]:
            data_gt = data_gt.reshape(-1, keypoint_number, 3)
            data_pred = data_pred.reshape(-1, keypoint_number, 3)
            predicted_pos = predicted_pos.reshape(-1, keypoint_number, 3)

            vis_output_video_name = self.dataset_name+"_"+self.estimator+"_3D_" + str(
                self.vis_seq_index) +"_frame_" +str(self.cfg.VIS.START)+"-"+str(self.cfg.VIS.END)+".mp4"

            visualize_3d(
                self.vis_output_video_path,
                vis_output_video_name,
                data_pred,
                data_gt,
                predicted_pos,
                self.cfg.VIS.START,
                self.cfg.VIS.END,
                self.dataset_name,
                self.estimator
            )
        else:
            print("Not Implemented!")


    def visualize_smpl(self, model):

        data_gt = self.ground_truth_data["pose"][self.vis_seq_index]
        data_pred = self.detected_data["pose"][self.vis_seq_index]

        if self.cfg.TRAIN.USE_6D_SMPL:
            data_pred = numpy_axis_to_rot6D(data_pred.reshape(-1, 3)).reshape(
                -1, self.input_dimension)

        data_imgname = self.ground_truth_data["imgname"][self.vis_seq_index]

        data_gt = torch.tensor(data_gt).to(self.device)
        data_pred = torch.tensor(data_pred).to(self.device)

        data_len = data_pred.shape[0]
        data_pred_window = torch.as_strided(
            data_pred, ((data_len - self.slide_window_size) // self.slide_window_step+1,
                        self.slide_window_size, self.input_dimension),
            (self.slide_window_step * self.input_dimension,
             self.input_dimension, 1),
            storage_offset=0).reshape(-1, self.slide_window_size,
                                      self.input_dimension)

        with torch.no_grad():
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos= model(data_pred_window)
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos=predicted_pos.permute(0,2,1)

        predicted_pos = slide_window_to_sequence(predicted_pos,self.slide_window_step,self.slide_window_size).reshape(-1, self.input_dimension)

        data_len = predicted_pos.shape[0]
        data_pred = data_pred[:data_len, :]
        data_gt = data_gt[:data_len, :]

        data_imgname = data_imgname[:data_len]

        if self.cfg.TRAIN.USE_6D_SMPL:
            data_pred = rot6D_to_axis(data_pred.reshape(-1, 6)).reshape(
                -1, 24 * 3)
            predicted_pos = rot6D_to_axis(predicted_pos.reshape(-1,
                                                                6)).reshape(
                                                                    -1, 24 * 3)

        data_gt = np.array(data_gt.reshape(-1, 24 * 3).cpu())
        data_pred = np.array(data_pred.reshape(-1, 24 * 3).cpu())
        predicted_pos = np.array(predicted_pos.reshape(-1, 24 * 3).cpu())

        smpl_neural = SMPL(model_path=self.cfg.SMPL_MODEL_DIR,
                           create_transl=False)

        if self.dataset_name in ["aist","h36m","pw3d"]:
            vis_output_video_name = self.dataset_name+"_"+self.estimator+"_SMPL_" + str(
                self.vis_seq_index) +"_frame_" +str(self.cfg.VIS.START)+"-"+str(self.cfg.VIS.END)+ ".mp4"

            visualize_smpl(
                self.vis_output_video_path,
                vis_output_video_name,
                smpl_neural,
                data_pred,
                data_gt,
                predicted_pos,
                self.cfg.VIS.START,
                self.cfg.VIS.END,
            )

    def visualize_2d(self, model):
        keypoint_number =self.input_dimension//2

        data_gt = self.ground_truth_data["joints_2d"][self.vis_seq_index]
        data_pred = self.detected_data["joints_2d"][self.vis_seq_index]

        if self.dataset_name=="jhmdb":
            data_imageshape=self.ground_truth_data["imgshape"][self.vis_seq_index][:2][::-1].copy()
        elif self.dataset_name=="h36m":
            data_imageshape=1000

        data_gt = torch.tensor(data_gt).to(self.device)
        len_seq = data_gt.shape[0]
        data_pred=data_pred[:len_seq,:]
        if self.dataset_name=="jhmdb":
            data_pred_norm=torch.tensor(data_pred.reshape(-1,2)/data_imageshape).to(self.device).reshape_as(data_gt)
        elif self.dataset_name=="h36m":
            data_pred_norm =(torch.tensor(data_pred).to(self.device)-data_imageshape/2)/(data_imageshape/2) # normalization

        data_pred = torch.tensor(data_pred).to(self.device)

        data_len = data_pred.shape[0]
        data_pred_window = torch.as_strided(
            data_pred_norm, ((data_len - self.slide_window_size) // self.slide_window_step+1,
                        self.slide_window_size, keypoint_number, 2),
            (self.slide_window_step * keypoint_number * 2,
             keypoint_number * 2, 2, 1),
            storage_offset=0).reshape(-1, self.slide_window_size,
                                      self.input_dimension)

        with torch.no_grad():
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos = model(data_pred_window)
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos=predicted_pos.permute(0,2,1)

        predicted_pos = slide_window_to_sequence(predicted_pos,self.slide_window_step,self.slide_window_size).reshape(-1, keypoint_number, 2)

        data_len = predicted_pos.shape[0]
        data_pred = data_pred[:data_len, :].reshape(-1, keypoint_number, 2)
        data_gt = data_gt[:data_len, :].reshape(-1, keypoint_number, 2)
        if self.dataset_name=="jhmdb":
            predicted_pos = predicted_pos.reshape(-1, keypoint_number, 2)*torch.tensor(data_imageshape).to(predicted_pos.device)
        elif self.dataset_name=="h36m":
            predicted_pos = predicted_pos.reshape(-1, keypoint_number, 2)*torch.tensor(data_imageshape/2).to(predicted_pos.device)+torch.tensor(data_imageshape/2)

        if self.dataset_name in ["jhmdb","h36m"]:
            vis_output_video_name = self.dataset_name+"_"+self.estimator+"_2D_" + str(
                self.vis_seq_index) +"_frame_" +str(self.cfg.VIS.START)+"-"+str(self.cfg.VIS.END)+ ".mp4"

            visualize_2d(
                self.vis_output_video_path,
                vis_output_video_name,
                predicted_pos,
                data_pred,
                data_gt,
                self.cfg.VIS.START,
                self.cfg.VIS.END,
                self.dataset_name,
                self.estimator
            )
        else:
            print("Not Implemented!")

    def visualize(self, model):
        model.eval()
        if self.cfg.BODY_REPRESENTATION == "3D":
            self.visualize_3d(model)

        elif self.cfg.BODY_REPRESENTATION == "smpl":
            self.visualize_smpl(model)

        elif self.cfg.BODY_REPRESENTATION == "2D":
            self.visualize_2d(model)
