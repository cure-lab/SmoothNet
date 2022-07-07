import os
from lib.utils.eval_metrics import calculate_mpjpe, calculate_pampjpe, calculate_accel_error
import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import trange
from matplotlib.backends.backend_agg import FigureCanvasAgg
from lib.visualize.visualize_skeleton import *

SUB_FIG_SIZE = 4
SUB_FIG_UNIT = 100
VIEW_NUM = 1

SKELETON_AX_DIST = 6.5
FRAME_RATE = 30

SKELRTON_COLOR = ['red', 'black']


def plot_skeleton(ax_2d, skeleton, color, edges, joints):
    skeleton=-skeleton
    center = np.mean(skeleton, axis=0)
    max_len=np.max(np.max(skeleton,axis=0)-np.min(skeleton,axis=0))

    ax_2d.axis(xmin=center[0] - max_len,
                xmax=center[0] + max_len,
                ymin=center[1] - max_len,
                ymax=center[1] + max_len)
   
    for idx, lr in joints:
        joint_color = color[0] if lr else color[1]
        ax_2d.plot(skeleton[idx, 0], skeleton[idx, 1],
                     joint_color)
        ax_2d.text(skeleton[idx, 0], skeleton[idx, 1],
                     str(idx))

    for kpta, kptb, lr in edges:
        line_color = color[0] if lr else color[1]
        ax_2d.plot(skeleton[[kpta, kptb], 0], skeleton[[kpta, kptb], 1],
                    line_color)

    return


def visualize_2d(vis_output_video_path,
                      vis_output_video_name,
                      predicted_pos,
                      data_pred,
                      data_gt,
                      start_frame,
                      end_frame,
                      dataset_name,
                      estimator_name):
    print("Visualizing the result ...")

    dataset_edges=eval(dataset_name.upper()+"_"+estimator_name.upper()+"_2D_EDGES")
    dataset_joints=eval(dataset_name.upper()+"_"+estimator_name.upper()+"_2D_JOINTS")

    if not os.path.exists(vis_output_video_path):
        os.makedirs(vis_output_video_path)

    len_seq = data_gt.shape[0]

    # calculate errors
    mpjpe_in = np.array(calculate_mpjpe(data_pred.reshape(len_seq,-1,2), data_gt.reshape(len_seq,-1,2)).cpu())
    mpjpe_out = np.array(calculate_mpjpe(predicted_pos.reshape(len_seq,-1,2), data_gt.reshape(len_seq,-1,2)).cpu())
    
    acc_in = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(data_pred.reshape(len_seq,-1,2),
                                        data_gt.reshape(len_seq,-1,2)).cpu()), np.array([0])))
    acc_out = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(predicted_pos.reshape(len_seq,-1,2),
                                        data_gt.reshape(len_seq,-1,2)).cpu()), np.array([0])))

    data_pred = np.array(data_pred.cpu())
    data_gt = np.array(data_gt.cpu())
    predicted_pos = np.array(predicted_pos.cpu())

    anim_output = {
        'Estimator': data_pred,
        'Estimator + SmoothNet': predicted_pos,
        'Ground truth': data_gt
    }

    videoWriter = cv2.VideoWriter(
        os.path.join(vis_output_video_path, vis_output_video_name),
        cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE,
        (SUB_FIG_SIZE * VIEW_NUM * len(anim_output) * SUB_FIG_UNIT,
         SUB_FIG_SIZE * 3 * SUB_FIG_UNIT))

    for frame_i in trange(max(0, start_frame), min(len_seq, end_frame)):
        fig = plt.figure(figsize=(SUB_FIG_SIZE * VIEW_NUM * len(anim_output),
                                  SUB_FIG_SIZE * 3))
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=None,
                            hspace=0.5)

        for view_i in range(VIEW_NUM):
            for index, (title, data) in enumerate(anim_output.items()):
                ax_2d = fig.add_subplot(3,
                                        len(anim_output) * VIEW_NUM,
                                        (view_i) * len(anim_output) + index +
                                        1)

                ax_2d.set_xticklabels([])
                ax_2d.set_yticklabels([])

                ax_2d.dist = SKELETON_AX_DIST
                ax_2d.set_title(title, fontsize=3*SUB_FIG_SIZE)

                plot_skeleton(ax_2d, data[frame_i, :, :], SKELRTON_COLOR,
                              dataset_edges, dataset_joints)

        ax_acc = fig.add_subplot(3, 1, 2)
        ax_acc.set_title('Accel Error Visualize', fontsize=4*SUB_FIG_SIZE)

        ax_mpjpe = fig.add_subplot(3, 1, 3)
        ax_mpjpe.set_title('MPJPE Visualize', fontsize=4*SUB_FIG_SIZE)

        ax_acc.plot(acc_in[:frame_i],
                    color=(202 / 255, 0 / 255, 32 / 255),
                    label='Estimator (Accel)')
        ax_acc.plot(acc_out[:frame_i],
                    'c',
                    label='Estimator + SmoothNet (Accel)')
        
        ax_acc.legend()
        ax_acc.grid(True)
        ax_acc.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
        ax_acc.set_ylabel('Mean Acceleration Error (pixel/s2)', fontsize=3*SUB_FIG_SIZE)
        ax_acc.set_xlim((0, len(acc_in)))
        ax_acc.set_ylim((0, np.max((np.max(acc_in), np.max(acc_out)))))
        ax_acc.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
        ax_acc.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
        ax_acc.legend(fontsize=3*SUB_FIG_SIZE)

        ax_mpjpe.plot(mpjpe_in[:frame_i],
                      color=(202 / 255, 0 / 255, 32 / 255),
                      label='Estimator (MPJPE)')
        ax_mpjpe.plot(mpjpe_out[:frame_i],
                      'c',
                      label='Estimator + SmoothNet (MPJPE)')
        
        ax_mpjpe.legend()
        ax_mpjpe.grid(True)
        ax_mpjpe.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
        ax_mpjpe.set_ylabel('Mean Position Error (pixel)', fontsize=3*SUB_FIG_SIZE)
        ax_mpjpe.set_xlim((0, len(mpjpe_in)))
        ax_mpjpe.set_ylim((0, np.max((np.max(mpjpe_in), np.max(mpjpe_out)))))
        ax_mpjpe.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
        ax_mpjpe.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
        ax_mpjpe.legend(fontsize=3*SUB_FIG_SIZE)

        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        final_img = np.array(canvas.renderer.buffer_rgba())[:, :, [2, 1, 0]]

        #plt.savefig("tmp" + str(frame_i) + ".png")

        videoWriter.write(final_img)
        plt.close()

    videoWriter.release()
    print(f"Finish! The video is stored in "+os.path.join(vis_output_video_path, vis_output_video_name))