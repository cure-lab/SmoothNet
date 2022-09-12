import torch
import os
import cv2
import numpy as np
from tqdm import trange
from lib.utils.eval_metrics import calculate_mpjpe, calculate_pampjpe, calculate_accel_error

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import trange
from matplotlib.backends.backend_agg import FigureCanvasAgg

################# pw3d ##################
SUB_FIG_SIZE = 4
SUB_FIG_UNIT = 100
VIEW_NUM = 2
VIEW_CAMERA = [[10, 10], [-45, 10]]

SKELETON_3D_RADIUS = 1.7
SKELETON_AX_DIST = 6.5
FRAME_RATE = 30


def plot_vertics(ax_3d, data):
    data = -data[:, [2, 0, 1]]
    center = np.mean(data, axis=0)*1.2

    ax_3d.set_xlim3d([
        center[0] - SKELETON_3D_RADIUS / 2, center[0] + SKELETON_3D_RADIUS / 2,
    ])
    ax_3d.set_ylim3d([
        center[1] - SKELETON_3D_RADIUS / 2, center[1] + SKELETON_3D_RADIUS / 2
    ])
    ax_3d.set_zlim3d([
        center[2] - SKELETON_3D_RADIUS / 2, center[2] + SKELETON_3D_RADIUS / 2
    ])

    ax_3d.scatter(data[:,0],data[:,1],data[:,2],alpha=0.4,marker='.',s=0.5)

    return



def visualize_smpl(vis_output_video_path,
                        vis_output_video_name,
                        smpl_neural,
                        in_poses,
                        gt_poses,
                        smoothnet_poses,
                        start_frame,
                        end_frame,
                        interval=10):
    print("Visualizing the result ...")

    with torch.no_grad():
        in_smpl = smpl_neural(
            body_pose=torch.from_numpy(in_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(in_poses[:, 0:3]).float())

        gt_smpl = smpl_neural(
            body_pose=torch.from_numpy(gt_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(gt_poses[:, 0:3]).float())

        smoothnet_smpl = smpl_neural(
            body_pose=torch.from_numpy(smoothnet_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(smoothnet_poses[:, 0:3]).float())


    if not os.path.exists(vis_output_video_path):
        os.makedirs(vis_output_video_path)

    m2mm=1000

    predicted_vertices = smoothnet_smpl.vertices
    vertices_pred = in_smpl.vertices
    vertices_gt = gt_smpl.vertices

    predicted_joints =smoothnet_smpl.joints
    joints_pred=in_smpl.joints
    joints_gt=gt_smpl.joints

    len_seq = joints_gt.shape[0]

    # calculate errors
    mpjpe_in = np.array(calculate_mpjpe(joints_pred, joints_gt).cpu())*m2mm
    mpjpe_out = np.array(calculate_mpjpe(predicted_joints, joints_gt).cpu())*m2mm
    
    acc_in = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(joints_pred,
                                        joints_gt).cpu()), np.array([0])))*m2mm
    acc_out = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(predicted_joints,
                                        joints_gt).cpu()), np.array([0])))*m2mm

    vertices_pred = np.array(vertices_pred.cpu())
    vertices_gt = np.array(vertices_gt.cpu())
    predicted_vertices = np.array(predicted_vertices.cpu())

    anim_output = {
        'Estimator': vertices_pred,
        'Estimator + SmoothNet': predicted_vertices,
        'Ground truth': vertices_gt
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
            view_camera = VIEW_CAMERA[view_i]
            for index, (title, data) in enumerate(anim_output.items()):
                ax_3d = fig.add_subplot(3,
                                        len(anim_output) * VIEW_NUM,
                                        (view_i) * len(anim_output) + index +
                                        1,
                                        projection='3d')
                ax_3d.view_init(elev=view_camera[0], azim=view_camera[1])

                ax_3d.set_xticklabels([])
                ax_3d.set_yticklabels([])
                ax_3d.set_zticklabels([])

                ax_3d.dist = SKELETON_AX_DIST
                ax_3d.set_title(title, fontsize=3*SUB_FIG_SIZE)

                plot_vertics(ax_3d, data[frame_i, :, :])

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
        ax_acc.set_ylabel('Mean Acceleration Error (mm/s2)', fontsize=3*SUB_FIG_SIZE)
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
        ax_mpjpe.set_ylabel('Mean Position Error (mm)', fontsize=3*SUB_FIG_SIZE)
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


# you can also use pyrender to visualize the vertices more clearly. However, it is quite hard to intall
# it properly in Linux, so we recommend you use the function above. Users who run this code in Windows 
# can try functions below. You might need to modify some lines of code to fit different datasets

from lib.utils.render import Renderer

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

FONT_HEIGHT = 200
FONT_SIZE = 95
FRAME_RATE = 30
COLOR_GRAY = [230, 230, 230]
FONT_ORI = "(a) Video"
FONT_GT = "(b) Ground Truth"
FONT_IN = "(c) Estimator"
FONT_DECIWATCH = "(d) Ours"


def visualize_smpl_detailed(
                        vis_output_video_path,
                        vis_output_video_name,
                        smpl_neural,
                        in_poses,
                        gt_poses,
                        smoothnet_poses,
                        start_frame,
                        end_frame):
    print("Visualizing the result ...")

    with torch.no_grad():
        in_smpl = smpl_neural(
            body_pose=torch.from_numpy(in_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(in_poses[:, 0:3]).float())

        gt_smpl = smpl_neural(
            body_pose=torch.from_numpy(gt_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(gt_poses[:, 0:3]).float())

        smoothnet_smpl = smpl_neural(
            body_pose=torch.from_numpy(smoothnet_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(smoothnet_poses[:, 0:3]).float())

    imgsize_h, imgsize_w = 1000,1000

    videoWriter = cv2.VideoWriter(
        os.path.join(vis_output_video_path, vis_output_video_name),
        cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE,
        (imgsize_w * 3, imgsize_h + FONT_HEIGHT))

    if not os.path.exists(vis_output_video_path):
        os.makedirs(vis_output_video_path)

    for frame_i in trange(max(0, start_frame), min(in_smpl.joints.shape[0],
                                                   end_frame)):
        gt_image = np.zeros((imgsize_h, imgsize_w, 3))
        in_image = np.zeros((imgsize_h, imgsize_w, 3))
        smoothnet_image = np.zeros((imgsize_h, imgsize_w, 3))
        render = Renderer(smpl_neural.faces,
                          resolution=(imgsize_w, imgsize_h),
                          orig_img=True,
                          wireframe=False)

        gt_rendered_img = render.render(
            gt_image,
            gt_smpl.vertices.numpy()[frame_i],
            [imgsize_h * 0.001 * 0.6, imgsize_w * 0.001 * 0.6, 0.1, 0.2],
            color=np.array(COLOR_GRAY) / 255)
        in_rendered_img = render.render(
            in_image,
            in_smpl.vertices.numpy()[frame_i],
            [imgsize_h * 0.001 * 0.6, imgsize_w * 0.001 * 0.6, 0.1, 0.2],
            color=np.array(COLOR_GRAY) / 255)
        smoothnet_rendered_img = render.render(
            smoothnet_image,
            smoothnet_smpl.vertices.numpy()[frame_i],
            [imgsize_h * 0.001 * 0.6, imgsize_w * 0.001 * 0.6, 0.1, 0.2],
            color=np.array(COLOR_GRAY) / 255)

        gt_rendered_img = np.concatenate(
            (gt_rendered_img, np.zeros((FONT_HEIGHT, imgsize_w, 3))), axis=0)
        in_rendered_img = np.concatenate(
            (in_rendered_img, np.zeros((FONT_HEIGHT, imgsize_w, 3))), axis=0)
        smoothnet_rendered_img = np.concatenate(
            (smoothnet_rendered_img, np.zeros((FONT_HEIGHT, imgsize_w, 3))),
            axis=0)

        # font = ImageFont.truetype('simhei',size=FONT_SIZE)
        font = ImageFont.truetype('DejaVuSansCondensed-Bold', size=FONT_SIZE)

        img_gt = Image.fromarray(gt_rendered_img.astype(np.uint8))
        img_in = Image.fromarray(in_rendered_img.astype(np.uint8))
        img_smoothnet = Image.fromarray(smoothnet_rendered_img.astype(
            np.uint8))

        draw_gt = ImageDraw.Draw(img_gt)
        draw_gt.text(((imgsize_w - len(FONT_GT) * FONT_SIZE / 2) / 2,
                      imgsize_h + FONT_HEIGHT / 4),
                     FONT_GT, (255, 255, 255),
                     font=font)
        gt_rendered_img = np.array(img_gt)

        draw_in = ImageDraw.Draw(img_in)
        draw_in.text(((imgsize_w - len(FONT_IN) * FONT_SIZE / 2) / 2,
                      imgsize_h + FONT_HEIGHT / 4),
                     FONT_IN, (255, 255, 255),
                     font=font)
        in_rendered_img = np.array(img_in)

        draw_detected = ImageDraw.Draw(img_smoothnet)
        draw_detected.text(
            ((imgsize_w - len(FONT_DECIWATCH) * FONT_SIZE / 2) / 2,
             imgsize_h + FONT_HEIGHT / 4),
            FONT_DECIWATCH, (255, 255, 255),
            font=font)
        detected_rendered_img = np.array(img_smoothnet)
        output_img = np.concatenate(
            (gt_rendered_img, in_rendered_img,
             detected_rendered_img),
            axis=1)

        videoWriter.write(output_img)

    videoWriter.release()
    print(f"Finish! The video is stored in "+os.path.join(vis_output_video_path, vis_output_video_name))

