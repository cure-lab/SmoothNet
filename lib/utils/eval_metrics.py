import imp
import torch
import numpy as np
from lib.models.smpl import SMPL
from lib.utils.geometry_utils import *
from lib.utils.utils import slide_window_to_sequence
from lib.models.gaus1d_filter import GAUS1DFilter
from lib.models.oneeuro_filter import ONEEUROFilter
from lib.models.savgol_filer import SAVGOLFilter

def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    S1 = S1.to(torch.float32)
    S2 = S2.to(torch.float32)

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = torch.norm(accel_pred - accel_gt, dim=2)

    if vis is None:
        new_vis = torch.ones(len(normed), dtype=bool)
    else:
        invis = torch.logical_not(vis)
        invis1 = torch.roll(invis, -1)
        invis2 = torch.roll(invis, -2)
        new_invis = torch.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = torch.logical_not(new_invis)

    acc=torch.mean(normed[new_vis], axis=1)

    return acc[~acc.isnan()]


def calculate_mpjpe(predicted, gt):
    mpjpe = torch.sqrt(((predicted - gt)**2).sum(dim=-1))
    mpjpe = mpjpe.mean(dim=-1)
    return mpjpe[~mpjpe.isnan()]


def calculate_pampjpe(predicted, gt):
    S1_hat = batch_compute_similarity_transform_torch(predicted, gt)
    # per-frame accuracy after procrustes alignment
    mpjpe_pa = torch.sqrt(((S1_hat - gt)**2).sum(dim=-1))
    mpjpe_pa = mpjpe_pa.mean(dim=-1)
    return mpjpe_pa[~mpjpe_pa.isnan()]


def calculate_accel_error(predicted, gt):
    accel_err = compute_error_accel(joints_pred=predicted, joints_gt=gt)

    accel_err=torch.concat((torch.tensor([0]).to(accel_err.device),accel_err,torch.tensor([0]).to(accel_err.device)))
    return accel_err


def calculate_jhmdb_PCK(predicted, gt, bbox, imgshape, thresh):

    # 0: neck    1:belly   2: face
    # 3: right shoulder  4: left shoulder
    # 5: right hip       6: left hip
    # 7: right elbow     8: left elbow
    # 9: right knee      10: left knee
    # 11: right wrist    12: left wrist
    # 13: right ankle    14: left ankle

    orderJHMDB = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    N = predicted.shape[0]

    HitPoint = torch.zeros((len(orderJHMDB))).to(gt.device)
    Point_to_use = torch.ones((len(orderJHMDB))).to(gt.device)

    
    test_gt = gt.reshape(-1, len(orderJHMDB), 2) * imgshape[0, [1,0]]
    test_out = predicted.reshape(-1, len(orderJHMDB), 2) * imgshape[0, [1,0]]

    seqError = torch.zeros(N, len(orderJHMDB)).to(gt.device)
    seqThresh = torch.zeros(N, len(orderJHMDB)).to(gt.device)

    bodysize = torch.max(bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1])

    for frame in range(0, N):
        gt_frame = test_gt[frame, :]
        pred_frame = test_out[frame, :]

        error_dis = torch.norm(gt_frame - pred_frame,
                                   p=2,
                                   dim=1,
                                   keepdim=False)

        seqError[frame] = error_dis
        seqThresh[frame] = (bodysize[frame] * thresh) * torch.ones(
                len(orderJHMDB)).to(gt.device)

    less_than_thresh = seqError <= seqThresh
    HitPoint = torch.sum(less_than_thresh, dim=0)

    finalPCK = torch.divide(HitPoint, N)

    return finalPCK


def evaluate_smoothnet_2D(model,
                          test_dataloader,
                          device,
                          cfg,dataset_index,dataset,
                          show_detail=True):
    keypoint_number = test_dataloader.dataset.input_dimension//2

    if dataset =="h36m":
        input_mpjpe = torch.empty((0)).to(device)
        input_pampjpe = torch.empty((0)).to(device)
        input_accel = torch.empty((0)).to(device)

        denoise_mpjpe = torch.empty((0)).to(device)
        denoise_pampjpe = torch.empty((0)).to(device)
        denoise_accel = torch.empty((0)).to(device)

        if cfg.EVALUATE.TRADITION !="":
            filter_mpjpe = torch.empty((0)).to(device)
            filter_pampjpe = torch.empty((0)).to(device)
            filter_accel = torch.empty((0)).to(device)

            filter=eval(cfg.EVALUATE.TRADITION.upper()+"Filter()")


        for i, data in enumerate(test_dataloader):
            data_pred = data["pred"].to(device).squeeze(0)
            data_gt = data["gt"].to(device).squeeze(0)

            with torch.no_grad():
                data_pred=data_pred.permute(0,2,1)
                denoised_pos = model(
                    data_pred)
                data_pred=data_pred.permute(0,2,1)
                denoised_pos=denoised_pos.permute(0,2,1)


            denoised_pos = slide_window_to_sequence(denoised_pos,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)
            data_pred = slide_window_to_sequence(data_pred,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)
            data_gt = slide_window_to_sequence(data_gt,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)

            frame_num=denoised_pos.shape[0]

            H36M_IMG_SHAPE=1000

            if cfg.EVALUATE.TRADITION !="":
                filter_pos=filter(data_pred.reshape(frame_num, -1, 2))

            denoised_pos=denoised_pos.reshape(frame_num, -1, 2)*(H36M_IMG_SHAPE/2)+(H36M_IMG_SHAPE/2)
            data_pred=data_pred.reshape(frame_num, -1, 2)*(H36M_IMG_SHAPE/2)+(H36M_IMG_SHAPE/2)
            data_gt=data_gt.reshape(frame_num, -1, 2)*(H36M_IMG_SHAPE/2)+(H36M_IMG_SHAPE/2)
            filter_pos=filter_pos.reshape(frame_num, -1, 2)*(H36M_IMG_SHAPE/2)+(H36M_IMG_SHAPE/2)

            input_mpjpe = torch.cat(
                (input_mpjpe, calculate_mpjpe(data_pred, data_gt)), dim=0)
            input_pampjpe = torch.cat(
                (input_pampjpe, calculate_pampjpe(data_pred, data_gt)), dim=0)
            input_accel = torch.cat(
                (input_accel, calculate_accel_error(data_pred, data_gt)), dim=0)


            denoise_mpjpe = torch.cat(
                (denoise_mpjpe, calculate_mpjpe(denoised_pos, data_gt)), dim=0)
            denoise_pampjpe = torch.cat(
                (denoise_pampjpe, calculate_pampjpe(denoised_pos, data_gt)), dim=0)
            denoise_accel = torch.cat(
                (denoise_accel, calculate_accel_error(denoised_pos, data_gt)), dim=0)

            if cfg.EVALUATE.TRADITION !="":
                filter_mpjpe = torch.cat(
                    (filter_mpjpe, calculate_mpjpe(filter_pos, data_gt)), dim=0)
                filter_pampjpe = torch.cat(
                    (filter_pampjpe, calculate_pampjpe(filter_pos, data_gt)), dim=0)
                filter_accel = torch.cat(
                    (filter_accel, calculate_accel_error(filter_pos, data_gt)), dim=0)


        _,mpjpe_top10_index=torch.sort(input_mpjpe)
        mpjpe_top10_index=mpjpe_top10_index[int(len(mpjpe_top10_index)*0.9):]

        input_mpjpe_top10=input_mpjpe[mpjpe_top10_index]
        input_accel_top10=input_accel[mpjpe_top10_index]
        denoise_mpjpe_top10=denoise_mpjpe[mpjpe_top10_index]
        denoise_accel_top10=denoise_accel[mpjpe_top10_index]
        
        if cfg.EVALUATE.TRADITION !="":
            eval_dict = {
                    "input_mpjpe": input_mpjpe.mean() ,
                    "output_mpjpe": denoise_mpjpe.mean() ,
                    "improvement_mpjpe": denoise_mpjpe.mean()  - input_mpjpe.mean() ,
                    "filter_mpjpe": filter_mpjpe.mean()  ,
                    "input_pampjpe": input_pampjpe.mean() ,
                    "output_pampjpe": denoise_pampjpe.mean() ,
                    "improvement_pampjpe":
                    denoise_pampjpe.mean()- input_pampjpe.mean(),
                    "filter_pampjpe": filter_pampjpe.mean()  ,
                    "input_accel": input_accel.mean(),
                    "output_accel": denoise_accel.mean(),
                    "improvement_accel": denoise_accel.mean() - input_accel.mean() ,
                    "filter_accel": filter_accel.mean()  ,
                }
        else:
            eval_dict = {
                    "input_mpjpe": input_mpjpe.mean() ,
                    "output_mpjpe": denoise_mpjpe.mean() ,
                    "improvement_mpjpe": denoise_mpjpe.mean()  - input_mpjpe.mean() ,
                    "input_pampjpe": input_pampjpe.mean() ,
                    "output_pampjpe": denoise_pampjpe.mean() ,
                    "improvement_pampjpe":
                    denoise_pampjpe.mean()- input_pampjpe.mean(),
                    "input_accel": input_accel.mean(),
                    "output_accel": denoise_accel.mean(),
                    "improvement_accel": denoise_accel.mean() - input_accel.mean() ,
                }
        
        return eval_dict

    # evaluate on jhmdb
    elif dataset == "jhmdb":

        # original pck
        input_pck_005 = torch.empty((keypoint_number, 0)).to(device)
        input_pck_01 = torch.empty((keypoint_number, 0)).to(device)
        input_pck_02 = torch.empty((keypoint_number, 0)).to(device)

        # deciwatch denoise pck
        denoise_pck_005 = torch.empty((keypoint_number, 0)).to(device)
        denoise_pck_01 = torch.empty((keypoint_number, 0)).to(device)
        denoise_pck_02 = torch.empty((keypoint_number, 0)).to(device)

        if cfg.EVALUATE.TRADITION !="":
            filter_pck_005 = torch.empty((keypoint_number, 0)).to(device)
            filter_pck_01 = torch.empty((keypoint_number, 0)).to(device)
            filter_pck_02 = torch.empty((keypoint_number, 0)).to(device)

            filter=eval(cfg.EVALUATE.TRADITION.upper()+"Filter()")

        # calculate each sequence error
        for i, data in enumerate(test_dataloader):
            # get data
            data_pred = data["pred"].to(device).squeeze(0)
            data_gt = data["gt"].to(device).squeeze(0)
            data_bbox = data["bbox"].to(device).squeeze(0)
            data_imgshape = data["imgshape"].to(device)
            
            with torch.no_grad():
                data_pred=data_pred.permute(0,2,1)
                denoised_pos = model(data_pred)
                data_pred=data_pred.permute(0,2,1)
                denoised_pos=denoised_pos.permute(0,2,1)

            # slide window to sequence
            denoised_pos = slide_window_to_sequence(denoised_pos,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE).reshape(-1, keypoint_number, 2)
            data_pred = slide_window_to_sequence(data_pred,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE).reshape(-1, keypoint_number, 2)
            data_gt = slide_window_to_sequence(data_gt,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE).reshape(-1, keypoint_number, 2)
            data_bbox = slide_window_to_sequence(data_bbox,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE).type(torch.int32)
            
            if cfg.EVALUATE.TRADITION !="":
                filter_pos=filter(data_pred)

            # input pck
            input_pck_005 = torch.cat(
                (input_pck_005,
                 calculate_jhmdb_PCK(data_pred, data_gt, data_bbox,
                                     data_imgshape, 0.05).reshape(
                                         keypoint_number, 1)),
                dim=1)

            input_pck_01 = torch.cat(
                (input_pck_01,
                 calculate_jhmdb_PCK(data_pred, data_gt, data_bbox,
                                     data_imgshape, 0.1).reshape(
                                         keypoint_number, 1)),
                dim=1)

            input_pck_02 = torch.cat(
                (input_pck_02,
                 calculate_jhmdb_PCK(data_pred, data_gt, data_bbox,
                                     data_imgshape, 0.2).reshape(
                                         keypoint_number, 1)),
                dim=1)
            
            # deciwatch denoise pck
            denoise_pck_005 = torch.cat(
                (denoise_pck_005,
                 calculate_jhmdb_PCK(denoised_pos, data_gt, data_bbox,
                                     data_imgshape, 0.05).reshape(
                                         keypoint_number, 1)),
                dim=1)

            denoise_pck_01 = torch.cat(
                (denoise_pck_01,
                 calculate_jhmdb_PCK(denoised_pos, data_gt, data_bbox,
                                     data_imgshape, 0.1).reshape(
                                         keypoint_number, 1)),
                dim=1)

            denoise_pck_02 = torch.cat(
                (denoise_pck_02,
                 calculate_jhmdb_PCK(denoised_pos, data_gt, data_bbox,
                                     data_imgshape, 0.2).reshape(
                                         keypoint_number, 1)),
                dim=1)

            if cfg.EVALUATE.TRADITION !="":
                filter_pck_005 = torch.cat(
                (filter_pck_005,
                 calculate_jhmdb_PCK(filter_pos, data_gt, data_bbox,
                                     data_imgshape, 0.05).reshape(
                                         keypoint_number, 1)),
                dim=1)

                filter_pck_01 = torch.cat(
                    (filter_pck_01,
                    calculate_jhmdb_PCK(filter_pos, data_gt, data_bbox,
                                        data_imgshape, 0.1).reshape(
                                            keypoint_number, 1)),
                    dim=1)

                filter_pck_02 = torch.cat(
                    (filter_pck_02,
                    calculate_jhmdb_PCK(filter_pos, data_gt, data_bbox,
                                        data_imgshape, 0.2).reshape(
                                            keypoint_number, 1)),
                    dim=1)


        def print_detail(pck, show_detail=None):
            if show_detail is not None:
                print(show_detail)
                print(
                    'Head,     Shoulder, Elbow,    Wrist,    Hip,      Knee,     Ankle,    Mean'
                )
                print(
                    '{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}'
                    .format(pck[0, :].mean(),
                            0.5 * (pck[1, :].mean() + pck[2, :].mean()),
                            0.5 * (pck[3, :].mean() + pck[4, :].mean()),
                            0.5 * (pck[5, :].mean() + pck[6, :].mean()),
                            0.5 * (pck[7, :].mean() + pck[8, :].mean()),
                            0.5 * (pck[9, :].mean() + pck[10, :].mean()),
                            0.5 * (pck[11, :].mean() + pck[12, :].mean()),
                            pck.mean()))

        if show_detail:
                print_detail(input_pck_005, "INPUT PCK 0.05")
                print_detail(denoise_pck_005, "OUTPUT PCK 0.05")
                print_detail(input_pck_01, "INPUT PCK 0.1")
                print_detail(denoise_pck_01, "OUTPUT PCK 0.1")
                print_detail(input_pck_02, "INPUT PCK 0.2")
                print_detail(denoise_pck_02, "OUTPUT PCK 0.2")

        if cfg.EVALUATE.TRADITION !="":
                eval_dict = {
                    "input_pck_005": input_pck_005.mean(),
                    "output_pck_005": denoise_pck_005.mean(),
                    "improvement_pck_005": denoise_pck_005.mean() - input_pck_005.mean(),
                    "filter_pck_005":filter_pck_005.mean(),
                    "input_pck_01": input_pck_01.mean(),
                    "output_pck_01": denoise_pck_01.mean(),
                    "improvement_pck_01": denoise_pck_01.mean() - input_pck_01.mean(),
                    "filter_pck_01":filter_pck_01.mean(),
                    "input_pck_02": input_pck_02.mean(),
                    "output_pck_02": denoise_pck_02.mean(),
                    "improvement_pck_02": denoise_pck_02.mean() - input_pck_02.mean(),
                    "filter_pck_02":filter_pck_02.mean(),
                }
        else:
            eval_dict = {
                    "input_pck_005": input_pck_005.mean(),
                    "output_pck_005": denoise_pck_005.mean(),
                    "improvement_pck_005": denoise_pck_005.mean() - input_pck_005.mean(),
                    "input_pck_01": input_pck_01.mean(),
                    "output_pck_01": denoise_pck_01.mean(),
                    "improvement_pck_01": denoise_pck_01.mean() - input_pck_01.mean(),
                    "input_pck_02": input_pck_02.mean(),
                    "output_pck_02": denoise_pck_02.mean(),
                    "improvement_pck_02": denoise_pck_02.mean() - input_pck_02.mean(),
                }
            
        return eval_dict


def evaluate_smoothnet_3D(model, test_dataloader, device, dataset_name,estimator,cfg):
    keypoint_root=eval("cfg.DATASET.ROOT_"+dataset_name.upper()+"_"+estimator.upper()+"_3D")

    input_mpjpe = torch.empty((0)).to(device)
    input_pampjpe = torch.empty((0)).to(device)
    input_accel = torch.empty((0)).to(device)

    denoise_mpjpe = torch.empty((0)).to(device)
    denoise_pampjpe = torch.empty((0)).to(device)
    denoise_accel = torch.empty((0)).to(device)

    if cfg.EVALUATE.TRADITION !="":
        filter_mpjpe = torch.empty((0)).to(device)
        filter_pampjpe = torch.empty((0)).to(device)
        filter_accel = torch.empty((0)).to(device)

        filter=eval(cfg.EVALUATE.TRADITION.upper()+"Filter()")

    for i, data in enumerate(test_dataloader):
        data_pred = data["pred"].to(device).squeeze(0)
        data_gt = data["gt"].to(device).squeeze(0)

        with torch.no_grad():
            data_pred=data_pred.permute(0,2,1)
            denoised_pos = model(
                data_pred)
            data_pred=data_pred.permute(0,2,1)
            denoised_pos=denoised_pos.permute(0,2,1)


        denoised_pos = slide_window_to_sequence(denoised_pos,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)
        data_pred = slide_window_to_sequence(data_pred,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)
        data_gt = slide_window_to_sequence(data_gt,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)

        frame_num=denoised_pos.shape[0]

        denoised_pos=denoised_pos.reshape(frame_num, -1, 3)
        data_pred=data_pred.reshape(frame_num, -1, 3)
        data_gt=data_gt.reshape(frame_num, -1, 3)

        if cfg.EVALUATE.ROOT_RELATIVE:
            denoised_pos = denoised_pos - denoised_pos[:,
                                                          keypoint_root, :].mean(
                                                              axis=1).reshape(
                                                                  -1, 1, 3)
            data_pred = data_pred - data_pred[:, keypoint_root, :].mean(
                axis=1).reshape(-1, 1, 3)
            data_gt = data_gt - data_gt[:, keypoint_root, :].mean(
                axis=1).reshape(-1, 1, 3)


        if cfg.EVALUATE.TRADITION !="":
            filter_pos=filter(data_pred)
        
        input_mpjpe = torch.cat(
            (input_mpjpe, calculate_mpjpe(data_pred, data_gt)), dim=0)
        input_pampjpe = torch.cat(
            (input_pampjpe, calculate_pampjpe(data_pred, data_gt)), dim=0)
        input_accel = torch.cat(
            (input_accel, calculate_accel_error(data_pred, data_gt)), dim=0)


        denoise_mpjpe = torch.cat(
            (denoise_mpjpe, calculate_mpjpe(denoised_pos, data_gt)), dim=0)
        denoise_pampjpe = torch.cat(
            (denoise_pampjpe, calculate_pampjpe(denoised_pos, data_gt)), dim=0)
        denoise_accel = torch.cat(
            (denoise_accel, calculate_accel_error(denoised_pos, data_gt)), dim=0)

        if cfg.EVALUATE.TRADITION !="":
            filter_mpjpe = torch.cat(
                (filter_mpjpe, calculate_mpjpe(filter_pos, data_gt)), dim=0)
            filter_pampjpe = torch.cat(
                (filter_pampjpe, calculate_pampjpe(filter_pos, data_gt)), dim=0)
            filter_accel = torch.cat(
                (filter_accel, calculate_accel_error(filter_pos, data_gt)), dim=0)

    
    _,mpjpe_top_index=torch.sort(input_mpjpe)
    mpjpe_top10_index=mpjpe_top_index[int(len(mpjpe_top_index)*0.9):]

    input_mpjpe_top10=input_mpjpe[mpjpe_top10_index]
    input_accel_top10=input_accel[mpjpe_top10_index]
    denoise_mpjpe_top10=denoise_mpjpe[mpjpe_top10_index]
    denoise_accel_top10=denoise_accel[mpjpe_top10_index]

    m2mm = 1000


    if cfg.EVALUATE.TRADITION !="":
        eval_dict = {
                "input_mpjpe": input_mpjpe.mean() * m2mm,
                "output_mpjpe": denoise_mpjpe.mean() * m2mm,
                "improvement_mpjpe": denoise_mpjpe.mean() * m2mm - input_mpjpe.mean() * m2mm,
                "filter_mpjpe":filter_mpjpe.mean()*m2mm,
                "input_pampjpe": input_pampjpe.mean() * m2mm,
                "output_pampjpe": denoise_pampjpe.mean() * m2mm,
                "improvement_pampjpe":
                denoise_pampjpe.mean() * m2mm - input_pampjpe.mean() * m2mm,
                "filter_pampjpe":filter_pampjpe.mean()*m2mm,
                "input_accel": input_accel.mean() * m2mm,
                "output_accel": denoise_accel.mean() * m2mm,
                "improvement_accel": denoise_accel.mean() * m2mm - input_accel.mean() * m2mm,
                "filter_accel":filter_accel.mean()*m2mm
            }
    else:
         eval_dict = {
                "input_mpjpe": input_mpjpe.mean() * m2mm,
                "output_mpjpe": denoise_mpjpe.mean() * m2mm,
                "improvement_mpjpe": denoise_mpjpe.mean() * m2mm - input_mpjpe.mean() * m2mm,
                "input_pampjpe": input_pampjpe.mean() * m2mm,
                "output_pampjpe": denoise_pampjpe.mean() * m2mm,
                "improvement_pampjpe":
                denoise_pampjpe.mean() * m2mm - input_pampjpe.mean() * m2mm,
                "input_accel": input_accel.mean() * m2mm,
                "output_accel": denoise_accel.mean() * m2mm,
                "improvement_accel": denoise_accel.mean() * m2mm - input_accel.mean() * m2mm,
            }
        
        
    return eval_dict


def evaluate_smoothnet_smpl(model, test_dataloader, device,cfg,dataset):
    SMPL_TO_J14 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 38]
    keypoint_root = [2, 3]

    smpl = SMPL(model_path=cfg.SMPL_MODEL_DIR, gender="neutral",
                batch_size=1).to(device)

    input_mpjpe = torch.empty((0)).to(device)
    input_pampjpe = torch.empty((0)).to(device)
    input_accel = torch.empty((0)).to(device)
    input_mpvpe = torch.empty((0)).to(device)

    denoise_mpjpe = torch.empty((0)).to(device)
    denoise_pampjpe = torch.empty((0)).to(device)
    denoise_mpvpe = torch.empty((0)).to(device)
    denoise_accel = torch.empty((0)).to(device)

    if cfg.EVALUATE.TRADITION !="":
        filter_mpjpe = torch.empty((0)).to(device)
        filter_pampjpe = torch.empty((0)).to(device)
        filter_mpvpe = torch.empty((0)).to(device)
        filter_accel = torch.empty((0)).to(device)

        filter=eval(cfg.EVALUATE.TRADITION.upper()+"Filter()")


    for i, data in enumerate(test_dataloader):
        data_pred = data["pred"].to(device).squeeze(0)
        data_gt = data["gt"].to(device).squeeze(0)

        if cfg.TRAIN.USE_6D_SMPL:
            rotation_dimension = 6
        else:
            rotation_dimension = 3

        data_pred_pose = data_pred[:, :, :24 * rotation_dimension]
        data_gt_pose = data_gt[:, :, :24 * rotation_dimension]
        data_pred_shape = data_pred[:, :, 24 * rotation_dimension:]

        if dataset != "aist":
            data_gt_shape = data_gt[:, :, 24 * rotation_dimension:]
        else:
            data_gt_trans = data_gt[:, :, 24 * rotation_dimension:24 *
                                    rotation_dimension + 3]
            data_gt_scaling = data_gt[:, :, 24 * rotation_dimension + 3:]

        with torch.no_grad():
            data_pred_pose=data_pred_pose.permute(0,2,1)
            denoised_pos = model(data_pred_pose)
            data_pred_pose=data_pred_pose.permute(0,2,1)
            denoised_pos=denoised_pos.permute(0,2,1)

        denoised_pos=slide_window_to_sequence(denoised_pos,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)
        data_gt_pose=slide_window_to_sequence(data_gt_pose,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)
        data_pred_pose=slide_window_to_sequence(data_pred_pose,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)

        data_pred_shape=slide_window_to_sequence(data_pred_shape,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)
        if dataset != "aist":
            data_gt_shape=slide_window_to_sequence(data_gt_shape,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)
        else:
            data_gt_trans=slide_window_to_sequence(data_gt_trans,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)
            data_gt_scaling=slide_window_to_sequence(data_gt_scaling,cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE,cfg.MODEL.SLIDE_WINDOW_SIZE)

        if cfg.EVALUATE.TRADITION !="":
            filter_pos=filter(data_pred_pose.reshape(-1,24,3))
         
        
        if cfg.TRAIN.USE_6D_SMPL:
            denoised_pos = rot6D_to_axis(denoised_pos.reshape(
                -1, 6)).reshape(-1, 24 * 3)
            data_gt_pose = rot6D_to_axis(data_gt_pose.reshape(
                -1, 6)).reshape(-1, 24 * 3)
            data_pred_pose = rot6D_to_axis(
                data_pred_pose.reshape(-1, 6)).reshape(-1, 24 * 3)
            filter_pos = rot6D_to_axis(
                filter_pos.reshape(-1, 6)).reshape(-1, 24 * 3)
            

        with torch.no_grad():
            if dataset != "aist":
                gt_smpl_result = smpl.forward(
                    global_orient=data_gt_pose[:, 0:3].to(torch.float32),
                    body_pose=data_gt_pose[:, 3:].to(torch.float32),
                    betas=data_gt_shape.to(torch.float32),
                )
            else:
                gt_smpl_result = smpl.forward(
                    global_orient=data_gt_pose[:, 0:3].to(torch.float32),
                    body_pose=data_gt_pose[:, 3:].to(torch.float32),
                    # transl=data_gt_trans_j.to(torch.float32),
                    scaling=data_gt_scaling.to(torch.float32),
                )

            input_smpl_result = smpl.forward(
                global_orient=data_pred_pose[:, 0:3].to(torch.float32),
                body_pose=data_pred_pose[:, 3:].to(torch.float32),
                betas=data_pred_shape.to(torch.float32),
            )

            denoise_smpl_result = smpl.forward(
                global_orient=denoised_pos[:, 0:3].to(torch.float32),
                body_pose=denoised_pos[:, 3:].to(torch.float32),
                betas=data_pred_shape.to(torch.float32),
            )

            filter_smpl_result = smpl.forward(
                global_orient=filter_pos[:, 0:3].to(torch.float32),
                body_pose=filter_pos[:, 3:].to(torch.float32),
                betas=data_pred_shape.to(torch.float32),
            )

        input_smpl_result_joints = input_smpl_result.joints[:,
                                                            SMPL_TO_J14, :]
        gt_smpl_result_joints = gt_smpl_result.joints[:, SMPL_TO_J14, :]
        
        denoise_smpl_result_joints = denoise_smpl_result.joints[:,
                                                            SMPL_TO_J14, :]
        filter_smpl_result_joints = filter_smpl_result.joints[:,
                                                            SMPL_TO_J14, :]                                          
        
        if cfg.EVALUATE.ROOT_RELATIVE:
            input_smpl_result_joints = input_smpl_result_joints - input_smpl_result_joints[:, keypoint_root, :].mean(
                axis=1).reshape(-1, 1, 3)
            gt_smpl_result_joints = gt_smpl_result_joints - gt_smpl_result_joints[:, keypoint_root, :].mean(
                axis=1).reshape(-1, 1, 3)
            denoise_smpl_result_joints = denoise_smpl_result_joints - denoise_smpl_result_joints[:, keypoint_root, :].mean(
                axis=1).reshape(-1, 1, 3)
            filter_smpl_result_joints = filter_smpl_result_joints - filter_smpl_result_joints[:, keypoint_root, :].mean(
                axis=1).reshape(-1, 1, 3)

        input_mpjpe = torch.cat((input_mpjpe,
                                    calculate_mpjpe(input_smpl_result_joints,
                                                    gt_smpl_result_joints)),
                                dim=0)
        input_pampjpe = torch.cat(
            (input_pampjpe,
                calculate_pampjpe(input_smpl_result_joints,
                                gt_smpl_result_joints)),
            dim=0)
        input_accel = torch.cat(
            (input_accel,
                calculate_accel_error(input_smpl_result_joints,
                                    gt_smpl_result_joints)),
            dim=0)
        input_mpvpe = torch.cat(
            (input_mpvpe,
                calculate_mpjpe(input_smpl_result.vertices,
                                gt_smpl_result.vertices)),
            dim=0)

       
        denoise_mpjpe = torch.cat((denoise_mpjpe,
                                calculate_mpjpe(denoise_smpl_result_joints,
                                                gt_smpl_result_joints)),
                                dim=0)
        denoise_pampjpe = torch.cat(
            (denoise_pampjpe,
                calculate_pampjpe(denoise_smpl_result_joints,
                                gt_smpl_result_joints)),
            dim=0)
        denoise_mpvpe = torch.cat((denoise_mpvpe,
                                calculate_mpjpe(denoise_smpl_result.vertices,
                                                gt_smpl_result.vertices)),
                                dim=0)
        denoise_accel = torch.cat(
            (denoise_accel,
                calculate_accel_error(denoise_smpl_result_joints,
                                    gt_smpl_result_joints)),
            dim=0)

        if cfg.EVALUATE.TRADITION !="":
            filter_mpjpe = torch.cat((filter_mpjpe,
                                    calculate_mpjpe(filter_smpl_result_joints,
                                                    gt_smpl_result_joints)),
                                    dim=0)
            filter_pampjpe = torch.cat(
                (filter_pampjpe,
                    calculate_pampjpe(filter_smpl_result_joints,
                                    gt_smpl_result_joints)),
                dim=0)
            filter_mpvpe = torch.cat((filter_mpvpe,
                                    calculate_mpjpe(filter_smpl_result.vertices,
                                                    gt_smpl_result.vertices)),
                                    dim=0)
            filter_accel = torch.cat(
                (filter_accel,
                    calculate_accel_error(filter_smpl_result_joints,
                                        gt_smpl_result_joints)),
                dim=0)

    m2mm = 1000

    _,mpjpe_top10_index=torch.sort(input_mpjpe)
    mpjpe_top10_index=mpjpe_top10_index[int(len(mpjpe_top10_index)*0.9):]

    input_mpjpe_top10=input_mpjpe[mpjpe_top10_index]
    input_accel_top10=input_accel[mpjpe_top10_index]
    denoise_mpjpe_top10=denoise_mpjpe[mpjpe_top10_index]
    denoise_accel_top10=denoise_accel[mpjpe_top10_index]

    if cfg.EVALUATE.TRADITION !="":
        eval_dict = {
                "input_mpjpe": input_mpjpe.mean() * m2mm,
                "output_mpjpe": denoise_mpjpe.mean() * m2mm,
                "improvement_mpjpe": denoise_mpjpe.mean() * m2mm - input_mpjpe.mean() * m2mm,
                "filter_mpjpe":filter_mpjpe.mean()*m2mm,
                "input_pampjpe": input_pampjpe.mean() * m2mm,
                "output_pampjpe": denoise_pampjpe.mean() * m2mm,
                "improvement_pampjpe":
                denoise_pampjpe.mean() * m2mm - input_pampjpe.mean() * m2mm,
                "filter_pampjpe":filter_pampjpe.mean()*m2mm,
                "input_accel": input_accel.mean() * m2mm,
                "output_accel": denoise_accel.mean() * m2mm,
                "improvement_accel": denoise_accel.mean() * m2mm - input_accel.mean() * m2mm,
                "filter_accel":filter_accel.mean()*m2mm,
                "input_mpvpe": input_mpvpe.mean() * m2mm,
                "output_mpvpe": denoise_mpvpe.mean() * m2mm,
                "improvement_mpvpe": denoise_mpvpe.mean() * m2mm - input_mpvpe.mean() * m2mm,
                "filter_mpvpe":filter_mpvpe.mean()*m2mm,
            }
    else:
        eval_dict = {
                "input_mpjpe": input_mpjpe.mean() * m2mm,
                "output_mpjpe": denoise_mpjpe.mean() * m2mm,
                "improvement_mpjpe": denoise_mpjpe.mean() * m2mm - input_mpjpe.mean() * m2mm,
                "input_pampjpe": input_pampjpe.mean() * m2mm,
                "output_pampjpe": denoise_pampjpe.mean() * m2mm,
                "improvement_pampjpe":
                denoise_pampjpe.mean() * m2mm - input_pampjpe.mean() * m2mm,
                "input_accel": input_accel.mean() * m2mm,
                "output_accel": denoise_accel.mean() * m2mm,
                "improvement_accel": denoise_accel.mean() * m2mm - input_accel.mean() * m2mm,
                "input_mpvpe": input_mpvpe.mean() * m2mm,
                "output_mpvpe": denoise_mpvpe.mean() * m2mm,
                "improvement_mpvpe": denoise_mpvpe.mean() * m2mm - input_mpvpe.mean() * m2mm,
            }

    return eval_dict
