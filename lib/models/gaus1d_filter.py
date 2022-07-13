import warnings

import numpy as np
import scipy.signal as signal
import torch
from scipy.ndimage.filters import gaussian_filter1d

class GAUS1DFilter:
    """
    Args:
        x (np.ndarray): input pose
        window_size (int, optional): for median filters (must be odd).
        sigma (float, optional): Sigma for gaussian smoothing.

    Returns:
        np.ndarray: Smoothed poses
    """

    def __init__(self, cfg):
        super(GAUS1DFilter, self).__init__()

        self.window_size = cfg.EVALUATE.TRADITION_GAUS1D.WINDOW_SIZE
        self.sigma = cfg.EVALUATE.TRADITION_GAUS1D.SIGMA

    def __call__(self, x=None):
        if self.window_size % 2 == 0:
            window_size = self.window_size - 1
        else:
            window_size = self.window_size
        if window_size > x.shape[0]:
            window_size = x.shape[0]
        if len(x.shape) != 3:
            warnings.warn('x should be a tensor or numpy of [T*M,K,C]')
        assert len(x.shape) == 3
        x_type = x
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.cpu().numpy()
            else:
                x = x.numpy()
        
        T,K,C=x.shape
        x=x.reshape(T,K*C)

        smooth_poses=np.empty((0,K*C))
        for i in range(T//window_size):
            smooth_poses=np.concatenate((smooth_poses,gaussian_filter1d(input=x[i*window_size:(i+1)*window_size,:],
                                        sigma=self.sigma,
                                        axis=0)),0)
        smooth_poses=np.concatenate((smooth_poses,x[(T//window_size)*window_size:,:]),0)
        smooth_poses=smooth_poses.reshape(T,K,C)
            
        if isinstance(x_type, torch.Tensor):
            # we also return tensor by default
            if x_type.is_cuda:
                smooth_poses = torch.from_numpy(smooth_poses).cuda()
            else:
                smooth_poses = torch.from_numpy(smooth_poses)

        return smooth_poses