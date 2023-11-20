import cv2
from hutils.fileio import readEXR
from hutils.visualization import save_normal_no_margin, plt_error_map_cv2, save_transparent_img
import os
import numpy as np


def world2obj(normal_map):
    N_obj =  normal_map.copy()
    N_obj[:,:,2] *= -1
    N_obj[:, :, 0] *= -1
    return N_obj

data_dir = r'G:\Dropbox\paper\NeurIPS23\DiLiGenT_Data\PS_SSS\10_armadillo\10_armadillo.obj_pose_0_material_0'

# load normal
N_path = os.path.join(data_dir, 'normals_alt.exr')
Normal = readEXR(N_path)
Normal = Normal / np.linalg.norm(Normal, axis = -1, keepdims=True)

# load depth
depth_path = os.path.join(data_dir, 'depth.exr')
depth = readEXR(depth_path)
depth = depth[:,:,2]
mask = depth < 1e5


import cv2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.erode(mask.astype(np.uint8), kernel).astype(np.bool)
Normal[~mask] = 0


N_obj = world2obj(Normal)
N_show = N_obj/2 + 0.5
N_show[~mask] = 0

save_normal_no_margin(N_obj, mask, N_path[:-3] + 'png', white=True)
save_transparent_img(N_path[:-3] + 'png', mask, output_img_path = N_path[:-4] + '_t.png')