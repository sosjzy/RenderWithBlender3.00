import numpy as np
from  matplotlib import pyplot as plt
import os
from hutils.fileio import readEXR, createDir, readNormal16bitRGB
import glob
from hutils.visualization import save_normal_no_margin, plt_error_map_cv2, save_transparent_img
import cv2
import configparser




def world2obj(normal_map):
    N_obj =  normal_map.copy()
    N_obj[:,:,2] *= -1
    N_obj[:, :, 0] *= -1
    return N_obj

def process_blender_rendering_data(data_dir, light_path, light_selection):

    assert os.path.exists(data_dir)
    print('process {}...'.format(data_dir))


    # load light
    LED_locs = np.load(light_path)
    numLights = len(LED_locs)
    numCombinations, numSelection = light_selection.shape


    # load normal
    N_path = os.path.join(data_dir, 'render_para/normal_obj.npy')
    N_obj = np.load(N_path)
    N_obj = N_obj / np.linalg.norm(N_obj, axis = -1, keepdims=True)
    N_world = world2obj(N_obj)
    np.save(os.path.join(data_dir, 'render_para/normal_world.npy'), N_world)
    # load depth

    mask = np.load(os.path.join(data_dir, 'render_para/mask.npy'))
    N_obj[~mask] = 0

    N_show = N_obj/2 + 0.5
    N_show[~mask] = 0

    save_normal_no_margin(N_obj, mask, N_path[:-3] + 'png', white=True)
    save_transparent_img(N_path[:-3] + 'png', mask, output_img_path = N_path[:-4] + '_t.png')

    img_dir = os.path.join(data_dir, 'render_img')
    img_set = np.load(os.path.join(img_dir, 'imgs_diligent_gray.npy'))

    if numSelection == 3:
        save_dir = os.path.join(img_dir, 'PS3')
    else:
        save_dir = os.path.join(img_dir, 'PS2')
    createDir(save_dir)

    for i in range(numCombinations):
        save_dir_i = os.path.join(save_dir, '{}'.format(i))
        createDir(save_dir_i)
        LED_locs_i = LED_locs[light_selection[i]]
        img_set_i = img_set[light_selection[i]]
        img_set_hand_i = img_set[light_selection[i]]

        np.save(os.path.join(save_dir_i, 'light_dir_obj.npy'), LED_locs_i)
        np.savetxt(os.path.join(save_dir_i, 'light_dir_obj.txt'), LED_locs_i)
        np.save(os.path.join(save_dir_i, 'imgs_blender.npy'), img_set_i)
        np.save(os.path.join(save_dir_i, 'imgs_hand.npy'), img_set_hand_i)

        max_value = img_set_i.max() * 0.1
        for j in range(numSelection):
            img_j = img_set_i[j] / max_value * 255
            cv2.imwrite(os.path.join(save_dir_i, 'img_{}.png'.format(j)), img_j.astype('int'))



if __name__ == '__main__':

    base_data_dir = r'J:\Project\DeepPS3\data\Real'
    shape_set = np.genfromtxt(os.path.join(base_data_dir, 'shape_list_name.csv'), delimiter=',', dtype=np.str)
    # shape_set = ['ballPNG']


    light_dir_path = os.path.join(base_data_dir, 'L_dir_obj.npy')

    numCombinations = 20
    numSelection = 3
    numLights = 96

    light_selection_path = os.path.join(base_data_dir, 'light_selections.csv')
    if not os.path.exists(light_selection_path):
        light_selections = np.zeros([numCombinations, numSelection])
        for i in range(numCombinations):
            light_selections[i] = np.random.permutation(range(numLights))[:numSelection]
        np.savetxt(light_selection_path, light_selections, delimiter=',', fmt='%d')
    else:
        light_selections = np.loadtxt(light_selection_path, delimiter=',')


    for numSelection in [3]:
        if light_selections.shape[1] == 3 and numSelection == 2:
            light_selections = light_selections[:, :numSelection]

        light_selections = light_selections.astype('int')
        for shape_name in shape_set:
            data_dir = os.path.join(base_data_dir, shape_name)
            print('process ', data_dir)
            process_blender_rendering_data(data_dir, light_dir_path, light_selections)