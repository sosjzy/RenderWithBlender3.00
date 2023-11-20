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

    # load image
    config_path = os.path.join(data_dir, 'save.ini')
    ini_config = configparser.ConfigParser()
    ini_config.optionxform = str
    ini_config.read(config_path)
    camera_model = ini_config['camera']['model']


    # load light
    LED_locs = np.load(light_path)
    numLights = len(LED_locs)
    numCombinations, numSelection = light_selection.shape


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


    row, column = mask.shape
    yy, xx = np.mgrid[:row, :column]
    yy = np.flip(yy, axis=0)
    xx = np.flip(xx, axis=1)
    pixel_coords = np.stack([xx, yy], axis=-1).astype(np.float32)  ## -yy, xx
    pixel_coords[:, :, 0] = pixel_coords[:, :, 0] / (row - 1)  # xx
    pixel_coords[:, :, 1] = pixel_coords[:, :, 1] / (column - 1)
    pixel_coords -= 0.5
    pixel_coords *= 2.


    if camera_model == 'orthographic':
        point_set_world = np.concatenate([pixel_coords, depth[:, :, np.newaxis]], axis=2)
    else:
        focal_len = float(ini_config['camera']['focal_length'])
        sensor_h = float(ini_config['camera']['sensor_height'])
        sensor_w = float(ini_config['camera']['sensor_width'])

        sensor_coord = np.zeros_like(pixel_coords)
        sensor_coord[:, :, 0] = pixel_coords[:, :, 0] * (sensor_w / 2)  # * depth / focal_len
        sensor_coord[:, :, 1] = pixel_coords[:, :, 1] * (sensor_h / 2)  # * depth / focal_len

        distance_0 = np.sqrt(focal_len ** 2 + sensor_coord[:, :, 0] ** 2 + sensor_coord[:, :, 1] ** 2)

        depth_true = focal_len * depth / distance_0
        pixel_coords[:, :, 0] = pixel_coords[:, :, 0] * (sensor_w / 2) * depth_true / focal_len
        pixel_coords[:, :, 1] = pixel_coords[:, :, 1] * (sensor_h / 2) * depth_true / focal_len
        point_set_world = np.concatenate([pixel_coords, depth_true[:, :, np.newaxis]], axis=2)


    N_obj = world2obj(Normal)
    N_show = N_obj/2 + 0.5
    N_show[~mask] = 0

    save_normal_no_margin(N_obj, mask, N_path[:-3] + 'png', white=True)
    save_transparent_img(N_path[:-3] + 'png', mask, output_img_path = N_path[:-4] + '_t.png')

    # load image
    img_filenames_path = os.path.join(data_dir, 'filenames.ini')
    ini = configparser.ConfigParser()
    ini.optionxform = str
    ini.read(img_filenames_path)

    img_set = []
    img_set_hand_render = []
    for i in range(1, numLights+1):
        img_path = os.path.join(data_dir, ini['filenames']['{:>03d}'.format(i)])
        img = readEXR(img_path)
        img[~mask] = 0
        im_show = img #/ img.max()
        im_gamma_correct = np.clip(np.power(im_show, 0.45), 0, 1)
        cv2.imwrite(img_path[:-3]+'png', im_gamma_correct[:,:,::-1]*255)
        save_transparent_img(img_path[:-3]+'png', mask)
        img_set.append(img)

        light_dir_i = LED_locs[i-1]
        img_render_flat = N_obj[mask] @  light_dir_i
        img_render = np.zeros_like(img)
        img_render[mask] = img_render_flat[:, np.newaxis]
        img_render[img_render < 0] = 0
        img_set_hand_render.append(img_render)

    img_set = np.array(img_set)
    img_set_hand_render = np.array(img_set_hand_render)


    para_dir = os.path.join(data_dir, 'render_para')
    img_dir = os.path.join(data_dir, 'render_img')
    createDir(img_dir)
    createDir(para_dir)
    np.save(os.path.join(para_dir, 'normal_obj.npy'), N_obj)
    np.save(os.path.join(para_dir, 'mask.npy'), mask)
    cv2.imwrite(os.path.join(para_dir, 'mask.png'), mask.astype(np.uint8) * 255)
    np.save(os.path.join(para_dir, 'depth.npy'), point_set_world[:, :, 2])
    np.save(os.path.join(para_dir, 'point_set_world.npy'), point_set_world)
    np.save(os.path.join(para_dir, 'normal_world.npy'), Normal)
    np.save(os.path.join(img_dir, 'imgs_blender_all.npy'), img_set)
    np.save(os.path.join(img_dir, 'imgs_hand_all.npy'), img_set_hand_render)
    np.save(os.path.join(para_dir, 'light_dir_obj_all.npy'), LED_locs)
    np.save(os.path.join(para_dir, 'light_selections.npy'), light_selection)

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
        img_set_hand_i = img_set_hand_render[light_selection[i]]

        np.save(os.path.join(save_dir_i, 'light_dir_obj.npy'), LED_locs_i)
        np.savetxt(os.path.join(save_dir_i, 'light_dir_obj.txt'), LED_locs_i)
        np.save(os.path.join(save_dir_i, 'imgs_blender.npy'), img_set_i)
        np.save(os.path.join(save_dir_i, 'imgs_hand.npy'), img_set_hand_i)

        max_value = img_set_i.max()
        for j in range(numSelection):
            img_j = img_set_i[j] / max_value * 255
            cv2.imwrite(os.path.join(save_dir_i, 'img_{}.png'.format(j)), img_j.astype('int'))



if __name__ == '__main__':

    para_folder = '../supp'
    shape_set = np.genfromtxt(os.path.join(para_folder, 'shape_list_name.csv'), delimiter=',', dtype=np.str)
    # shape_set = ['Sphere']

    base_data_dir = r'J:\Project\DeepPS3\data\synthetic'
    light_dir_path = os.path.join(para_folder, 'light_100.npy')

    numCombinations = 20
    numSelection = 3
    numLights = 100

    light_selection_path = os.path.join(para_folder, 'light_selections.csv')
    if not os.path.exists(light_selection_path):
        light_selections = np.zeros([numCombinations, numSelection])
        for i in range(numCombinations):
            light_selections[i] = np.random.permutation(range(numLights))[:numSelection]
        np.savetxt(light_selection_path, light_selections, delimiter=',', fmt='%d')
    else:
        light_selections = np.loadtxt(light_selection_path, delimiter=',')


    for numSelection in [2, 3]:
        if light_selections.shape[1] == 3 and numSelection == 2:
            light_selections = light_selections[:, :numSelection]

        light_selections = light_selections.astype('int')
        for shape_name in shape_set:
            data_dir = os.path.join(base_data_dir, shape_name, 'distant_light_100')
            print('process ', data_dir)
            process_blender_rendering_data(data_dir, light_dir_path, light_selections)