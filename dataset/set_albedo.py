#%%

import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from hutils.draw_3D import generate_mesh
import glob
import cv2
from hutils.visualization import create_gif, save_transparent_img


def set_albedo(data_dir, albedo_folder, light_str, choice = None):
    imgs = np.load(os.path.join(data_dir, 'render_img/{}/imgs_hand.npy'.format(light_str)))
    mask = np.load(os.path.join(data_dir, 'render_para/mask.npy'))

    textures_path = glob.glob(os.path.join(albedo_folder, '*.jpg'))
    albedo_path = os.path.join(data_dir, 'render_para/albedo.npy')
    # choose texture
    if not os.path.exists(albedo_path):

        print("The albedo is not assigned yet for {}".format(albedo_path))
        shading = imgs[0]

        h, w = mask.shape

        fig, axes = plt.subplots(4, int(len(textures_path) / 4) + 1)
        axes = axes.ravel()

        scale_set = np.zeros([len(textures_path), 3])
        albedo_set = []
        for i, tex_path in enumerate(textures_path):
            albedo = cv2.imread(tex_path)[:, :, ::-1]

            albedo = cv2.resize(albedo, (h, w)).transpose([1, 0, 2])

            albedo = cv2.GaussianBlur(albedo, (7, 7), 0)
            # albedo = albedo[:h, :w]
            img = shading * albedo
            scale_set[i] = 1.0 / np.max(img.reshape(-1, 3), axis=0)

            albedo_set.append(scale_set[i][np.newaxis, np.newaxis, :] * albedo)
            img = scale_set[i][np.newaxis, np.newaxis, :] * img
            axes[i].imshow(img)
            axes[i].set_title('{}: {}'.format(i, os.path.basename(tex_path)))
            axes[i].axis('off')

        plt.show()
        try:
            choice = int(input('Input texture label:'))
        except ValueError:
            print("Not a number")

        fig, axes = plt.subplots(1, 2)
        axes = axes.ravel()
        if choice != -1:
            chosen_albedo = albedo_set[choice]
            axes[0].imshow(chosen_albedo)
            axes[1].imshow(chosen_albedo * shading)
            plt.show()
        else:
            chosen_albedo = np.ones_like(albedo_set[0])
        plt.close('all')
        # save_albedo

        np.save(albedo_path, chosen_albedo)
        chosen_albedo[~mask] = 0
        show_albedo = chosen_albedo / chosen_albedo[mask].max() * 255
        cv2.imwrite(albedo_path[:-3] + 'png', np.uint8(show_albedo[:, :, ::-1]))
        save_transparent_img(albedo_path[:-3] + 'png', mask)

    else:
        albedo_path = os.path.join(data_dir, 'render_para/albedo.npy')
        chosen_albedo = np.load(albedo_path)

    imgs_albedo = chosen_albedo[np.newaxis] * imgs
    img_sv_albedo_path = os.path.join(data_dir, 'render_img/{}/imgs_sv_albedo.npy'.format(light_str))
    np.save(img_sv_albedo_path, imgs_albedo)


    for i in range(len(imgs_albedo)):
        show_img = imgs_albedo[i] / imgs_albedo[i].max() * 255
        cv2.imwrite(os.path.join(data_dir, 'render_img/{}/img_{}.png'.format(light_str, i)), np.uint8(show_img[:,:,::-1]))
        save_transparent_img(os.path.join(data_dir, 'render_img/{}/img_{}.png'.format(light_str, i)), mask)

    create_gif(imgs_albedo, img_sv_albedo_path[:-3]+'gif', mask, fps=3)

    return choice



def demo_set_albedo():
    para_folder = '../supp'
    albedo_folder = '../supp/texture'
    shape_set = np.genfromtxt(os.path.join(para_folder, 'shape_list.csv'), delimiter=',', dtype=np.str)
    base_data_dir = r'J:\Project\DeepPS3\data\synthetic'

    # shape_set = ['Nefertiti']
    light_variations = np.arange(20)

    for shape_name in shape_set:
        texture_choice = None

        for i in light_variations:
            data_dir = os.path.join(base_data_dir, shape_name.split('.')[0], 'distant_light_100_w_cast')
            texture_choice = set_albedo(data_dir, albedo_folder, 'PS3/{}'.format(i), texture_choice)
            texture_choice = set_albedo(data_dir, albedo_folder, 'PS2/{}'.format(i), texture_choice)


            data_dir = os.path.join(base_data_dir, shape_name.split('.')[0], 'distant_light_100')
            texture_choice = set_albedo(data_dir, albedo_folder, 'PS2/{}'.format(i), texture_choice)
            texture_choice = set_albedo(data_dir, albedo_folder, 'PS3/{}'.format(i), texture_choice)


if __name__ == '__main__':
    demo_set_albedo()

