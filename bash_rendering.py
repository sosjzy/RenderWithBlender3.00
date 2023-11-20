import numpy as np
import time
import math
import configparser
import os

def createDir(dirpath):
    import os
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    return dirpath

def generate_random_material():

    roughness = np.random.uniform(0.01, 1)
    specular = np.random.uniform(0.0, 2)
    ior = 2.0 / (1-np.sqrt(0.08 * specular)) - 1
    subsurface = np.random.uniform(0.1, 1)
    metallic = np.random.uniform(0.0, 0.1)
    return {'roughness': roughness, 'specular': specular, 'ior': ior, 'subsurface': subsurface, 'metallic': metallic}


def run(output_base_dir):
    ini_basic_path = './config/templete.ini'
    ini_used_path = './config/current.ini'
    render_script_path = 'render_engine.py'
    para_dir = './supp'
    shape_set = np.arange(30)

    ini = configparser.ConfigParser()
    ini.optionxform = str
    ini.read(ini_basic_path)

    obj_poses = np.loadtxt(os.path.join(para_dir, 'pose_euler_51.csv'), delimiter=',', dtype='float')
    numPose = len(obj_poses)

    ini['settings']['working_dir'] = os.path.join(para_dir, 'shape')
    ini['settings']['out_dir'] = output_base_dir
    createDir(output_base_dir)

    LED_path = os.path.join(para_dir, 'light_euler_100.csv')
    LED_position = np.loadtxt(LED_path, delimiter=',', dtype='float')
    LEDs_file_name = os.path.join(ini['settings']['out_dir'], 'light_position_elur.csv')
    ini['light']['light_directions_file'] = LEDs_file_name
    np.savetxt(LEDs_file_name, LED_position, delimiter=',')

    ini['light']['energy'] = '{}'.format(5)

    for j in range(numPose):
        for shape_name in shape_set:
            material_set = generate_random_material()
            ini['plastic_matte']['roughness'] = '{}'.format(material_set['roughness'])
            ini['plastic_matte']['specular'] = '{}'.format(material_set['specular'])
            ini['plastic_matte']['ior'] = '{}'.format(material_set['ior'])
            ini['plastic_matte']['subsurface'] = '{}'.format(material_set['subsurface'])
            ini['plastic_matte']['metallic'] = '{}'.format(material_set['metallic'])


            ini['object_custom']['rotation'] = '{}, {}, {}'.format(obj_poses[j, 0], obj_poses[j, 1], obj_poses[j, 2])
            filename_str = 'shape_{}_pose_{}'.format(shape_name, j)
            ini['settings']['filename_str'] = filename_str


            ini['settings']['object_file'] = os.path.join(para_dir, 'obj_data/{}.obj'.format(shape_name))
            if shape_name == 'Sphere':
                ini['object']['preset'] = 'object_sphere'
            else:
                ini['object']['preset'] = 'object_custom'

            with open(ini_used_path, 'w') as f:
                ini.write(f)

            os.system("blender -b -P {} {} ".format(render_script_path, ini_used_path))



if __name__ =='__main__':
    run('DiLiGenT_RT_proj_res')