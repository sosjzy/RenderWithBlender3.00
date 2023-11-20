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

shape_set = ['1', '2']
project_name = 'DiLiGenT_RT_proj'

output_base_dir = r'E:\tools\blender3.00\DiLiGenT_RT_proj\result'
ini_basic_path = r'E:\tools\blender3.00\DiLiGenT_RT_proj\config\templete.ini'
ini_used_path = r'E:\tools\blender3.00\DiLiGenT_RT_proj\config\current1.ini'
render_script_path = 'render_engine.py'
para_dir = r'E:\tools\blender3.00\DiLiGenT_RT_proj\supp'
LED_path = os.path.join(para_dir, 'light_euler_4.csv')
camera_pose_path = os.path.join(para_dir, 'pose_euler_4.csv')


ini = configparser.ConfigParser()
ini.optionxform = str
ini.read(ini_basic_path)
print(ini_basic_path)

ini['settings']['working_dir'] = para_dir
LED_position = np.loadtxt(LED_path, delimiter=',', dtype='float')
ini['settings']['out_dir'] = output_base_dir
createDir(output_base_dir)
ini['light']['energy'] = '{}'.format(6)

num_light_samples = 100
num_rotation_samples = 0

for shape_name in shape_set:
    ini['settings']['object_file'] = 'obj_data/{}.obj'.format(shape_name)
    LEDs_file_name_i = os.path.join(ini['settings']['out_dir'], 'light_positions_of_{}_rot_{}.csv'.format(shape_name, 0))
    ini['light']['light_directions_file'] = LEDs_file_name_i
    ini['object']['object_directions_file'] = camera_pose_path
    ini['object']['enable_rendering_by_degree'] = "False"

    np.savetxt(LEDs_file_name_i, LED_position, delimiter=',')
    filename_str = '{}_rot_{}'.format(shape_name, 0)
    ini['settings']['filename_str'] = filename_str
    with open(ini_used_path, 'w') as f:
        ini.write(f)

    os.system("blender -b -P {} {} ".format(render_script_path, ini_used_path))
