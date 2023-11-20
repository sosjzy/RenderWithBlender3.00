import numpy as np
import time
import math
import configparser
import os
#输入：.npy .hdr .obj .tga
#输出：渲染图片文件夹


def createDir(dirpath):
    import os
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    return dirpath

def renderMultiInput(shape_set,hdri_set,camera_poses_file,texture_name_set):
    
    project_name = 'DiLiGenT_RT_proj'

    output_base_dir = r'E:\tools\blender3.00\DiLiGenT_RT_proj/result'
    ini_basic_path = r'E:\tools\blender3.00\DiLiGenT_RT_proj/config/templete1.ini'
    ini_used_path = r'E:\tools\blender3.00\DiLiGenT_RT_proj/config/current.ini'
    render_script_path = 'render_engine2.py'

    para_dir = r'E:\tools\blender3.00\DiLiGenT_RT_proj/supp'
    # LED_path = os.path.join(para_dir, 'light_euler_4.csv')
    # camera_pose_path = os.path.join(para_dir, 'pose_euler_4.csv')

    #npy
    print(camera_poses_file)
    camera_poses_name = camera_poses_file
    #hdr
    # hdri_name = "hdri-54.hdr"
    #tex
    # texture_name = "Animal-alien_cell_growth"


    ini = configparser.ConfigParser()
    ini.optionxform = str
    ini.read(ini_basic_path)
    print(ini_basic_path)

    ini['settings']['working_dir'] = para_dir
    ini['settings']['out_dir'] = output_base_dir
    createDir(output_base_dir)
    ini['light']['energy'] = "0"

    # num_light_samples = 100
    # num_rotation_samples = 0

    for shape_name in shape_set:
        for hdri_name in hdri_set:
            for texture_name in texture_name_set:
                ini['settings']['object_file'] = 'obj_data/{}.obj'.format(shape_name)
                # camera_pose_file_name_i = os.path.join(ini['settings']['out_dir'], 'camera_pose_of_{}_rot_{}.csv'.format(shape_name, 0))
    
                ini['object']['enable_rendering_by_degree'] = "False"
                #load texture
                ini['camera']['camera_poses_path'] = camera_poses_name
                ini['rendering']['use_hdri'] = hdri_name
                ini['sample2']['color_texture'] = texture_name+"/"+texture_name+"_baseColor.tga"
                ini['sample2']['metallic_texture']= texture_name+"/"+texture_name+"_metallic.tga"
                ini['sample2']['specular_texture']= texture_name+"/"+texture_name+"_specular.tga"
                ini['sample2']['roughness_texture']= texture_name+"/"+texture_name+"_roughness.tga"
                ini['sample2']['normal_texture']= texture_name+"/"+texture_name+"_normal.png"

                # np.savetxt(camera_pose_file_name_i, camera_poses_path, delimiter=',')
                filename_str = '{}_{}_{}'.format(shape_name,hdri_name,texture_name)
                ini['settings']['filename_str'] = filename_str
                with open(ini_used_path, 'w') as f:
                    ini.write(f)

                os.system("blender -b -P {} {} ".format(render_script_path, ini_used_path))

if __name__ =='__main__':
    shape_set = ['2', '1']
    hdri_set = ["hdri-54.hdr","hdri-29.hdr"]
    camera_poses_file = "NERO_camera.npy"
    texture_name_set = ["Animal-alien_cell_growth","Plastic-carbon_fiber_plain_weave"]
    
    renderMultiInput(shape_set,hdri_set,camera_poses_file,texture_name_set)

