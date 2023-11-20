import numpy as np
import os
import sys
import bpy
import time
import math
import configparser
import _cycles
from mathutils import Matrix, Vector

sys.path.append('..')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# File config.ini must exist in the following location.
arglist = []
for i in sys.argv:
    if i.lower().endswith(".ini") and os.path.exists(i):
        arglist.append(i)


def setup_pose_inv(R_world2cv, T_world2cv):
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))
    R_cv2world = R_world2cv.T
    rotation = Matrix(R_cv2world.tolist()) @ R_bcam2cv
    location = -R_cv2world @ T_world2cv
    matrix_world = Matrix.Translation(location) @ rotation.to_4x4()
    return matrix_world


def setup_pose(R_world2cv, T_world2cv):
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))
    R_cv2world = R_world2cv
    rotation = Matrix(R_cv2world.tolist()) @ R_bcam2cv
    location = T_world2cv
    matrix_world = Matrix.Translation(location) @ rotation.to_4x4()
    return matrix_world


#return if it express true
def str2bool(s):
    return s.lower() in ["true", "t", "yes", "1"]

#return if it's axis or express true
def axis2bool(s):
    return s.lower() in ["true", "t", "yes", "1", "x", "y", "z"]


# RGB_data supports (255, 255, 255) or (255, 255, 255, opacity=255)
def get_RGB_float(RGB_data):
    if len(RGB_data) == 3 or len(RGB_data) == 4:
        if len(RGB_data) == 3:
            RGB_data = np.append(RGB_data, 255)
        RGB_float = RGB_data / 255
    return RGB_float


def delete_all_objects():
    '''
    remove all objects, meshes and materials
    '''
    for item in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(item)
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)
    for item in bpy.data.materials:
        bpy.data.materials.remove(item)


def setup_environment(params_rendering):
    '''
    set up cycles ray-tracing environment and misc settings
    Parameters
    ----------
    params_rendering : dict
        rendering parameters, used to set samples, bounces and so on
    '''
    scene = bpy.data.scenes['Scene']
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = int(params_rendering['samples'])
    scene.cycles.max_bounces = int(params_rendering['max_bounces'])
    scene.cycles.min_bounces = int(params_rendering['min_bounces'])
    scene.cycles.glossy_bounces = int(params_rendering['glossy_bounces'])
    scene.cycles.transmission_bounces = int(
        params_rendering['transmission_bounces'])
    scene.cycles.volume_bounces = int(params_rendering['volume_bounces'])
    scene.cycles.transparent_max_bounces = int(
        params_rendering['transparent_max_bounces'])
    scene.cycles.transparent_min_bounces = int(
        params_rendering['transparent_min_bounces'])
    scene.cycles.tile_x = int(params_rendering['tile_x'])
    scene.cycles.tile_y = int(params_rendering['tile_y'])
    scene.render.resolution_x = int(params_rendering['resolution_x'])
    scene.render.resolution_y = int(params_rendering['resolution_y'])
    bpy.types.RenderLayer.use_sky = True
    scene.display_settings.display_device = 'None'
    # bpy.context.scene.sequencer_colorspace_settings.name = 'Non-Color'
    bpy.context.scene.render.resolution_percentage = 100
    scene.render.image_settings.color_depth = '16'
    scene.render.image_settings.color_mode = params_rendering['color_mode']
    if 'disable_anti_aliasing' in params_rendering and str2bool(params_rendering['disable_anti_aliasing']):
        bpy.context.scene.cycles.pixel_filter_type = 'GAUSSIAN'
        bpy.context.scene.cycles.filter_width = 0.01
    bpy.context.scene.render.film_transparent = str2bool(
        params_rendering['transparent_background'])

    bpy.context.scene.view_layers['ViewLayer'].cycles.use_denoising = str2bool(
        params_rendering['denoising_legacy'])
    bpy.context.scene.view_layers['ViewLayer'].cycles.use_sky = False
    bpy.context.scene.view_layers['ViewLayer'].cycles.use_ao = False
    bpy.context.scene.view_layers['ViewLayer'].cycles.use_strand = False
    bpy.context.scene.cycles_curves.use_curves = False


def set_background(params_rendering):
    '''
    set background color in rendering
    Parameters
    ----------
    params_rendering : dict
        rendering parameters, used to decide enabling custom color or not
    '''
    # select world node tree
    nt = bpy.data.worlds['World'].node_tree
    rgb_node = nt.nodes.new("ShaderNodeRGB")
    RGB_data = np.array([float(x.strip()) for x in params_rendering['background_color'].split(
        ',') if not x.strip() == ''], dtype=np.float32)
    # RGB_data supports (255, 255, 255) or (255, 255, 255, opacity=255)
    if len(RGB_data) == 3 or len(RGB_data) == 4:
        if len(RGB_data) == 3:
            RGB_data = np.append(RGB_data, 255)
        RGB_data = RGB_data / 255
    rgb_node.outputs['Color'].default_value = RGB_data
    back_node = nt.nodes['Background']
    nt.links.new(rgb_node.outputs['Color'], back_node.inputs['Color'])


def set_hdri(params_rendering, working_dir):
    '''
    set environment HDRI with IBL, required hdr format image
    Parameters
    ----------
    params_rendering : dict
        rendering parameters, used to load hdr file and adjust HDRI offset
    working_dir : str
        working directory, used to load hdr file
    '''
    nt = bpy.data.worlds['World'].node_tree
    if not os.path.exists(params_rendering['use_hdri']):
        params_rendering['use_hdri'] = os.path.join(
            working_dir, params_rendering['use_hdri'])
    input_node = nt.nodes.new("ShaderNodeTexCoord")
    mapping_node = nt.nodes.new("ShaderNodeMapping")
    mapping_node.inputs[2].default_value[0] = math.pi / 2 + 0
    mapping_node.inputs[2].default_value[2] = float(
        params_rendering['hdri_offset'])
    env_node = nt.nodes.new("ShaderNodeTexEnvironment")
    env_node.image = bpy.data.images.load(params_rendering['use_hdri'])
    back_node = nt.nodes['Background']
    nt.links.new(input_node.outputs[0], mapping_node.inputs[0])
    nt.links.new(mapping_node.outputs[0], env_node.inputs[0])
    nt.links.new(env_node.outputs['Color'], back_node.inputs['Color'])
    if 'no_background' in params_rendering and str2bool(params_rendering['no_background']):
        bpy.context.scene.world.cycles_visibility.camera = False
    bpy.context.view_layer.update()


def set_light(params_light):
    '''
    set light in accordance with params_light
    Parameters
    ----------
    params_light : dict
        light parameters, defined by [light] section

    Returns
    -------
    light_names : list
        set of light names, set size is defined by light*amount (usual is 1)
    light_type : str
        light type, SUN(distant) or POINT(near)
    '''
    light_names = []
    if params_light['model'].lower() == "area":
        light_type = "AREA"

    light_objects = []
    for i in range(int(params_light['amount'])):
        light_data = bpy.data.lights.new(
            name="Light{}".format(i + 1), type=light_type)
        light_data.use_nodes = True
        light_object = bpy.data.objects.new(
            name='Light{}'.format(i + 1), object_data=light_data)
        light_object.data.energy = float(params_light['energy'])
        light_object.data.cycles.max_bounces = 0
        light_object.data.size = float(params_light['AREA_size'])
        light_object.data.shape = 'SQUARE'
        bpy.data.scenes['Scene'].collection.objects.link(light_object)
        light_object.select_set(state=True)
        bpy.context.view_layer.update()
        light_names.append('Light{}'.format(i + 1))
        light_object.select_set(state=False)
        light_objects.append(light_object)
    return light_names, light_type, light_objects   

def adjust_camera(camera_object, pose):
    camera_object.location = pose[:3]
    camera_object.rotation_euler = pose[3:]

    return camera_object


def set_camera(params_camera):
    '''
    set camera in accordance with params_camera
    Parameters
    ----------
    params_camera : dict
        camera parameters, defined by [camera] section
    '''
    camera_data = bpy.data.cameras.new("Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.data.scenes['Scene'].collection.objects.link(camera_object)
    bpy.data.scenes['Scene'].camera = camera_object

    camera_object.location = [0, 0, 0]
    camera_object.rotation_euler = [0, np.pi, 0]
    if 'location' in params_camera and params_camera['location'] != '':
        camera_object.location = [float(
            x.strip()) for x in params_camera['location'].split(',') if not x.strip() == '']
    # attention; changing camera rotation makes illegal normal map
    if 'rotation' in params_camera and params_camera['rotation'] != '':
        camera_object.rotation_euler = [float(
            x.strip()) for x in params_camera['rotation'].split(',') if not x.strip() == '']
    # perspective or orthographic
    if params_camera['model'].lower() == "orthographic":
        camera_object.data.type = 'ORTHO'
    if params_camera['model'].lower() == "perspective":
        camera_object.data.type = 'PERSP'
    if params_camera['model'].lower() == "panorama":
        camera_object.data.type = 'PANO'
    # setup Full Frame 35mm Camera
    camera_object.data.sensor_fit = 'HORIZONTAL'
    if params_camera['model'].lower() != "orthographic":
        camera_object.data.sensor_width = float(params_camera['sensor_width'])
        camera_object.data.sensor_height = float(params_camera['sensor_height'])
        camera_object.data.lens = float(params_camera['focal_length'])

    for key, value in params_camera.items():
        if key.lower() in ['clip_start', 'clip_end', 'ortho_scale', 'shift_x', 'shift_y']:
            setattr(camera_object.data, key.lower(), float(value))
        if key.lower() in ['use_dof'] and str2bool(value):
            camera_object.data.dof.use_dof = True
        if key.lower() in ['focus_distance', 'aperture_fstop', 'aperture_rotation', 'aperture_ratio']:
            setattr(camera_object.data.dof, key.lower(), float(value))
        if key.lower() in ['aperture_blades']:
            setattr(camera_object.data.dof, key.lower(), int(value))
    return camera_object

def set_sphere(params_sphere, sphere_count):
    '''
    set sphere in accordance with params_sphere
    Parameters
    ----------
    params_sphere : dict
        object parameters, defined by [object][preset] -> [object_sphere] section
    sphere_count : int
        when setting second or higher sphere, object name is numbered automatically

    Returns
    -------
    name : str
        sphere name, e.g. 'Sphere2'
    '''
    sphere_location = [float(x.strip()) for x in params_sphere['location'].split(
        ',') if not x.strip() == '']
    sphere_rotation = [float(x.strip()) for x in params_sphere['rotation'].split(
        ',') if not x.strip() == '']
    bpy.ops.mesh.primitive_uv_sphere_add(segments=int(params_sphere['segments']),
                                         ring_count=int(
                                             params_sphere['ring_count']),
                                         radius=float(params_sphere['radius']),
                                         calc_uvs=True, enter_editmode=False, align='WORLD',
                                         location=sphere_location,
                                         rotation=sphere_rotation)
    if sphere_count == 0:
        name = bpy.context.selected_objects[0].name = 'Sphere'
    else:
        name = bpy.context.selected_objects[0].name = 'Sphere{}'.format(
            sphere_count + 1)
    obj_object = bpy.data.objects[name]
    if str2bool(params_sphere['use_smooth']) and obj_object.type == "MESH":
        for polygon in obj_object.data.polygons:
            polygon.use_smooth = True
    if 'no_shadow' in params_sphere and str2bool(params_sphere['no_shadow']):
        obj_object.cycles_visibility.shadow = False

    if str2bool(params_sphere['subdivision_surface']) and obj_object.type == "MESH":
        mod = obj_object.modifiers.new('Subdivision', 'SUBSURF')
        mod.render_levels = 2
        mod.quality = 3
    return name


def set_custom_object(params_custom, custom_object_name, working_dir):
    '''
    set custom object from local file in accordance with params_custom
    supported format: obj, fbx, x3d, ply and stl
    Parameters
    ----------
    params_custom : dict
        object parameters, defined by [object][preset] -> [object_custom] section
    custom_object_name : str
        object file location

    Returns
    -------
    name : str
        object name, e.g. 'Bunny'
    '''
    if not os.path.exists(custom_object_name):
        custom_object_name = os.path.join(working_dir, custom_object_name)
    if custom_object_name.lower().endswith('obj'):
        # when importing from .obj file
        bpy.ops.import_scene.obj(filepath=custom_object_name, split_mode='OFF')
    elif custom_object_name.lower().endswith('fbx'):
        # when importing from .fbx file
        object_dir = os.path.dirname(custom_object_name)
        bpy.ops.import_scene.fbx(
            filepath=custom_object_name, directory=object_dir)
    elif custom_object_name.lower().endswith('x3d'):
        # when importing from .x3d file
        bpy.ops.import_scene.x3d(filepath=custom_object_name)
    elif custom_object_name.lower().endswith('ply'):
        # when importing .ply object
        object_dir = os.path.dirname(custom_object_name)
        bpy.ops.import_mesh.ply(
            filepath=custom_object_name, directory=object_dir)
    else:
        # when importing .stl object
        object_dir = os.path.dirname(custom_object_name)
        bpy.ops.import_mesh.stl(
            filepath=custom_object_name, directory=object_dir)
    name = bpy.context.selected_objects[0].name = os.path.splitext(
        os.path.basename(custom_object_name))[0]
    obj_object = bpy.data.objects[name]

    # set object center to object origin
    if str2bool(params_custom['geometry_to_origin']):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    if 'location' in params_custom and params_custom['location'] != '':
        object_location = [float(x.strip()) for x in params_custom['location'].split(
            ',') if not x.strip() == '']
        obj_object.location = object_location
    else:
        obj_object.location = [0, 0, 0]
    if 'rotation' in params_custom and params_custom['rotation'] != '':
        object_rotation = [float(x.strip()) for x in params_custom['rotation'].split(
            ',') if not x.strip() == '']
        obj_object.rotation_euler = object_rotation
    else:
        obj_object.rotation_euler = [0, 0, 0]
    if 'max_dimension' in params_custom and params_custom['max_dimension'] != '':
        if params_custom['max_dimension'].lower() not in ['none']:
            obj_object.scale = (1, 1, 1)
            bpy.context.view_layer.update()
            dims = obj_object.dimensions
            scale = float(params_custom['max_dimension']
                          ) / max(dims[0], dims[1], dims[2])
            obj_object.scale = (scale, scale, scale)

    if str2bool(params_custom['use_smooth']) and obj_object.type == "MESH":
        for polygon in obj_object.data.polygons:
            polygon.use_smooth = True
    if str2bool(params_custom['no_shadow']):
        obj_object.cycles_visibility.shadow = False
    return name


def set_material(idx, params_current_shader, current_object, working_dir):
    '''
    set Principled BSDF material to particular object
    Parameters
    ----------
    idx : int
        index of params_shader, found in [shader][preset]
    params_current_shader : dict
        Principled BSDF parameters of current object
    current_object : str
        object name that material is applied to
    working_dir : str
        working directory, used to load texture maps
    '''
    uv_flag = False
    isLambert = False
    mat = bpy.data.materials.new('Material{}'.format(idx))
    mat.use_nodes = True
    pbs_node = mat.node_tree.nodes['Principled BSDF']
    tex_node = mat.node_tree.nodes.new('ShaderNodeTexCoord')
    for key, value in params_current_shader.items():
        if key.lower().replace('_', ' ') in ["subsurface", "metallic", "specular", "specular tint", "roughness", "anisotropic",
                                             "anisotropic rotation", "sheen", "sheen tint", "clearcoat", "clearcoat roughness",
                                             "transmission", "transmission roughness", "alpha"] and value != '':
            pbs_node.inputs[key.title().replace(
                '_', ' ')].default_value = float(value)
        if key.lower() in ["ior"] and value != '':
            pbs_node.inputs[key.upper()].default_value = float(value)
        if key.lower().replace('_', ' ') in ["base color", "subsurface color", "emission"] and value != '':
            RGB_data = np.array([float(x.strip()) for x in params_current_shader[key].split(
                ',') if not x.strip() == ''], dtype=np.float32)
            RGB_data = get_RGB_float(RGB_data)
            pbs_node.inputs[key.title().replace(
                '_', ' ')].default_value = RGB_data


    # set texture:
    for key, value in params_current_shader.items():
        if key.lower() in ["load_uv"] and str2bool(value) and bpy.data.objects[current_object].type == 'MESH':
            if bpy.data.meshes[current_object].uv_layers.active is not None:
                uv_flag = True
        if key.lower() in ["color_texture", "metallic_texture", "specular_texture", "roughness_texture", "sheen_texture",
                           "sheentint_texture", "ior_texture"] and value != '':
            socket, amp_node, texture_node = set_texture_mapping(idx=idx, key=key, value=value,
                                working_dir=working_dir, uv_flag=uv_flag)

            if key.lower() == "color_texture":
                mat.node_tree.links.new(amp_node.outputs[0], pbs_node.inputs[0])
                mat.node_tree.links.new(texture_node.outputs[0], pbs_node.inputs[0])
            else:
                if not isLambert:
                    mat.node_tree.links.new(amp_node.outputs[0], pbs_node.inputs[socket])


    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE':
            node.interpolation = 'Closest'
            if not uv_flag:
                node.projection = 'SPHERE'
    inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
    outp = pbs_node.outputs['BSDF']
    mat.node_tree.links.new(inp, outp)
    bpy.data.objects[current_object].active_material = mat


def set_texture_mapping(idx, key, value, working_dir, uv_flag):
    '''
    connect texture image to Principled BSDF parameters
    Parameters
    ----------
    idx : int
        index of params_shader, found in [shader][preset]
    key : str
        keyword of distinguishing texture, e.g. 'metallic_texture'
    value : str
        value of parameters, supposed this is texture image file location
    working_dir : str
        working directory, used to load texture maps
    uv_flag : boolean
        whether or not set texture according to UV, if exists
    '''
    mat = bpy.data.materials['Material{}'.format(idx)]
    coordinate_node = mat.node_tree.nodes.new('ShaderNodeTexCoord')
    texture_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
    amp_node = mat.node_tree.nodes.new('ShaderNodeMath')
    amp_node.operation = 'MULTIPLY'
    # pbs_node = mat.node_tree.nodes['Principled BSDF']
    if not os.path.exists(value):
        value = os.path.join(working_dir, value)
    try:
        texture_node.image = bpy.data.images[os.path.basename(value)]
    except KeyError:
        texture_node.image = bpy.data.images.load(value)
    bpy.data.images[os.path.basename(
        value)].colorspace_settings.name = 'Non-Color'
    if key.lower() in ["specular_texture", "ior_texture"]:
        amp_node.inputs[1].default_value = 10
    else:
        amp_node.inputs[1].default_value = 1
    mat.node_tree.links.new(coordinate_node.outputs[0], texture_node.inputs[0])
    mat.node_tree.links.new(texture_node.outputs[0], amp_node.inputs[0])
    if key.lower() == "color_texture": socket = 0
    if key.lower() == "metallic_texture": socket = 4
    if key.lower() == "specular_texture": socket = 5
    if key.lower() == "roughness_texture": socket = 7
    if key.lower() == "sheen_texture": socket = 10
    if key.lower() == "sheentint_texture": socket = 11
    if key.lower() == "ior_texture": socket = 14
    if uv_flag:
        mat.node_tree.links.new(
            coordinate_node.outputs[2], texture_node.inputs[0])

    # mat.node_tree.links.new(amp_node.outputs[0], pbs_node.inputs[socket])
    # if key.lower() == "color_texture":
    #     mat.node_tree.links.new(texture_node.outputs[0], pbs_node.inputs[0])

    return socket, amp_node, texture_node


# def create_light_directions(params_light, working_dir):
#     '''
#     create light_directions_euler data from csv file or 360 light directions
#     Parameters
#     ----------
#     params_light : dict
#         light parameters, used to adjust light properties
#     working_dir : str
#         working directiory, used to load light directions file

#     Returns
#     -------
#     light_directions_euler : array
#         contains [Euler_X][Euler_Y][Euler_Z][Energy][SUN_angple or POINT_size]
#         euler angle is based on XYZ
#     '''
#     if params_light['model'].lower() == 'distant':
#         if axis2bool(params_light['enable_rendering_by_degree']):
#             degree = float(params_light['degree'])
#             light_directions_euler = np.zeros((int(360 / degree), 5))
#             for i in range(0, int(360 / degree)):
#                 if params_light['enable_rendering_by_degree'].lower() in ["x"]: axis = 0
#                 elif params_light['enable_rendering_by_degree'].lower() in ["z"]: axis = 2
#                 else: axis = 1
#                 light_directions_euler[i][axis] = math.pi * i * degree / 180
#                 light_directions_euler[i][3] = float(params_light['energy'])
#                 light_directions_euler[i][4] = float(params_light['SUN_angle'])
#         else:
#             if not os.path.exists(params_light['light_directions_file']):
#                 params_light['light_directions_file'] = os.path.join(
#                     working_dir, params_light['light_directions_file'])
#             with open(os.path.abspath(params_light['light_directions_file']), 'r') as f:
#                 light_directions = [np.asarray(ldir.strip().split(',')).astype(
#                     np.float32) for ldir in f.readlines()]
#             for i in range(0, len(light_directions)):
#                 if len(light_directions[i]) == 3 or light_directions[i][3] == '':
#                     light_directions[i] = np.append(light_directions[i], -1)
#                 if len(light_directions[i]) == 4 or light_directions[i][4] == '':
#                     light_directions[i] = np.append(
#                         light_directions[i], float(params_light['SUN_angle']))
#             light_directions_euler = np.zeros((len(light_directions), 5))
#             for i in range(0, len(light_directions)):
#                 if str2bool(params_light['file_normalization']):
#                     light_directions_euler[i][0] = -np.arcsin(light_directions[i][1] / np.sqrt(
#                         light_directions[i][1] ** 2 + light_directions[i][2] ** 2))
#                     light_directions_euler[i][1] = np.arcsin(
#                         light_directions[i][0] / np.sqrt(light_directions[i][0] ** 2 + light_directions[i][2] ** 2))
#                 else:
#                     light_directions_euler[i][0:3] = light_directions[i][0:3]
#                 light_directions_euler[i][3:5] = light_directions[i][3:5]
#     if params_light['model'].lower() == 'near':
#         if axis2bool(params_light['enable_rendering_by_degree']):
#             degree = float(params_light['degree'])
#             light_directions_euler = np.zeros((int(360 / degree), 5))
#             radius = float(params_light['POINT_size'])
#             for i in range(0, int(360 / degree)):
#                 if params_light['enable_rendering_by_degree'].lower() in ["x"]: axis = 0
#                 elif params_light['enable_rendering_by_degree'].lower() in ["y"]: axis = 1
#                 else: axis = 2
#                 light_directions_euler[i][(
#                     axis + 1) % 3] = radius * math.cos(math.pi * i * degree / 180)
#                 light_directions_euler[i][(
#                     axis + 2) % 3] = radius * math.sin(math.pi * i * degree / 180)
#                 light_directions_euler[i][3] = float(params_light['energy'])
#                 light_directions_euler[i][4] = float(
#                     params_light['POINT_size'])
#         else:
#             if not os.path.exists(params_light['light_directions_file']):
#                 params_light['light_directions_file'] = os.path.join(
#                     working_dir, params_light['light_directions_file'])
#             with open(os.path.abspath(params_light['light_directions_file']), 'r') as f:
#                 light_directions = [np.asarray(ldir.strip().split(',')).astype(
#                     np.float32) for ldir in f.readlines()]
#             for i in range(0, len(light_directions)):
#                 if len(light_directions[i]) == 3 or light_directions[i][3] == '':
#                     light_directions[i] = np.append(light_directions[i], float(params_light['energy']))
#                 if len(light_directions[i]) == 4 or light_directions[i][4] == '':
#                     light_directions[i] = np.append(
#                         light_directions[i], float(params_light['POINT_size']))


#             light_directions_euler = np.zeros((len(light_directions), 5))
#             for i in range(0, len(light_directions)):
#                 light_directions_euler[i][0:5] = light_directions[i][0:5]

#     amount = int(params_light['amount'])
#     light_directions_euler = light_directions_euler.reshape(
#         int(len(light_directions_euler) / amount), amount, 5)

#     return light_directions_euler


def create_object_directions(params_object_global, working_dir):
    '''
    create object_directions_euler data from csv file or 360 light directions
    Parameters
    ----------
    params_object_global : dict
        light parameters, used to adjust object properties
    working_dir : str
        working directiory, used to load object directions file

    Returns
    -------
    object_directions_euler : array
        contains [Euler_X][Euler_Y][Euler_Z]
        euler angle is based on XYZ
    '''
    if str2bool(params_object_global['enable_multiple_views']):
        if axis2bool(params_object_global['enable_rendering_by_degree']):
            degree = float(params_object_global['degree'])
            object_directions_euler = np.zeros((int(360 / degree), 3))
            if params_object_global['enable_rendering_by_degree'].lower() in ["x"]: axis = 0
            elif params_object_global['enable_rendering_by_degree'].lower() in ["z"]: axis = 2
            else: axis = 1
            for i in range(0, int(360 / degree)):
                object_directions_euler[i][axis] = math.pi * i * degree / 180
        else:
            if not os.path.exists(params_object_global['object_directions_file']):
                params_object_global['object_directions_file'] = os.path.join(
                    working_dir, params_object_global['object_directions_file'])
            with open(os.path.abspath(params_object_global['object_directions_file']), 'r') as f:
                object_directions = [np.asarray(ldir.strip().split(',')).astype(
                    np.float32) for ldir in f.readlines()]
            object_directions = np.asarray(object_directions)
            object_directions_euler = np.zeros(object_directions.shape)
            for i in range(object_directions.shape[0]):
                object_directions_euler[i][0:3] = object_directions[i][0:3]
    else:
        object_directions_euler = np.zeros((1, 3))

    return object_directions_euler


def get_intrinsic_matrix():
    camera_data = bpy.data.objects["Camera"].data
    scene = bpy.context.scene
    focal_length = camera_data.lens
    resolution_x = scene.render.resolution_x
    resolution_y = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width = camera_data.sensor_width
    sensor_height = camera_data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    s_u = resolution_x * scale / sensor_width
    s_v = resolution_y * scale * pixel_aspect_ratio / sensor_height
    # Parameters of intrinsic calibration matrix K
    alpha_u = focal_length * s_u
    alpha_v = focal_length * s_v
    u_0 = resolution_x * scale / 2
    v_0 = resolution_y * scale / 2
    skew = 0  # only use rectangular pixels
    K = [alpha_u, skew, u_0, 0, alpha_v, v_0, 0, 0, 1]
    return K


def write_info(out_dir_v, filenames, light_type, lights, obj_names):
    camera_object = bpy.data.objects['Camera']
    with open(os.path.join(out_dir_v, 'filenames.ini'), 'w') as f:
        f.write("[filenames]")
        for i in range(0, len(filenames)):
            f.write("\n{0:03d} = {1}".format(i + 1, filenames[i]))
        if light_type == 'SUN':
            f.write("\n\n[light rotations]")
        if light_type == 'POINT':
            f.write("\n\n[light locations]")
        for i in range(0, len(filenames)):
            f.write("\n{:03d} = ".format(i + 1))
            for j in lights[i]:
                f.write("{0:.5f}, {1:.5f}, {2:.5f}\t".format(j[0], j[1], j[2]))
        f.write("\n\n[info]")
        loc, rot = camera_object.location, camera_object.rotation_euler
        f.write("\ncamera_location = {0:.5f}, {1:.5f}, {2:.5f} [m]".format(
            loc[0], loc[1], loc[2]))
        f.write("\ncamera_rotation = {0:.5f}, {1:.5f}, {2:.5f} [rad]".format(
            rot[0], rot[1], rot[2]))
        f.write("\ncamera_intrinsic = {}".format(get_intrinsic_matrix()))
        for item in obj_names:
            loc, rot = bpy.data.objects[item].location, bpy.data.objects[item].rotation_euler
            f.write("\n{0}_location = {1:.5f}, {2:.5f}, {3:.5f} [m]".format(
                item, loc[0], loc[1], loc[2]))
            f.write("\n{0}_rotation = {1:.5f}, {2:.5f}, {3:.5f} [rad]".format(
                item, rot[0], rot[1], rot[2]))
            if bpy.data.objects[item].type == 'MESH':
                data = bpy.data.objects[item].data
                f.write("\n{0}_vertices = {1}\n{0}_edges = {2}\n{0}_polygons = {3}".format(
                    item, len(data.vertices), len(data.edges), len(data.polygons)))




def create_normal_map(filepath, npy=False):
    '''
    create normal map and some variants
    Parameters
    ----------
    filepath : str
        output directory of normal maps
    npy : boolean
        whether or not save normal as npy, as well as png
    '''
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.color_depth = '32'
    bpy.context.scene.render.image_settings.exr_codec = 'NONE'

    # bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    # turn off gamma correction. default is "sRGB".
    bpy.context.scene.display_settings.display_device = "None"

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_position = False
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True

    vector_curves = tree.nodes.new('CompositorNodeCurveVec')
    curves = vector_curves.mapping.curves
    for curve in curves:
        curve.points[0].location.xy = [-1, 0]
        curve.points[1].location.xy = [1, 1]

    comp_node = tree.nodes.new('CompositorNodeComposite')
    viewer_node = tree.nodes.new('CompositorNodeViewer')
    invert = tree.nodes.new(type="CompositorNodeInvert")
    output_node = tree.nodes.new('CompositorNodeOutputFile')
    output_node.base_path = filepath

    # normal path of Render Layers to Vector Curves
    links.new(rl.outputs[3], vector_curves.inputs[0])
    # invert to Composite
    links.new(vector_curves.outputs[0], comp_node.inputs[0])
    links.new(vector_curves.outputs[0],
              viewer_node.inputs[0])  # invert to Viewer
    links.new(rl.outputs[3], output_node.inputs[0])  # save normals_alt.png

    bpy.ops.render.render()

    bpy.data.images['Render Result'].save_render(
        filepath=os.path.join(filepath, 'normals.exr'))
    # bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = False
    try:
        os.rename(os.path.join(filepath, 'Image0001.exr'),
                  os.path.join(filepath, 'normals_alt.exr'))
    except:
        if os.path.exists(os.path.join(filepath, 'normals_alt.exr')):
            os.remove(os.path.join(filepath, 'normals_alt.exr'))
        os.rename(os.path.join(filepath, 'Image0001.exr'),
                  os.path.join(filepath, 'normals_alt.exr'))

    bpy.context.scene.use_nodes = False


def create_depth_map(filepath, npy=False):
    '''
    create normalized depth map, distance map and some masks
    Parameters
    ----------
    filepath : str
        output directory of depth maps
    npy : boolean
        whether or not save depth as npy, as well as png
        distance.npy and mask.npy are saved regardless of this
    '''
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.color_depth = '32'
    bpy.context.scene.render.image_settings.exr_codec = 'NONE'

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    # turn off gamma correction. default is "sRGB".
    bpy.context.scene.display_settings.display_device = "None"

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')
    comp_node = tree.nodes.new('CompositorNodeComposite')
    viewer_node = tree.nodes.new('CompositorNodeViewer')

    links.new(rl.outputs[2], comp_node.inputs[0])
    links.new(rl.outputs[2], viewer_node.inputs[0])  # invert to Viewer

    # render
    bpy.ops.render.render()
    bpy.data.images['Render Result'].save_render(
        filepath=os.path.join(filepath, 'depth.exr'))

    bpy.context.scene.use_nodes = False


def set_scene_enable_gpu(scene):
    scene.cycles.device = 'GPU'
    avail_devices = _cycles.available_devices("CUDA")
    print(avail_devices)
    cprefs = bpy.context.preferences.addons["cycles"].preferences
    cprefs.get_devices()

    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            print(compute_device_type)
            break
        except TypeError:
            pass

    # Enable all CPU and GPU devices
    for device in cprefs.devices:
        print(device)
        device.use = True


def denoising():
    '''
    enable denoising
    powered by Intel Open Image Denoise
    '''
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    denoise_node = tree.nodes.new('CompositorNodeDenoise')
    comp_node = tree.nodes.new('CompositorNodeComposite')
    links.new(rl.outputs[0], denoise_node.inputs[0])
    links.new(rl.outputs[3], denoise_node.inputs[1])
    links.new(denoise_node.outputs[0], comp_node.inputs[0])


def render_object(object_shape, params_rendering, params_camera, params_object_global,
                  params_object, params_shader, params_light, working_dir, out_dir, filename_str = None, *args, **kwargs):

    # light_directions_euler = create_light_directions(
    #     params_light=params_light, working_dir=working_dir)

    delete_all_objects()
    scene = bpy.data.scenes['Scene']
    # setup cycles environment
    setup_environment(params_rendering=params_rendering)
    set_scene_enable_gpu(scene)

    # world editting - background and HDRI
    scene.world.use_nodes = True
    use_hdri = False
    if 'background_color' in params_rendering and params_rendering['background_color'] != '':
        set_background(params_rendering=params_rendering)
    if 'use_hdri' in params_rendering and (os.path.exists(params_rendering['use_hdri']) or
                                           os.path.exists(os.path.join(working_dir, params_rendering['use_hdri']))):
        use_hdri = True
        set_hdri(params_rendering=params_rendering, working_dir=working_dir)

    # set light
    light_names, light_type, light_objects = set_light(params_light=params_light)
    # set camera
    camera_obj = set_camera(params_camera=params_camera)
    # set object
    sphere_count = 0
    obj_names = []
    object_shape_list = [x.strip()
                         for x in object_shape.split(',') if not x.strip() == '']
    for i in range(0, len(object_shape_list)):

        if object_shape_list[i].lower() == 'sphere':
            name = set_sphere(
                params_sphere=params_object[i], sphere_count=sphere_count)
            sphere_count += 1
        else:
            name = set_custom_object(
                params_custom=params_object[i], custom_object_name=object_shape_list[i], working_dir=working_dir)
        obj_names.append(name)
    # set material
    for material in bpy.data.materials:
        if 'Default_OBJ' in material.name or 'None' in material.name or 'Material0' in material.name:
            bpy.data.materials.remove(material)
    for i in range(0, len(params_shader)):
        set_material(
            idx=i, params_current_shader=params_shader[i], current_object=obj_names[i], working_dir=working_dir)

    obj_name = '_'.join(obj_names)
    out_dir = os.path.join(os.path.abspath(out_dir), '{}/{}'.format(obj_name, filename_str))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for area in bpy.context.screen.areas:
        if area.type in ['IMAGE_EDITOR', 'VIEW_3D']:
            area.tag_redraw()

    # create object euler data
    # object_directions_euler = create_object_directions(
    #     params_object_global=params_object_global, working_dir=working_dir)

    # rendering
    bpy.context.scene.transform_orientation_slots[0].type = 'GLOBAL'
    bpy.context.scene.tool_settings.transform_pivot_point = 'CURSOR'

    # for index, label in enumerate(object_directions_euler):
    #     if str2bool(params_rendering['denoising']):
    #         denoising()
    #     if str2bool(params_object_global['enable_multiple_views']):
    #         out_dir_v = os.path.join(os.path.abspath(
    #             out_dir), "view_{:03d}".format(index + 1))
    #     else:
    #         out_dir_v = out_dir
    #     if not os.path.exists(out_dir_v):
    #         os.makedirs(out_dir_v)
    camera_poses_set = np.load(os.path.join(working_dir, params_camera['camera_poses_path']))
    for cidx, pose in enumerate(camera_poses_set):

        out_dir_v = os.path.join(os.path.abspath(
            out_dir), "view_{:03d}".format(cidx + 1))
        if not os.path.exists(out_dir_v):
            os.makedirs(out_dir_v)

        R, T = pose[:, :3], pose[:, 3:]
        camera_pose_current =  setup_pose_inv(R, T)
        camera_obj.matrix_world = camera_pose_current

        if str2bool(params_rendering['denoising']):
            denoising()

        for item in obj_names:
            bpy.data.objects[item].select_set(state=True)

        # ov = bpy.context.copy()
        # ov['area'] = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"][0]
        # bpy.ops.transform.rotate(ov, value=-label[0], orient_axis='X')
        # bpy.ops.transform.rotate(ov, value=-label[1], orient_axis='Y')
        # bpy.ops.transform.rotate(ov, value=-label[2], orient_axis='Z')

        filenames = []
        lights = []
        # for i, l in enumerate(light_directions_euler):
        #     for j, k in zip(light_names, l):
        #         if k[3] >= 0:
        #             bpy.data.objects[j].data.energy = k[3]
        #         if light_type == 'SUN':
        #             bpy.data.objects[j].data.angle = k[4]
        #         if light_type == 'POINT':
        #             bpy.data.objects[j].data.shadow_soft_size = k[4]
        #             bpy.data.objects[j].data.cycles.max_bounces = 0

        #         k = np.resize(k, 3)


        #         if light_type == 'SUN':
        #             bpy.data.objects[j].rotation_euler = k.tolist()
        #         if light_type == 'POINT':
        #             bpy.data.objects[j].location = k.tolist()
        for j in light_names:

            bpy.data.objects[j].data.energy = float(params_light['energy'])
            bpy.data.objects[j].data.shape = 'RECTANGLE'
            bpy.data.objects[j].data.cycles.max_bounces = 0
            bpy.data.objects[j].data.size = float(params_light['AREA_size'])
            bpy.data.objects[j].data.size_y = float(params_light['AREA_size_y'])
            bpy.data.objects[j].matrix_world = camera_pose_current


            area_light_data = bpy.data.objects[j].data
            area_light_node = area_light_data.node_tree.nodes
            area_texture_node = area_light_node.new('ShaderNodeTexImage')
            area_texture_node.image = bpy.data.images.load(os.path.join(working_dir, params_light['texture_path']))
            area_geo_node = area_light_node.new(type="ShaderNodeNewGeometry")
            area_emission_node = area_light_node.get('Emission')
            area_emission_node.inputs[1].default_value = float(params_light['emission_strength'])

            area_light_links = area_light_data.node_tree.links
            area_light_links.new(area_texture_node.outputs[0], area_emission_node.inputs[0])
            area_light_links.new(area_geo_node.outputs[5], area_texture_node.inputs[0])


        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.render.image_settings.color_depth = '32'
        bpy.context.scene.render.image_settings.exr_codec = 'NONE'
        bpy.ops.render.render()
        filepath = os.path.join(out_dir_v, '{:03d}.exr'.format(i + 1))

        bpy.data.images['Render Result'].save_render(filepath=filepath)

        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_depth = '16'
        # Define the filepath for PNG
        filepath_png = filepath[:-3] + 'png'

        # Save the render result in PNG format
        bpy.data.images['Render Result'].save_render(filepath=filepath_png)


            # filenames.append(os.path.basename(filepath))
            # tmp = []
            # for items in light_names:
            #     if light_type == 'SUN':
            #         light_info = bpy.data.objects[items].rotation_euler
            #     if light_type == 'POINT':
            #         light_info = bpy.data.objects[items].location
            #     tmp.append((light_info[0], light_info[1], light_info[2]))
            # lights.append(tmp)
            # bpy.data.images['Render Result'].save_render(filepath=filepath)

        filenames.append(os.path.basename(filepath))
        tmp = []
        for items in light_names:
            tmp.append((bpy.data.objects[items].rotation_euler[0], bpy.data.objects[items].rotation_euler[1], bpy.data.objects[items].rotation_euler[2],
                        bpy.data.objects[items].location[0], bpy.data.objects[items].location[1], bpy.data.objects[items].location[2],))


        lights.append(tmp)

        write_info(out_dir_v=out_dir_v, filenames=filenames,
                   light_type=light_type, lights=lights, obj_names=obj_names)


        enable_npy = str2bool(params_rendering['save_depth_and_normal_as_npy'])
        # enable_npy = True
        if 'use_dof' in params_camera and str2bool(params_camera['use_dof']):
            bpy.data.objects["Camera"].data.dof.use_dof = False

        create_normal_map(filepath=out_dir_v, npy=enable_npy)
        create_depth_map(filepath=out_dir_v, npy=enable_npy)

        if 'use_dof' in params_camera and str2bool(params_camera['use_dof']):
            bpy.data.objects["Camera"].data.dof.use_dof = True

        for item in bpy.data.objects:
            item.select_set(state=False)
        for item in obj_names:
            bpy.data.objects[item].select_set(state=True)

        # bpy.ops.transform.rotate(ov, value=label[0], orient_axis='X')
        # bpy.ops.transform.rotate(ov, value=label[1], orient_axis='Y')
        # bpy.ops.transform.rotate(ov, value=label[2], orient_axis='Z')
    bpy.ops.export_mesh.ply(filepath=os.path.join( os.path.abspath(
            out_dir), 'mesh.ply'), use_selection=True)
    camera_intrinsic = get_intrinsic_matrix()
    np.save(os.path.join(os.path.abspath(out_dir), 'camera_intrinsic.npy'), camera_intrinsic)

    return obj_name

# From this line, main starts.
for arg in arglist:
    lines = []
    with open(arg) as f:
        for line in f:
            lines.append(line)
    ini = configparser.ConfigParser()
    ini.optionxform = str
    ini.read(arg)
    rendering = dict(ini.items('rendering'))
    light = dict(ini.items('light'))
    camera = dict(ini.items('camera'))
    object_global = dict(ini.items('object'))
    object_files = [x.strip() for x in ini['settings']
                    ['object_file'].split(',') if not x.strip() == '']
    object_presets = [x.strip() for x in ini['object']
                      ['preset'].split(',') if not x.strip() == '']
    shader_presets = [x.strip() for x in ini['shader']
                      ['preset'].split(',') if not x.strip() == '']
    while len(object_files) != len(object_presets):
        if len(object_files) > len(object_presets):
            object_presets.append(object_presets[-1])
        if len(object_files) < len(object_presets):
            object_presets.pop(-1)
    while len(object_files) != len(shader_presets):
        if len(object_files) > len(shader_presets):
            shader_presets.append(shader_presets[-1])
        if len(object_files) < len(shader_presets):
            shader_presets.pop(-1)
    objects = []
    shaders = []
    for i in range(0, len(object_presets)):
        try:
            object_data = dict(ini.items(object_presets[i]))
        except configparser.NoSectionError:
            object_data = {}
        objects.append(object_data)
    for i in range(0, len(shader_presets)):
        try:
            shader = dict(ini.items(shader_presets[i]))
        except configparser.NoSectionError:
            shader = {}
        shaders.append(shader)

    start = time.time()
    obj_name = render_object(object_shape=ini['settings']['object_file'],
                             params_rendering=rendering,
                             params_camera=camera,
                             params_object_global=object_global,
                             params_object=objects,
                             params_shader=shaders,
                             params_light=light,
                             working_dir=ini['settings']['working_dir'],
                             out_dir=ini['settings']['out_dir'],
                             filename_str = ini['settings']['filename_str'],
                             )



    with open(os.path.join(ini['settings']['out_dir'], obj_name, ini['settings']['filename_str'], 'save.ini'), 'w') as f:
        for i in range(0, len(lines)):
            f.write(lines[i])
    elapsed_time = time.time() - start
    with open(os.path.join(ini['settings']['out_dir'], obj_name, 'time.txt'), 'w') as f:
        f.write("elapsed_time: {:.5f} [sec]".format(elapsed_time))
