import bpy
import os
import math, random
from typing import List, Tuple
import time
import numpy as np

def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))



def generate_shape_description(points, height):
    num_points = len(points)
    vertices = list(map(lambda x: (x[0], x[1], 0), points)) + list(map(lambda x: (x[0], x[1], height), points))
    edges = []
    for i in range(num_points):
        edges.append( (i, (i + 1) % num_points) )
    for i in range(num_points):
        edges.append( (i + num_points, (i + 1) % num_points + num_points) )
    for i in range(num_points):
        edges.append( (i, i + num_points) )
    
    faces = [
        tuple( range(num_points) ),
        tuple( map( lambda x: x + num_points, range(num_points)) ),
    ]
    
    for i in range(num_points):
        faces.append( ( i, (i + 1) % num_points, (i + 1) % num_points + num_points, i + num_points ) )
    
    return vertices, edges, faces

def generate_shape(points, height):
    vertices, edges, faces = generate_shape_description(points, height)
    new_mesh = bpy.data.meshes.new('new_mesh')
    new_mesh.from_pydata(vertices, edges, faces)
    new_mesh.update()
    new_object = bpy.data.objects.new('new_object', new_mesh)
    return new_object, vertices, edges, faces

def gen_base(max_coord):
    vertices = [ (-max_coord, -max_coord, 0), (-max_coord, max_coord, 0), (max_coord, max_coord, 0), (max_coord, -max_coord, 0) ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    faces = [(0, 1, 2, 3)]
    new_mesh = bpy.data.meshes.new('new_base')
    new_mesh.from_pydata(vertices, edges, faces)
    new_mesh.update()
    new_object = bpy.data.objects.new('new_base', new_mesh)
    return new_object, vertices, edges, faces
    
def create_uv(obj):
    uv = obj.data.uv_layers.new(name="UVMap")
    context = bpy.context
    scene = context.scene
    vl = context.view_layer
    bpy.ops.object.select_all(action='DESELECT')
    vl.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT') # for all faces
    bpy.ops.uv.smart_project(angle_limit=66, island_margin = 0.02)
    bpy.ops.object.editmode_toggle()
    obj.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    

def gen_path(points, type="POLY"):
    vertices = list(map(lambda x: (x[0], x[1], 0), points))
    vertices.append(vertices[0])
    # create the Curve Datablock
    curveData = bpy.data.curves.new('myCurve', type='CURVE')
    curveData.dimensions = '2D'
    curveData.resolution_u = 32

    # map coords to spline
    polyline = curveData.splines.new(type)
    polyline.points.add(len(vertices) - 1)
    for i, coord in enumerate(vertices):
        x,y,z = coord
        polyline.points[i].co = (x, y, z, 1)

    # create Object
    curveOB = bpy.data.objects.new('myCurve', curveData)

    return curveOB, vertices




def gen_camera():
    lamp_data = bpy.data.lights.new(name="Lamp", type='AREA')
    lamp_data.shape = "ELLIPSE"
    lamp_data.energy = 0.1
    lamp_data.size = 0.01
    lamp_data.size_y = 0.01
    lamp_data.specular_factor = 0
    lamp_data.diffuse_factor = 0.1
    lamp_object = bpy.data.objects.new(name="Lamp", object_data=lamp_data)
    lamp_object.location = (0, 0, 0)
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    camera_object.data.lens_unit = 'FOV'
    #camera_object.data.angle = 0.0401426
    camera_object.data.angle = 0.200713
    camera_object.data.clip_start = 0.001
    

    lamp_object.parent = camera_object
    return camera_object, lamp_object

def clamp_to_path(obj, path):
    obj.constraints.new(type='CLAMP_TO')
    obj.constraints["Clamp To"].target = path
    obj.constraints["Clamp To"].main_axis = "CLAMPTO_X"
    bpy.context.view_layer.update()


def set_texture(ob, repeated=10, img="Y:\\blender_prj\\IMG_0.png"):
    mat = bpy.data.materials.new(name="New_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    mapping = mat.node_tree.nodes.new('ShaderNodeMapping')
    mapping.inputs[3].default_value[0] = repeated
    mapping.inputs[3].default_value[1] = repeated
    mapping.inputs[3].default_value[2] = repeated
    coord = mat.node_tree.nodes.new('ShaderNodeTexCoord')
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(img)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    mat.node_tree.links.new(texImage.inputs['Vector'], mapping.outputs['Vector'])
    mat.node_tree.links.new(mapping.inputs['Vector'], coord.outputs['UV'])
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)

def set_color_red(ob):
    mat = bpy.data.materials.new(name="Red_Mat")
    mat.use_nodes = False
    mat.diffuse_color = (1, 0, 0, 1)
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)
        
def set_color_blue(ob):
    mat = bpy.data.materials.new(name="Blue_Mat")
    mat.use_nodes = False
    mat.diffuse_color = (0, 0, 1, 1)
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)
        
def set_color_red(ob, repeated=10):
    mat = bpy.data.materials.new(name="Red_Mat")
    mat.use_nodes = False
    mat.diffuse_color = (1, 0, 0, 1)
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)
        
def set_color_green(ob, repeated=10):
    mat = bpy.data.materials.new(name="Red_Mat")
    mat.use_nodes = False
    mat.diffuse_color = (0, 1, 0, 1)
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)

def get_world_pos(obj):
    m = obj.matrix_world
    return (m[0][3], m[1][3], m[2][3])
    
def check_pos_square(obj, delta, rad):
    pos = get_world_pos(obj)
    old_delta = obj.location[0]
    obj.location[0] += delta
    update()
    pos2 = get_world_pos(obj)
    obj.location[0] = old_delta
    update()
    return abs(pos[0] - pos2[0]) >= rad or abs(pos[1] - pos2[1]) >= rad or abs(pos[2] - pos2[2]) >= rad

def update():
    for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                ctx = {
                    "window": bpy.context.window, # current window, could also copy context
                    "area": area, # our 3D View (the first found only actually)
                    "region": None # just to suppress PyContext warning, doesn't seem to have any effect
                }
                bpy.ops.view3d.view_axis(ctx, type='TOP', align_active=True)
                area.spaces.active.region_3d.update()  # <---

def check_pos_not_moving(obj, delta):
    pos = get_world_pos(obj)
    old_delta = obj.location[0]
    obj.location[0] += delta
    update()
    pos2 = get_world_pos(obj)
    obj.location[0] = old_delta
    update()
    return pos[0] == pos2[0] and pos[1] == pos2[1] and pos[2] == pos2[2]

def binsearch(func, left, right, accuracy):
    l = left
    r = right
    while l < r - accuracy:
        m = (l + r) / 2
        if func(m):
            r = m
        else:
            l = m
    return r

def reset_camera_pos(obj):
    obj.location[0] = -10
    update()
    obj.location[0] += binsearch(lambda x: not check_pos_not_moving(obj, x), 0, 10, 0.000001)
    update()
    
def get_next_camera_pos(obj, delta, accuracy):
    return binsearch(lambda x: check_pos_square(obj, x, delta), 0, 10 * delta, 0.000001)

def move_camera_by_delta_any_axies(obj, delta):
    prev = get_world_pos(obj)
    to_move = get_next_camera_pos(obj, delta, 0.000001)
    print("Moving: ", to_move)
    obj.location[0] += to_move
    update()
    curr = get_world_pos(obj)
    print("Moved obj by:", (prev[0] - curr[0], prev[1] - curr[1], prev[2] - curr[2]))
    return to_move

def render_and_save_image(camera, name, path="Y:/blender_prj"):
    bpy.data.scenes['Scene'].camera = camera
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = path + '/' + name
    bpy.ops.render.render(write_still = 1)
    
def mesh_from_cruve(obj):
    context = bpy.context
    scene = context.scene
    vl = context.view_layer
    bpy.ops.object.select_all(action='DESELECT')
    vl.objects.active = obj
    obj.select_set(True)
    
    
    bpy.ops.object.convert(target='MESH', keep_original=True)
    new_obj = bpy.context.object
    
    obj.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    
    return new_obj

def delete_object(obj):
    context = bpy.context
    scene = context.scene
    vl = context.view_layer
    bpy.ops.object.select_all(action='DESELECT')
    vl.objects.active = obj
    obj.select_set(True)
    
    bpy.ops.object.delete(use_global=False, confirm=False)
    
    bpy.ops.object.select_all(action='DESELECT')

def points_from_mesh(obj):
    ret = []
    for v in obj.data.vertices:
        ret.append( (v.co[0], v.co[1]) )
    return ret

def update_path_pos(obj, height):
    # dunno why this is needed, but it dosent update object otherwise
    obj.data.dimensions = '2D'
    obj.location[2] = height
    
def create_sphere():
    bpy.ops.mesh.primitive_uv_sphere_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(0.00001, 0.00001, 0.00001))
    return bpy.context.object

def create_data_file(file_location, file_name):
    bpy.ops.image.open(filepath=file_location + '\\' + file_name)
    img = bpy.data.images[file_name]
    acc_delta = 0.2
    ret = [[], [], []]
    pixels = np.array(img.pixels)
    size = (round(len(pixels) / 4), 4)
    pixels = pixels.reshape(size)[:, 0:3]
    m = np.mean(pixels, 1)
    pixels = pixels - np.concatenate((m, m, m)).reshape((3, len(pixels))).T
    pixels = np.argwhere(pixels > acc_delta)
    for px in pixels:
        ret[px[1]].append(px[0])
    #for i in range(len(pixels)):
    #for i, pixel in enumerate(pixels):
        #pixel = pixels[i]
        #pixel = np.array(img.pixels[i * 4: i * 4 + 3])
        #pixel = (img.pixels[i * 4], img.pixels[i * 4 + 1], img.pixels[i * 4+ 2])
        # mean = np.sum(pixel) / 3
        #if np.max(pixel) > mean + acc_delta:
        #    ret[np.argmax(pixel)].append(i)
        # pixel = pixel - mean
        #for k in range(3):
        #    if pixel[k] > acc_delta:
        #        ret[k].append(i)
        #        break
        #mean = np.mean(pixel)
        #if np.max(pixel) > mean + acc_delta:
        #    ret[np.argmax(pixel)].append(i)
    bpy.data.images.remove(img)
    for i in range(len(ret)):
        ret[i] = list(map(lambda x: (round(x % bpy.context.scene.render.resolution_x), bpy.context.scene.render.resolution_y - round(x / bpy.context.scene.render.resolution_x)), ret[i]))
        acc = [0, 0]
        for u in ret[i]:
            acc[0] += u[0]
            acc[1] += u[1]
        acc[0] = 0 if len(ret[i]) == 0 else round(acc[0] / len(ret[i]))
        acc[1] = 0 if len(ret[i]) == 0 else round(acc[1] / len(ret[i]))
        ret[i] = acc
    return ret

def init_mov_scene(camera, camera_path, sphere_1, sphere_2, sphere_3, sphere_path, delta, path="Y:/blender_prj"):
    reset_camera_pos(camera)
    sphere_1.location = camera.location.copy()
    sphere_2.location = camera.location.copy()
    sphere_3.location = camera.location.copy()
    
    mov_1 = get_next_camera_pos(sphere_1, delta, delta / 1000)
    
    sphere_1.location[0] += mov_1
    sphere_2.location[0] += mov_1
    camera.location[0] += mov_1
    update()
    
    mov_2 = get_next_camera_pos(sphere_1, delta, delta / 1000)
    
    sphere_1.location[0] += mov_1
    update()
    
    for i in range(0):
        start = time.time()
        sphere_1.hide_render = False
        sphere_2.hide_render = False
        sphere_3.hide_render = False
        update()
        print("upd1", time.time()-start)
        start = time.time()
        render_and_save_image(camera, "dots_" + str(i) + ".png", path=path)
        print("rndr1", time.time()-start)
        start = time.time()
        sphere_1.hide_render = True
        sphere_2.hide_render = True
        sphere_3.hide_render = True
        update()
        print("upd2", time.time()-start)
        start = time.time()
        render_and_save_image(camera, "clean_" + str(i) + ".png", path=path)
        print("rndr2", time.time()-start)
        start = time.time()
        pos = create_data_file(path, "dots_" + str(i) + ".png")
        f = open(path + "/pos_" + str(i) + ".txt", "w")
        f.write(str(pos[0][0]) + ',' + str(pos[0][1]) + '\n' + str(pos[1][0]) + ',' + str(pos[1][1]) + '\n' + str(pos[2][0]) + ',' + str(pos[2][1]))
        f.close()
        print("sv", time.time()-start)
        start = time.time()
        mov_1, mov_2 = move_iteration(camera, sphere_1, sphere_2, sphere_3, mov_1, mov_2, delta)
        print("mv", time.time()-start)
        start = time.time()
        

def move_iteration(camera, sphere_1, sphere_2, sphere_3, mov_prev, mov_curr, delta):
    next_move = get_next_camera_pos(sphere_1, delta, delta / 1000)
    sphere_1.location[0] += next_move
    
    sphere_2.location[0] += mov_curr
    camera.location[0] += mov_curr
    
    sphere_3.location[0] += mov_prev
    
    update()
    
    return mov_curr, next_move
    
def delete_all():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    bpy.ops.outliner.orphans_purge()
    bpy.ops.outliner.orphans_purge()
    bpy.ops.outliner.orphans_purge()
    
    for c in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.unlink(c)

def gen_scene(num_points, seed=0, exec_path="Y:/blender_prj"):
    random.seed(seed)
    
    bpy.context.scene.render.resolution_x = 320
    bpy.context.scene.render.resolution_y = 240

    
    
    points = generate_polygon(center=(0, 0),
                                avg_radius=0.08,
                                irregularity=0.4,
                                spikiness=0.1,
                                num_vertices=num_points)
                                
                                
    path, _ = gen_path(points, type="POLY" if seed % 2 else 'NURBS')
    
    new_collection = bpy.data.collections.new('new_collection')
    bpy.context.scene.collection.children.link(new_collection)
    new_collection.objects.link(path)
    
    t_mesh = mesh_from_cruve(path)
    
    path_points = points_from_mesh(t_mesh)
    
    delete_object(t_mesh)
    
    path_for_dot, _ = gen_path(points, type="POLY" if seed % 2 else 'NURBS')
    
    shape, _, _, _ = generate_shape(path_points, height=0.003)
    base, _, _, _ = gen_base(0.2)
    camera, lamp = gen_camera();
    
    sphere_1 = create_sphere()
    sphere_2 = create_sphere()
    sphere_3 = create_sphere()
    
    new_collection.objects.link(path_for_dot)
    new_collection.objects.link(shape)
    new_collection.objects.link(camera)
    new_collection.objects.link(lamp)
    new_collection.objects.link(base)

    clamp_to_path(camera, path)
    clamp_to_path(sphere_1, path_for_dot)
    clamp_to_path(sphere_2, path_for_dot)
    clamp_to_path(sphere_3, path_for_dot)
    update_path_pos(path, 0.02)
    update_path_pos(path_for_dot, 0.003)
    
    create_uv(shape)
    create_uv(base)
    
    base_textures = [f for f in os.listdir(exec_path + '/base_texture')]
    shape_textures = [f for f in os.listdir(exec_path + '/shape_texture')]
    set_texture(shape, 80, img=exec_path + '/shape_texture/' + shape_textures[seed % len(shape_textures)])
    set_texture(base, 160, img=exec_path + '/base_texture/' + base_textures[seed % len(base_textures)])
    set_color_red(sphere_1)
    set_color_green(sphere_2)
    set_color_blue(sphere_3)
    
    if ( seed % 5 == 1 ):
        path.location[0] -= random.random() * 0.00006
    if ( seed % 5 == 2 ):
        path.location[0] += random.random() * 0.00006
    if ( seed % 5 == 3 ):
        path.location[1] -= random.random() * 0.00006
    if ( seed % 5 == 4 ):
        path.location[1] += random.random() * 0.00006
    
    init_mov_scene(camera, path, sphere_1, sphere_2, sphere_3, path_for_dot, 0.001, path=exec_path + '/' + str(seed))

    

#delete_all()
#gen_scene(128, 0)
delete_all()
gen_scene(128, 1)
