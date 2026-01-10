import pybullet as p
import numpy as np

'''
Helper functions for the different obstacles used throughout the project
'''

def build_room(
    center=[0, 0, 0],
    width=4.0,
    depth=4.0,
    height=3.0,
    color=[0.8, 0.8, 0.8, 1.0]):   # RGBA; A: alpha (transparency)

    """
    Build a rectangular room made of 4 walls + a floor.
    Returns a dictionary with created body IDs.
    """    
    cx, cy, cz = center
    wall_thickness = 0.1
    floor_thickness = 0.1
    wall_mass = 0 # Required for creating body, but 0 because it is static

    # Create floor
    floor_half = [width/2, depth/2, floor_thickness/2]
    floor_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=floor_half)
    floor_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=floor_half, rgbaColor=color)
    floor_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=floor_collision,
        baseVisualShapeIndex=floor_visual,
        basePosition=[cx, cy, cz - floor_thickness/2])

    # Helper function for creating multiple walls
    def make_wall(size, pos, origin=[0,0,0]):
        hx, hy, hz = size[0]/2, size[1]/2, size[2]/2
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=color)
        return p.createMultiBody(
            baseMass=wall_mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos)

    wall_ids = {}

    # Create walls
    xwall_size = [width, wall_thickness, height]
    ywall_size = [wall_thickness, depth, height]

    wall_ids["front"] = make_wall(xwall_size, pos=[cx, cy + depth/2, cz + height/2]) # Front wall (positive y)
    wall_ids["back"] = make_wall(xwall_size, pos=[cx, cy - depth/2, cz + height/2]) # Back wall (negative y)
    wall_ids["right"] = make_wall(ywall_size, pos=[cx + width/2, cy, cz + height/2]) # Right wall (positive x)
    wall_ids["left"] = make_wall(ywall_size, pos=[cx - width/2, cy, cz + height/2]) # Left wall (negative x)

    return {"floor": floor_id, "walls": wall_ids}

def place_5_cylinders(
    center=[0,0,0],
    width=4.0,  # width of the room they are to be placed in
    depth=4.0,  # depth of the room they are to be place in
    radius=0.1, # cross-sectional radius of the cylinders
    height=1.0, # height of the cylinders
    inset=0.5,  # how far the corner cylinders are inset into the room
    color=[0.8, 0.8, 0.8, 1.0]):

    """
    Place 5 cylinders in a 'dice 5' pattern:
    - Center
    - Four inset corners
    Returns a list of pybullet body IDs.
    """

    cx, cy, cz = center
    zpos = cz + height/2   # place cylinder on the floor
    mass = 0 # 0 because static

    # Create shapes once (reuse for all cylinders)
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height)

    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color)

    def build_cyl(pos):
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=pos)

    cylinders = []

    # Center cylinder
    cylinders.append(build_cyl([cx, cy, zpos]))

    # Corner cylinders (dice pattern)
    off_x = width/2 - inset
    off_y = depth/2 - inset

    corners = [
        [cx + off_x, cy + off_y, zpos],   # front-right
        [cx - off_x, cy + off_y, zpos],   # front-left
        [cx + off_x, cy - off_y, zpos],   # back-right
        [cx - off_x, cy - off_y, zpos]]    # back-left

    for c in corners:
        cylinders.append(build_cyl(c))

    return cylinders


def place_cross_beams_leftright(
        cylinders=[],
        cyl_height=1.0,
        radius=0.05,
        color=[0.8, 0.8, 0.8, 1.0]):
    
    '''
    Creates a diagonal cross-beam on the sides:
    - base of front-right to top of back-right
    - base of front-left to top of back-left

    Using cylinders from place_5_cylinders()

    '''

    def compute_beam(start, end):
        vec = np.array(end) - np.array(start)
        length = np.linalg.norm(vec)
        center = (np.array(start) + np.array(end)) / 2

        # Compute rotation of beam vector, because cylinders are vertical (0, 0, 1) by default
        u = np.array([0, 0, 1]) # input vector: default z-axis
        v = vec / length # output vector: direction of beam that quaternion should rotate it to
        axis = np.cross(u, v)
        axis_length = np.linalg.norm(axis)
        if axis_length < 1e-6: # Avoid division by 0, when u and v are almost parallel, no rotation needed
            print("no rotation needed")
            return center.tolist(), [0, 0, 0, 1], length
        else:
            unit_axis = axis / np.linalg.norm(axis)

            angle = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
            qw = np.cos(angle / 2)
            qx = unit_axis[0] * np.sin(angle / 2)
            qy = unit_axis[1] * np.sin(angle / 2)
            qz = unit_axis[2] * np.sin(angle / 2)
            quat = [qx, qy, qz, qw]
            return center.tolist(), quat, length
    
    # Retrieve base positions and compute start- end-coordinates 
    fr_pos = p.getBasePositionAndOrientation(cylinders[1])[0]
    br_pos = p.getBasePositionAndOrientation(cylinders[3])[0]
    fl_pos = p.getBasePositionAndOrientation(cylinders[2])[0]
    bl_pos = p.getBasePositionAndOrientation(cylinders[4])[0]

    front_right_base = [fr_pos[0], fr_pos[1], fr_pos[2] + cyl_height/2]
    back_right_top = [br_pos[0], br_pos[1], br_pos[2] - cyl_height/2]
    front_left_base = [fl_pos[0], fl_pos[1], fl_pos[2] + cyl_height/2]
    back_left_top = [bl_pos[0], bl_pos[1], bl_pos[2] - cyl_height/2]

    right_beam_vec = compute_beam(front_right_base, back_right_top)
    left_beam_vec = compute_beam(front_left_base, back_left_top)
    beam_vecs = [right_beam_vec, left_beam_vec]

    mass = 0 # 0 because static

    def build_beam(beam_vec):
        center, quat, length = beam_vec
        # Create shapes once (reuse for all cylinders)
        collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=length)

        visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=color)

        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=center,
            baseOrientation=quat)
    
    beams = []
    for vec in beam_vecs:
        beams.append(build_beam(vec))
    return beams

def place_cross_beams_frontback(
        cylinders=[],
        cyl_height=1.0,
        radius=0.05,
        color=[0.8, 0.8, 0.8, 1.0]):
    
    '''
    Creates a diagonal cross-beam on the front and back:
    - base of front-left to top of front-right
    - base of back-left to top of back-right

    Using cylinders from place_5_cylinders()
    '''

    def compute_beam(start, end):
        vec = np.array(end) - np.array(start)
        length = np.linalg.norm(vec)
        center = (np.array(start) + np.array(end)) / 2

        # Compute rotation of beam vector, because cylinders are vertical (0, 0, 1) by default
        u = np.array([0, 0, 1]) # input vector: default z-axis
        v = vec / length # output vector: direction of beam that quaternion should rotate it to
        axis = np.cross(u, v)
        axis_length = np.linalg.norm(axis)
        if axis_length < 1e-6: # Avoid division by 0, when u and v are almost parallel, no rotation needed
            print("no rotation needed")
            return center.tolist(), [0, 0, 0, 1], length
        else:
            unit_axis = axis / np.linalg.norm(axis)

            angle = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
            qw = np.cos(angle / 2)
            qx = unit_axis[0] * np.sin(angle / 2)
            qy = unit_axis[1] * np.sin(angle / 2)
            qz = unit_axis[2] * np.sin(angle / 2)
            quat = [qx, qy, qz, qw]
            return center.tolist(), quat, length
    
    # Retrieve base positions and compute start- end-coordinates 
    fr_pos = p.getBasePositionAndOrientation(cylinders[1])[0]
    br_pos = p.getBasePositionAndOrientation(cylinders[3])[0]
    fl_pos = p.getBasePositionAndOrientation(cylinders[2])[0]
    bl_pos = p.getBasePositionAndOrientation(cylinders[4])[0]

    front_right_top = [fr_pos[0], fr_pos[1], fr_pos[2] - cyl_height/2]
    back_right_top = [br_pos[0], br_pos[1], br_pos[2] - cyl_height/2]
    front_left_base = [fl_pos[0], fl_pos[1], fl_pos[2] + cyl_height/2]
    back_left_base = [bl_pos[0], bl_pos[1], bl_pos[2] + cyl_height/2]

    front_beam_vec = compute_beam(front_left_base, front_right_top)
    back_beam_vec = compute_beam(back_left_base, back_right_top)
    beam_vecs = [front_beam_vec, back_beam_vec]

    mass = 0 # 0 because static

    def build_beam(beam_vec):
        center, quat, length = beam_vec
        # Create shapes once (reuse for all cylinders)
        collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=length)

        visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=color)

        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=center,
            baseOrientation=quat)
    
    beams = []
    for vec in beam_vecs:
        beams.append(build_beam(vec))
    return beams

# TODO: implement dynamic obstacles
def place_dynamic_sphere():
    dyn_obstacles = None
    return dyn_obstacles