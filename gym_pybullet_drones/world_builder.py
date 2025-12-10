import pybullet as p
import numpy as np

def build_room(
    center=[0, 0, 0],
    width=4.0,
    depth=4.0,
    height=3.0,
    wall_thickness=0.1,
    floor_thickness=0.1,
    color=[0.8, 0.8, 0.8, 1.0],   # RGBA (opacity)
    wall_mass=0,                 # 0 = fixed
):
    """
    Build a rectangular room made of 4 walls + a floor.
    Returns a dictionary with created body IDs.
    """    
    cx, cy, cz = center

    # Create floor
    floor_half = [width/2, depth/2, floor_thickness/2]
    floor_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=floor_half)
    floor_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=floor_half, rgbaColor=color)
    floor_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=floor_collision,
        baseVisualShapeIndex=floor_visual,
        basePosition=[cx, cy, cz - floor_thickness/2]
    )

    # Helper function for creating multiple walls
    def make_wall(size, pos, ori=[0,0,0]):
        hx, hy, hz = size[0]/2, size[1]/2, size[2]/2
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=color)
        quat = p.getQuaternionFromEuler(ori)
        return p.createMultiBody(
            baseMass=wall_mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=quat)

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
    width=4.0,
    depth=4.0,
    radius=0.1,
    height=1.0,
    inset=0.5,                 # how far the corner cylinders are inset
    color=[0.3, 0.3, 0.9, 0.8],
    mass=0                     # 0 = static
):
    """
    Place 5 cylinders in a 'dice 5' pattern:
    - Center
    - Four inset corners
    Returns a list of pybullet body IDs.
    """

    cx, cy, cz = center
    zpos = cz + height/2   # place cylinder on the floor

    # Create shapes once (reuse for all cylinders)
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )

    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )

    def spawn(pos):
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=pos
        )

    bodies = []

    # Center cylinder
    bodies.append(spawn([cx, cy, zpos]))

    # Corner cylinders (dice pattern)
    off_x = width/2 - inset
    off_y = depth/2 - inset

    corners = [
        [cx + off_x, cy + off_y, zpos],   # front-right
        [cx - off_x, cy + off_y, zpos],   # front-left
        [cx + off_x, cy - off_y, zpos],   # back-right
        [cx - off_x, cy - off_y, zpos]    # back-left
    ]

    for c in corners:
        bodies.append(spawn(c))

    return bodies
