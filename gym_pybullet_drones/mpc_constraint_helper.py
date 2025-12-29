import numpy as np

def get_vertical_pillar_constraint(drone_pos, pillar_pos_2d, radius_pillar, radius_drone, margin):
    """
    Generates a linear constraint for the vertical cylindrical pillar (one of them).
    
    :param drone_pos: [x, y, z] current position of the drone (np.array)
    :param pillar_pos_2d: [x, y] center coordinates of the pillar (np.array)
    :param radius_pillar: radius of the pillar (float)
    :param radius_drone: radius of the drone (a collision sphere/circle) (float)
    :param margin: safety margin between drone and obstacle (float)

    Returns:
        n: [nx, ny, nz]: Normal vector from pillar center to drone center (np.array)
        a: scalar bound (float)
    """

    # Get 2D coordinates
    drone_pos_2d = drone_pos[0:2]

    # Vector from pillar center to drone
    d = drone_pos_2d - pillar_pos_2d
    dist = np.linalg.norm(d)
    n_hat = d / dist # normalize

    # Calculate boundary point on the edge of the 'safe zone' (pillar including drone radius and margin)
    total_radius = radius_pillar + radius_drone + margin
    p_bound_2d = pillar_pos_2d + total_radius * n_hat

    # Formulate the 3D constraint (z-component 0 because pillar is vertical)
    # So the constraint is: n^T p_k > a
    n = np.array([n_hat[0], n_hat[1], 0.0])
    a = np.dot(n_hat, p_bound_2d)

    return n, a


def get_crossbeam_constraint(drone_pos, p_base, p_top, radius_beam, radius_drone, margin):
    """
    Generates a linear constraint for the supporting cross-beams (one of them)
    
    :param drone_pos: [x, y, z] current position of the drone (np.array)
    :param p_base: [x, y, z] coordinate of the start of the cross-beam (np.array)
    :param p_top: [x, y, z] coordinates of the end of the cross-beam (np.array)
    :param radius_beam: radius of the beam (float)
    :param radius_drone: radius of the drone (a collision sphere/circle) (float)
    :param margin: safety margin between drone and obstacle (float)

    Returns:
        n: [nx, ny, nz]: Normal vector from beam center to drone center (np.array)
        a: scalar bound (float)
    """

    # Define beam vector
    vec_beam = p_top - p_base
    len_beam = np.linalg.norm(vec_beam)

    # Project drone position onto the line segment
    # t represents how far along the beam the drone is (t=0.0: drone is at the base, t=1.0: drone is at the top)
    t = np.dot(drone_pos - p_base, vec_beam) / len_beam
    t_clamped = np.clip(t, 0.0, 1.0)

    # Find closest point on centerline
    closest_point_centerline = p_base + t_clamped * vec_beam

    # Vector from closest point on line to drone
    d = drone_pos - closest_point_centerline
    dist = np.linalg.norm(d)
    n_hat = d / dist # normalize

    # Calculate boundary point on the edge of the 'safe zone' (beam including drone radius and margin)
    total_radius = radius_beam + radius_drone + margin
    p_bound = closest_point_centerline + total_radius * n_hat

    # Formulate the 3D constraint (z-component is not 0 here, so n is just n_hat)
    # So the constraint is: n^T p_k > a
    n = n_hat
    a = np.dot(n_hat, p_bound)
    return n, a