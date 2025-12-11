import pybullet as p
import pybullet_data
import obstacles as o
import time

'''
Some initialization of PyBullet simulation:
'''
p.connect(p.GUI) # Connect to physics server (DIRECT or GUI): GUI does physics simulation and rendering at the same time.
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Acces to PyBullet data (models etc.)
p.resetSimulation() # Start with fresh simulation
p.loadURDF('plane.urdf') # Simulation of ground plane at origin [0, 0, 0]
p.setGravity(0, 0, -9.807) # Define gravity

'''
Setting up the world environment
'''
def generate_world(phase):

    # Define all variables needed for the world
    center = [0, 0, 0] # origin
    room_width = 15 # width of the room [m] (x-direction)
    room_depth = 10 # depth of the room [m] (y-direction)
    room_height = 5 # height of the room [m] (z-direction)
    room_color = [0.9, 0.9, 0.9, 0.5] # color of the walls in RGBA (Red, Green, Blue, Alpha (transparency))

    cyl_radius = 0.4 # cross-sectional radius of the cylinders
    cyl_height = room_height # make the cylinders extend through entire room height
    cyl_inset = 3
    cyl_color = [0.9, 0.9, 0.9, 1.0]

    beam_radius = 0.1 # cross-sectional radius of the cross-beams

    if phase == 1:
        '''
        Build the world environment for the first phase: a 2D problem with static circular objects and walls.
        Simulated in PyBullet as a 3D environment with static cylindrical objects and walls.
        '''
        print("--- Generating world for phase 1: ---", "\n", "Building floor, walls and cylinders...")
        # Build the room: four walls and a floor
        room = o.build_room(
            center=center,
            width=room_width,
            depth=room_depth,
            height=room_height,
            color=room_color
        )

        # Build the circular obstacles: fives cylinders in a 'dice 5' shape
        cylinders = o.place_5_cylinders(
            center=center,
            width=room_width,
            depth=room_depth,
            radius=cyl_radius,
            height=cyl_height,
            inset=cyl_inset,
            color=cyl_color
        )

        return room, cylinders
    
    elif phase == 2:
        '''
        Build the world environment for the second phase: a 3D problem with static cylindrical objects and walls.
        Simulated in PyBullet as a 3D environment with static cylindrical objects and walls.
        '''
        print("--- Generating world for phase 2: ---", "\n", "Building floor, walls, cylinders and cross-beams...")
        # Build the room: four walls and a floor
        room = o.build_room(
            center=center,
            width=room_width,
            depth=room_depth,
            height=room_height,
            color=room_color
        )

        # Build the cylindrical obstacles: fives cylinders in a 'dice 5' shape
        cylinders = o.place_5_cylinders(
            center=center,
            width=room_width,
            depth=room_depth,
            radius=cyl_radius,
            height=cyl_height,
            inset=cyl_inset,
            color=cyl_color
        )

        # Build the cylindrical obstacles: four cross-beams
        # From base of front right cylinder to top of back right cylinder
        # From base of front left cylinder to top of back left cylinder
        # From base of front right cylinder to top of front left cylinder
        # From base of back rigth cylinder to top of back left cylinder
        beams_leftright = o.place_cross_beams_leftright(
            cylinders=cylinders,
            cyl_height=cyl_height,
            radius=beam_radius,
            color=cyl_color
        )
        beams_frontback = o.place_cross_beams_frontback(
            cylinders=cylinders,
            cyl_height=cyl_height,
            radius=beam_radius,
            color=cyl_color
        )
        beams = beams_leftright + beams_frontback

        return room, cylinders, beams
    
    elif phase == 3:
        '''
        Build the world environment for the third phase: a 3D problem with dynamic objects, static cylindrical objects and walls.
        Simulated in PyBullet as a 3D environment with static cylindrical objects and walls.
        '''
        print("--- Generating world for phase 3: ---", "\n", "Building floor, walls, cylinders and cross-beams...")
        # Build the room: four walls and a floor
        room = o.build_room(
            center=center,
            width=room_width,
            depth=room_depth,
            height=room_height,
            color=room_color
        )

        # Build the cylindrical obstacles: fives cylinders in a 'dice 5' shape
        cylinders = o.place_5_cylinders(
            center=center,
            width=room_width,
            depth=room_depth,
            radius=cyl_radius,
            height=cyl_height,
            inset=cyl_inset,
            color=cyl_color
        )

        # Build the cylindrical obstacles: four cross-beams
        # From base of front right cylinder to top of back right cylinder
        # From base of front left cylinder to top of back left cylinder
        # From base of front right cylinder to top of front left cylinder
        # From base of back rigth cylinder to top of back left cylinder
        beams_leftright = o.place_cross_beams_frontback(
            cylinders=cylinders,
            cyl_height=cyl_height,
            radius=beam_radius,
            color=cyl_color
        )
        beams_frontback = o.place_cross_beams_frontback(
            cylinders=cylinders,
            cyl_height=cyl_height,
            radius=beam_radius,
            color=cyl_color
        )
        beams = beams_leftright + beams_frontback

        moving = 1 # TODO: implement dynamic obstacle
        return room, cylinders, beams, moving
    
    elif phase not in [1, 2, 3]:
        raise ValueError("Phase variable must be integer 1, 2 or 3")

phase = None
while phase not in [1, 2, 3]:
    try:
        phase = int(input("Phase (1,2,3): "))
    except ValueError:
        print("Phase should be a valid integer!")
world = generate_world(phase)

'''
Running simulation:
(Just some R2D2 dummy falling down for now...)
'''

r2d2 = p.loadURDF('r2d2.urdf', [0, 2, 5]) # Simulation of R2D2 at certain position
time.sleep(2)

for i in range(5000):
    time.sleep(0.005)
    p.stepSimulation()

p.disconnect()