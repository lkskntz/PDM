from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import obstacles as o

class ObstacleHoverEnv(HoverAviary):
    
    def __init__(self, phase=1, **kwargs):
        assert phase in [1, 2, 3], "phase must be 1, 2, or 3"
        self.phase = phase
        super().__init__(**kwargs)

    def _addObstacles(self): # Called automatically after resetSimulation()
        '''
        Define parameters of obstacles that are needed for the environment
        '''
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

        '''
        Intialize obstacles
        '''
        self.room = None
        self.cylinders = []
        self.beams = []
        self.dyn_obstacles = None

        '''
        Build the world obstacles for the first phase: a 2D problem with static circular objects and walls.
        Simulated in PyBullet as a 3D environment with static cylindrical objects and walls.
        '''
        # Build room
        self.room = o.build_room(
            center=center,
            width=room_width,
            depth=room_depth,
            height=room_height,
            color=room_color
        )

        # Cylinders
        self.cylinders = o.place_5_cylinders(
            center=center,
            width=room_width,
            depth=room_depth,
            radius=cyl_radius,
            height=cyl_height,
            inset=cyl_inset,
            color=cyl_color
        )

        '''
        Build the world environment for the second phase: a 3D problem with static cylindrical objects and walls.
        Simulated in PyBullet as a 3D environment with static cylindrical objects and walls.
        '''
        if self.phase >= 2:
            beams_lr = o.place_cross_beams_leftright(
                cylinders=self.cylinders,
                cyl_height=cyl_height,
                radius=beam_radius
            )
            beams_fb = o.place_cross_beams_frontback(
                cylinders=self.cylinders,
                cyl_height=cyl_height,
                radius=beam_radius
            )
            self.beams = beams_lr + beams_fb
        
        '''
        Build the world obstacles for the third phase: a 3D problem with dynamic objects, static cylindrical objects and walls.
        Simulated in PyBullet as a 3D environment with static cylindrical objects and walls.
        '''
        if self.phase >= 3:
            self.dyn_obstacles = None # TODO: add dynamic obstacles at a later stage
        