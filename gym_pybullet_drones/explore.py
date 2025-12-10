import pybullet as p
import pybullet_data
from dataclasses import dataclass
import time
from world_builder import build_room, place_5_cylinders

p.connect(p.GUI) # Connect to physics server (DIRECT or GUI): GUI does physics simulation and rendering at the same time.
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Acces to PyBullet data (models etc.)

'''
Construct world: room with walls, pillars, etc.
'''
p.resetSimulation() #Start with fresh simulation
p.loadURDF('plane.urdf') # Simulation of ground plane at origin [0, 0, 0]

room_center = [0, 0, 0]
room_width, room_depth, room_height = 15, 10, 5
wall_thickness, floor_thickness = 0.1, 0.1
r_w, g_w, b_w = 0.8, 0.8, 0.8 # grey
a_w = 0.5 # see-through 

build_room(
  center=room_center,
  width=room_width,
  depth=room_depth,
  height=room_height,
  wall_thickness=wall_thickness,
  floor_thickness=floor_thickness,
  color=[r_w, g_w, b_w, a_w],
  wall_mass=0
)

cyl_radius = 0.4
cyl_height = 5
cyl_inset = 3
r_c, g_c, b_c = 0.5, 0.5, 0.5 # grey
a_c = 1.0 # see-through slightly

place_5_cylinders(
  center=room_center,
  width=room_width,
  depth=room_depth,
  radius=cyl_radius,
  height=cyl_height,
  inset=cyl_inset,
  color=[r_c, g_c, b_c, a_c],
  mass=0
)

r2d2 = p.loadURDF('r2d2.urdf', [0, 2, 5]) # Simulation of R2D2 at 0.5 height
print(p.getNumBodies()) # Check how many bodies we've loaded (should be 2)

'''
# Define class for joint info
@dataclass
class Joint:
  index: int
  name: str
  type: int
  gIndex: int
  uIndex: int
  flags: int
  damping: float
  friction: float
  lowerLimit: float
  upperLimit: float
  maxForce: float
  maxVelocity: float
  linkName: str
  axis: tuple
  parentFramePosition: tuple
  parentFrameOrientation: tuple
  parentIndex: int

  def __post_init__(self):
    self.name = str(self.name, 'utf-8')
    self.linkName = str(self.linkName, 'utf-8')

# Let's analyze the R2D2 droid!
print(f"r2d2 unique ID: {r2d2}")
for i in range(p.getNumJoints(r2d2)):
  joint = Joint(*p.getJointInfo(r2d2, i))
  print(joint)
'''
  
# Define gravity
p.setGravity(0, 0, -9.807)

# Run simulation for some steps:
for i in range(200):
  time.sleep(0.2)
  position, orientation = p.getBasePositionAndOrientation(r2d2)
  x, y, z = position
  roll, yaw, pitch = p.getEulerFromQuaternion(orientation)
  print(f"{i:3}: x={x:0.10f}, y={y:0.10f}, z={z:0.10f}), roll={roll:0.10f}, pitch={pitch:0.10f}, yaw={yaw:0.10f}")
  p.stepSimulation()