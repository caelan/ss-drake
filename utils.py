from __future__ import absolute_import, division, print_function

import os
import numpy as np
import pydrake
from pydrake.multibody.parsers import PackageMap
from pydrake.multibody.rigid_body_tree import (
    AddModelInstanceFromUrdfStringSearchingInRosPackages,
    AddModelInstancesFromSdfString,
    AddModelInstancesFromSdfStringSearchingInRosPackages,
    FloatingBaseType,
    RigidBodyFrame,
    RigidBodyTree,
    AddFlatTerrainToWorld,
)

from pydrake.multibody.rigid_body_plant import RigidBodyPlant, DrakeVisualizer, \
  CompliantMaterial, CompliantContactModelParameters
from pydrake.lcm import DrakeMockLcm, DrakeLcm, DrakeLcmInterface
from pydrake.systems.framework import BasicVector, DiagramBuilder
# TODO(eric.cousineau): Use `unittest` (after moving `ik` into `multibody`),
# declaring this as a drake_py_unittest in the BUILD.bazel file.

X_POSITION = 'base_x'
Y_POSITION = 'base_y'
Z_POSITION = 'base_z'
ROLL_POSITION = 'base_roll'
PITCH_POSITION = 'base_pitch'
YAW_POSITION = 'base_yaw'

POINT_POSITIONS = [X_POSITION, Y_POSITION, Z_POSITION]
EULER_POSITIONS = [ROLL_POSITION, PITCH_POSITION, YAW_POSITION]
POSE_POSITIONS = POINT_POSITIONS + EULER_POSITIONS
POSE2D_POSITIONS = [X_POSITION, Y_POSITION, YAW_POSITION]

def Point(x=0, y=0, z=0):
    return np.array([x, y, z])

def Euler(roll=0, pitch=0, yaw=0):
    return np.array([roll, pitch, yaw])

def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return np.concatenate([point, euler])

def point_from_pose(pose):
    return pose[:3]

def euler_from_pose(pose):
    return pose[3:]

def point_euler_from_pose(pose):
    return point_from_pose(pose), euler_from_pose(pose)

##################################################

def get_num_joints(tree):
    return tree.get_num_positions() # number_of_positions

def get_joint_name(tree, joint_id):
    return str(tree.get_position_name(joint_id))

def get_joint_names(tree, joint_ids):
    return [get_joint_name(tree, joint_id) for joint_id in joint_ids]

def get_joint_id(tree, joint_name, model_id=-1):
    return tree.findJointId(joint_name, model_id=model_id)

def get_joint_ids(tree, joint_names, model_id=-1):
    return [get_joint_id(tree, joint_name, model_id) for joint_name in joint_names]

def get_min_position(tree, joint_id):
    return tree.joint_limit_min[joint_id]

def get_max_position(tree, joint_id):
    return tree.joint_limit_max[joint_id]

##################################################

def get_world(tree):
    #assert(tree.world() is tree.get_body(0))
    return tree.world()

def get_num_models(tree):
    return tree.get_num_model_instances()

def get_body_name(tree, body_id):
    return tree.getBodyOrFrameName(body_id)

def get_frame_name(tree, frame_id):
    return tree.getBodyOrFrameName(frame_id)

def get_body(tree, body_id):
    return tree.get_body(body_id)

def get_bodies(tree, model_id=-1):
    if model_id != -1:
        return tree.FindModelInstanceBodies(model_id)
    return [get_body(tree, body_id) for body_id in range(tree.get_num_bodies())]

def get_base_bodies(tree, model_id=-1):
    return [get_body(tree, body_id) for body_id in tree.FindBaseBodies(model_id=model_id)]

def get_frame(tree, frame_id):
    return get_body(tree, frame_id).

def get_body_from_name(tree, body_name, model_id=-1):
    return tree.FindBody(body_name, model_id=model_id)

def get_frame_from_name(tree, frame_name, model_id=-1):
    return tree.findFrame(frame_name, model_id=model_id)

def get_base_body(tree, model_id):
    base_bodies = get_bodies(tree, model_id)
    #assert(len(base_bodies) == 1) # TODO: should make formal assumption that one root per URDF
    return base_bodies[0]

def get_model_name(tree, model_id):
    return str(get_base_body(tree, model_id).get_model_name())


def is_fixed(tree, model_index):
    base_body = base_bodies_from_model_index(tree, model_index)[0]
    return base_body.IsRigidlyFixedToWorld()

##################################################

def add_model(tree, model_file, pose=None, fixed=True):
    model_string = open(model_file).read()
    base_dir = os.path.dirname(model_file)
    package_map = PackageMap()

    base_type = FloatingBaseType.kFixed if fixed else FloatingBaseType.kRollPitchYaw
    name = 'frame' # TODO: body name is always frame
    pose = Pose() if pose is None else pose
    weld_frame = RigidBodyFrame(name, get_world(tree), pose) # point, euler

    if model_file.endswith('.urdf'):
        AddModelInstanceFromUrdfStringSearchingInRosPackages(
            model_string, package_map,
            base_dir, base_type,
            weld_frame, tree)
    elif model_file.endswith('.sdf'):
        AddModelInstancesFromSdfStringSearchingInRosPackages(
          model_string, package_map, 
          base_type, weld_frame, tree)
        #AddModelInstancesFromSdfString(model_string, base_type, weld_frame, tree) 
    else:
        raise ValueError(model_file)
    model_index = tree.get_num_model_instances() - 1
    return model_index

##################################################

class DrakeVisualizerHelper:
    def __init__(self, tree, default_q=None):
        lcm = DrakeLcm()
        self.tree = tree
        self.visualizer = DrakeVisualizer(tree=self.tree, lcm=lcm, enable_playback=True)
        if default_q is None:
            default_q = self.tree.getZeroConfiguration()
        self.x = np.concatenate([default_q, np.zeros(tree.get_num_velocities())])
        self.visualizer.PublishLoadRobot()
        self.draw(default_q)

    def draw(self, q=None):
        if q is not None:
            self.x[:self.tree.get_num_positions()] = q # Mutates
        context = self.visualizer.CreateDefaultContext()
        context.FixInputPort(0, BasicVector(self.x))
        self.visualizer.Publish(context)

##################################################

# Body names don't have to be unique

def get_frame_indices(tree):
    return range(tree.get_num_frames())

def get_body_indices(tree):
    return range(tree.get_num_bodies())

def body_name_from_index(tree, index):
    return tree.getBodyOrFrameName(index)

frame_name_from_index = body_name_from_index # frame_name == body_name

def body_from_index(tree, index):
    return tree.get_body(index)

def get_bodies(tree):
    return [body_from_index(tree, index) for index in get_body_indices(tree)]

def get_body_names(tree):
    return [body.get_name() for body in get_bodies(tree)]
    #return [tree.getBodyOrFrameName(index) for index in get_body_indices(tree)]

get_frame_names = get_body_names

def body_from_name(tree, name):
    return tree.FindBody(name)

def frame_from_name(tree, name):
    return tree.findFrame(name)

def name_from_body(body):
    return body.get_name()

get_name = name_from_body
name_from_frame = name_from_body

def index_from_frame(frame):
    return frame.get_frame_index()

def get_world(tree):
    #assert(tree.world() is tree.get_body(0))
    return tree.world()

def get_model_indices(tree):
    return range(tree.get_num_model_instances())

def bodies_from_model_index(tree, model_index):
    return tree.FindModelInstanceBodies(model_index)

def base_bodies_from_model_index(tree, model_index=-1):
    return [body_from_index(tree, body_index) for body_index in tree.FindBaseBodies(model_id=model_index)]

def get_model_base(tree, model_index):
    base_bodies = base_bodies_from_model_index(tree, model_index)
    #assert(len(base_bodies) == 1) # TODO: should make formal assumption that one root per URDF
    # This is also why the pybullet URDF didn't work
    return base_bodies[0]

def get_pose_positions(tree, model_index):
    base_body = get_model_base(tree, model_index)
    return range(base_body.get_position_start_index(), 
        base_body.get_position_start_index() + base_body.get_num_positions())

def model_name_from_index(tree, model_index):
    bodies = base_bodies_from_model_index(tree, model_index)
    #print(map(get_name, bodies)) # [u'world_link', u'r_gripper_l_finger_tip_frame', u'l_gripper_l_finger_tip_frame']
    return bodies[0].get_model_name()

def is_fixed(tree, model_index):
    base_body = base_bodies_from_model_index(tree, model_index)[0]
    return base_body.IsRigidlyFixedToWorld()

def get_pose(tree, q, model_index):
    positions = get_pose_positions(tree, model_index)
    return q[positions]

def set_pose(tree, q, model_index, pose):
    positions = get_pose_positions(tree, model_index)
    q[positions] = pose

##################################################


def print_tree_info(robot):
    print("Models:", robot.get_num_model_instances()) # Doesn't include manually added
    print("Positions:", robot.get_num_positions()) # 34 | number_of_positions
    print("Velocities:", robot.get_num_velocities()) # 34 | number_of_velocities
    print("Actuators:", robot.get_num_actuators()) # 28
    print("Frames:", robot.get_num_frames()) # 86
    print("Bodies:", robot.get_num_bodies()) # 86
    world = robot.world()
    print("Tree:", robot) # RigidBodyTree
    print("World:", world) # RigidBody
    print(world.get_name(), "elements:", world.get_visual_elements()) # List of visual elements

    q0 = robot.getZeroConfiguration()
    for index in get_model_indices(robot):
        print(index, model_name_from_index(robot, index), is_fixed(robot, index), get_pose(robot, q0, index))

    for body in get_bodies(robot):
        print(body.get_name(), body.get_position_start_index(), body.get_num_positions(), #body.get_joint_name(),
            body.has_joint(), body.has_parent_body(), body.IsRigidlyFixedToWorld(), body.get_parent() == robot.world()) #, body.get_parent()) # None if no parent


    kin_cache = get_kin_cache(robot, robot.getZeroConfiguration())
    print(kin_cache)
    #for body_index in get_body_indices(robot):
    #    body = robot.get_body(body_index)
    #    print(body_index, robot.getBodyOrFrameName(body_index), body.get_name())
    #    print(get_world_pose(robot, kin_cache, body_index))

    # Geometry
    # getFaces
    # getShape
    # getPoints
    # getBoundingBoxPoints

    # Element
    # getGeometry
    # getLocalTransform

    # RigidBody
    # get_visual_elements

def get_kin_cache(tree, q):
    return tree.doKinematics(q)

def get_world_pose(tree, kin_cache, body_index):
    base_index = 0
    #return tree.relativeTransform(kin_cache, base_index, body_index)
    return tree.CalcBodyPoseInWorldFrame(kin_cache, body_from_index(tree, body_index))

# TODO: self-collisions for pr2
# TODO: filter collisions

def sample_configuration(tree, positions=None):
    q = tree.getRandomConfiguration()
    if positions is None:
        return q
    return q[positions]

PR2_PATH = "examples/pr2/models/pr2_description/urdf/pr2_simplified.urdf"

def main():
    urdf_file = os.path.join(pydrake.getDrakePath(), PR2_PATH)

    # Load our model from URDF

    robot = RigidBodyTree()
    #print_tree_info(robot)

    # TODO: can define a collision clique for each body or something



    #add_model(robot, urdf_file, pose=Pose())
    add_model(robot, urdf_file, pose=Pose())
    print_tree_info(robot)

    """
    vis_helper = DrakeVisualizerHelper(robot)
    q0 = robot.getZeroConfiguration()
    vis_helper.draw(q0)
    #indices = []
    indices = get_body_indices(robot)
    result = robot.collisionDetect(get_kin_cache(robot, q0), indices, True)
    #print(result)
    #bodyA_idx, bodyB_idx = result
    phi, bodyA_idx, bodyB_idx = result
    print(len(bodyA_idx), len(bodyB_idx))
    print(len(indices))

    colliding = {(i, j) for d, i, j in zip(phi, bodyA_idx, bodyB_idx) if d < 1e-3}
    print(len(colliding))
    for i, j in colliding:
        body1 = body_from_index(robot, i)
        #print(body1.get_group_to_collision_ids_map()) # {u'non_gripper': [43871920]}
        #print(body1.get_collision_element_ids()) # [32922240]
        body2 = body_from_index(robot, j)
        print(body1.get_name(), body2.get_name(), 
            body1.adjacentTo(body2), body1.CanCollideWith(body2))
    return
    """

    table_file = os.path.join(pydrake.getDrakePath(), 
      "examples/kuka_iiwa_arm/models/table/",
      "extra_heavy_duty_table_surface_only_collision.sdf")
    #large_table_file = os.path.join(pydrake.getDrakePath(), 
    #  "examples/kuka_iiwa_arm/dev/box_rotation/models/",
    #  "large_extra_heavy_duty_table_surface_only_collision.sdf")
    box_file = os.path.join(pydrake.getDrakePath(), 
      "examples/kuka_iiwa_arm/models/objects/", 
      "block_for_pick_and_place_mid_size.urdf")

    print(table_file)

    add_model(robot, table_file, name='table1', pose=Pose(Point(2, 0, 0)), fixed=False)
    add_model(robot, table_file, name='table2', pose=Pose(Point(-2, 0, 0)), fixed=True)
    #add_model(robot, box_file, name='box1', pose=Pose(), fixed=False)
    #add_model(robot, box_file, name='box2', pose=Pose(), fixed=False)

    # https://github.com/kth-ros-pkg/pr2_ft_moveit_config
    # https://github.com/kth-ros-pkg/pr2_ft_moveit_config/blob/hydro/config/pr2.srdf


    #AddFlatTerrainToWorld(robot)
    AddFlatTerrainToWorld(robot, box_size=10, box_depth=.1) # Adds visual & collision
    print_tree_info(robot)

    vis_helper = DrakeVisualizerHelper(robot)
    #q = robot.getRandomConfiguration()
    q = robot.getZeroConfiguration()
    while True:
      vis_helper.draw(q)
      colliding = robot.collisionDetect(get_kin_cache(robot, q), get_body_indices(robot), True)
      #if not colliding:
      #    break
      if colliding:
          break
      print("Colliding: ", colliding)
      #raw_input("Continue?")
      q = robot.getRandomConfiguration()
      #print(q)
    
    vis_helper.draw(q)
    print("Conf", q)


    print([str(robot.get_position_name(i)) for i in xrange(robot.number_of_positions())])

    for i in xrange(robot.number_of_positions()):
        print(i, robot.get_position_name(i), robot.joint_limit_min[i], robot.joint_limit_max[i])


if __name__ == '__main__':
    main()
