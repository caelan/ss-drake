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
from pydrake.solvers import ik
from pydrake.systems.analysis import Simulator
from pydrake.multibody.rigid_body_plant import RigidBodyPlant, DrakeVisualizer, \
  CompliantMaterial, CompliantContactModelParameters
from pydrake.lcm import DrakeMockLcm, DrakeLcm, DrakeLcmInterface
from pydrake.multibody.shapes import Box, Cylinder, Mesh
from pydrake.systems.framework import BasicVector, DiagramBuilder
# TODO(eric.cousineau): Use `unittest` (after moving `ik` into `multibody`),
# declaring this as a drake_py_unittest in the BUILD.bazel file.

def pose_from_point_euler(point, euler):
    return np.concatenate([point, euler])

def euler_from_theta(theta):
    return np.array([0, 0, theta])

def Point():
    return np.zeros(3)

def Euler():
    return np.zeros(3)

def Pose():
    return np.concatenate([unit_point(), unit_euler()])


def add_model(tree, model_file, fixed_base=True):
    model_string = open(model_file).read()
    base_dir = os.path.dirname(model_file)
    package_map = PackageMap()
    weld_frame = None
    base_type = FloatingBaseType.kFixed if fixed_base else FloatingBaseType.kRollPitchYaw
    #base_type = FloatingBaseType.kQuaternion

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
    return tree


def load_robot_from_urdf(urdf_file, fixed_base=True):
    """
    This function demonstrates how to pass a complete
    set of arguments to Drake's URDF parser.  It is also
    possible to load a robot with a much simpler syntax
    that uses default values, such as:

      robot = RigidBodyTree(urdf_file)

    """
    # PR2 has explicit base joints

    # Load our model from URDF
    robot = RigidBodyTree()
    add_model(robot, urdf_file, fixed_base=fixed_base)

    #robot = RigidBodyTree(urdf_file, '0' if fixed_base else '1')
    #robot = RigidBodyTree(urdf_file)

    return robot

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

BASE_POSITIONS = ['base_x', 'base_y', 'base_z']

POSITION_NAMES = ['base_x', 'base_y', 'base_z', 'base_roll', 'base_pitch', 'base_yaw', 
  'x', 'y', 'theta', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint', 
  'r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint', 'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint', 
  'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint', 'r_gripper_l_finger_tip_joint', 'r_gripper_r_finger_tip_joint', 
  'l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_upper_arm_roll_joint', 'l_elbow_flex_joint', 'l_forearm_roll_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint', 
  'l_gripper_l_finger_joint', 'l_gripper_r_finger_joint', 'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_tip_joint']

def get_positions(tree):
    return range(tree.number_of_positions())

def get_position_names(tree):
    return [name_from_position(tree, i) for i in get_positions(tree)]

def name_from_position(tree, position):
    return str(tree.get_position_name(position))

def position_from_name(tree, name):
    for i in xrange(tree.number_of_positions()):
        if name_from_position(tree, i) == name:
            return i
    raise ValueError(name)

def position_min_limit(tree, position):
    return tree.joint_limit_min[position]

def position_max_limit(tree, position):
    return tree.joint_limit_max[position]

def gripper_positions(tree, arm):
    prefix_from_arm = {
        'right': 'r_gripper_',
        'left': 'l_gripper_',
    }
    return tuple(i for i in xrange(tree.number_of_positions()) 
        if name_from_position(tree, i).startswith(prefix_from_arm[arm]))

def arm_positions(tree, arm):
    prefix_from_arm = {
        'right': 'r_',
        'left': 'l_',
        'base': 'base_',
        'head': 'head_',
    }
    return tuple(i for i in xrange(tree.number_of_positions()) 
        if name_from_position(tree, i).startswith(prefix_from_arm[arm]) and (i not in gripper_positions(tree, arm)))

def print_tree_info(robot):
    print("Positions:", robot.get_num_positions()) # 34 | number_of_positions
    print("Velocities:", robot.get_num_velocities()) # 34 | number_of_velocities
    print("Actuators:", robot.get_num_actuators()) # 28
    print("Frames:", robot.get_num_frames()) # 86
    print("Bodies:", robot.get_num_bodies()) # 86
    world = robot.world()
    print("Tree:", robot) # RigidBodyTree
    print("World:", world) # RigidBody
    print(world.get_name(), "elements:", world.get_visual_elements()) # List of visual elements

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

def main():
    urdf_file = os.path.join(pydrake.getDrakePath(),
        "examples/pr2/models/pr2_description/urdf/pr2_simplified.urdf")

    # Load our model from URDF
    robot = load_robot_from_urdf(urdf_file)
    print_tree_info(robot)

    table_file = os.path.join(pydrake.getDrakePath(), 
      "examples/kuka_iiwa_arm/models/table/",
      "extra_heavy_duty_table_surface_only_collision.sdf")
    large_table_file = os.path.join(pydrake.getDrakePath(), 
      "examples/kuka_iiwa_arm/dev/box_rotation/models/",
      "large_extra_heavy_duty_table_surface_only_collision.sdf")
    box_file = os.path.join(pydrake.getDrakePath(), 
      "examples/kuka_iiwa_arm/dev/box_rotation/models/", 
      "box.urdf")

    print(table_file)

    add_model(robot, table_file, fixed_base=True)
    #AddFlatTerrainToWorld(robot)
    print_tree_info(robot)


    vis_helper = DrakeVisualizerHelper(robot)
    q = robot.getRandomConfiguration()
    #q = robot.getZeroConfiguration()
    vis_helper.draw(q)
    print("Conf", q)

    print(robot.getTerrainContactPoints(robot.world()))

    print(robot.joint_limit_min.shape, robot.joint_limit_max.shape)
    print(robot.joint_limit_min, robot.joint_limit_max)

    print([str(robot.get_position_name(i)) for i in xrange(robot.number_of_positions())])

    for i in xrange(robot.number_of_positions()):
        print(i, robot.get_position_name(i), robot.joint_limit_min[i], robot.joint_limit_max[i])

    arms = ['right', 'left']
    for arm in arms:
        print(arm, arm_positions(robot, arm))
        print(arm, gripper_positions(robot, arm))

    # KinematicsCache
    # http://drake.mit.edu/doxygen_cxx/class_kinematics_cache.html

if __name__ == '__main__':
    main()
