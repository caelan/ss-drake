from __future__ import absolute_import, division, print_function

import os
import numpy as np
import pydrake
from pydrake.multibody.parsers import PackageMap
from pydrake.multibody.rigid_body_tree import (
    AddModelInstanceFromUrdfStringSearchingInRosPackages,
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


def load_robot_from_urdf(urdf_file):
    """
    This function demonstrates how to pass a complete
    set of arguments to Drake's URDF parser.  It is also
    possible to load a robot with a much simpler syntax
    that uses default values, such as:

      robot = RigidBodyTree(urdf_file)

    """
    urdf_string = open(urdf_file).read()
    base_dir = os.path.dirname(urdf_file)
    package_map = PackageMap()
    weld_frame = None
    floating_base_type = FloatingBaseType.kRollPitchYaw

    # Load our model from URDF
    robot = RigidBodyTree()

    AddModelInstanceFromUrdfStringSearchingInRosPackages(
        urdf_string,
        package_map,
        base_dir,
        floating_base_type,
        weld_frame,
        robot)

    return robot

class DrakeVisualizerHelper:
    def __init__(self, tree):
        lcm = DrakeLcm()
        self.tree = tree
        self.visualizer = DrakeVisualizer(tree=self.tree, lcm=lcm, enable_playback=True)
        self.x = np.concatenate([self.tree.getZeroConfiguration(), 
                                 np.zeros(tree.get_num_velocities())])
        self.visualizer.PublishLoadRobot()
        self.draw(self.tree.getZeroConfiguration())

    def draw(self, q = None):
        if q is not None:
            self.x[:self.tree.get_num_positions()] = q
        context = self.visualizer.CreateDefaultContext()
        context.FixInputPort(0, BasicVector(self.x))
        self.visualizer.Publish(context)
        
    def inspect(self, slider_scaling = 1):
        # Setup widgets
        for i in range(self.tree.number_of_positions()):
            widgets.interact(
                self.__slider_callback,
                slider_value = widgets.FloatSlider(
                    value=slider_scaling * self.x[i],
                    min=slider_scaling * self.tree.joint_limit_min[i],
                    max=slider_scaling * self.tree.joint_limit_max[i],
                    description=self.tree.get_position_name(i)
                ),
                index=widgets.fixed(i),
                slider_scaling=widgets.fixed(slider_scaling)
            )

    def __slider_callback(self, slider_value, index, slider_scaling):
        self.x[index] = slider_value / slider_scaling
        self.draw()

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

def main():
    urdf_file = os.path.join(
        pydrake.getDrakePath(),
        "examples/pr2/models/pr2_description/urdf/pr2_simplified.urdf")

    # Load our model from URDF
    robot = load_robot_from_urdf(urdf_file)

    AddFlatTerrainToWorld(robot)
    print("Positions:", robot.get_num_positions()) # 34 | number_of_positions
    print("Velocities:", robot.get_num_velocities()) # 34 | number_of_velocities
    print("Actuators:", robot.get_num_actuators()) # 28
    print("Frames:", robot.get_num_frames()) # 86
    print("Bodies:", robot.get_num_bodies()) # 86
    world = robot.world()
    print("Robot:", robot) # RigidBodyTree
    print("World:", world) # RigidBody
    print(world.get_name(), world.get_visual_elements()) # List of visual elements

    vis_helper = DrakeVisualizerHelper(robot)
    #q = robot.getRandomConfiguration()
    q = robot.getZeroConfiguration()
    vis_helper.draw(q)
    print("Conf", q)

    print(robot.getTerrainContactPoints(world))

    print(robot.joint_limit_min.shape, robot.joint_limit_max.shape)
    print(robot.joint_limit_min, robot.joint_limit_max)

    print([str(robot.get_position_name(i)) for i in xrange(robot.number_of_positions())])

    for i in xrange(robot.number_of_positions()):
        print(i, robot.get_position_name(i), robot.joint_limit_min[i], robot.joint_limit_max[i])

    arms = ['right', 'left']
    for arm in arms:
        print(arm, arm_positions(robot, arm))
        print(arm, gripper_positions(robot, arm))

    vis_helper.inspect(np.rad2deg(1))



if __name__ == '__main__':
    main()
