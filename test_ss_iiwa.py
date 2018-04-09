#!/usr/bin/env python2.7

import time
import random
import numpy as np

from pr2_self_collision import PR2_COLLISION_PAIRS, INITIAL_COLLISION_PAIRS
from utils import get_drake_file, get_enabled_collision_filter, \
    get_disabled_collision_filter, add_model,get_position_name, DrakeVisualizerHelper, Conf, \
    get_position_ids, get_position_limits, plan_motion, dump_tree, set_pose, sample_placement, are_colliding, get_body_from_name, \
    get_model_position_ids, get_pose, multiply_poses, inverse_kinematics, get_world_pose, Pose, Point, set_min_positions, set_random_positions, \
    set_max_positions, set_center_positions 
from pr2_utils import PR2_URDF, TABLE_SDF, PR2_GROUPS, PR2_LIMITS, REST_LEFT_ARM, \
    rightarm_from_leftarm, open_pr2_gripper, get_pr2_limits, BLOCK_URDF, gripper_from_object, object_from_gripper, \
    GraspInfo, get_top_grasps

from test_kuka_iiwa import IIWA_URDF, SHORT_FLOOR_URDF, GRASP_NAMES
from test_pick import step_path, execute_path, convert_path
from pydrake.multibody.rigid_body_tree import RigidBodyTree, AddFlatTerrainToWorld

SINK_URDF = 'models/sink.urdf'
STOVE_URDF = 'models/sink.urdf'

# TODO: stacking

def main():
    tree = RigidBodyTree()
    robot = add_model(tree, get_drake_file(IIWA_URDF), fixed_base=True)
    sink = add_model(tree, SINK_URDF, pose=Pose(Point(x=-0.5)), fixed_base=True)
    stove = add_model(tree, STOVE_URDF, pose=Pose(Point(x=+0.5)), fixed_base=True)
    ground = add_model(tree, SHORT_FLOOR_URDF, fixed_base=True)
    block = add_model(tree, get_drake_file(BLOCK_URDF), fixed_base=False)
    dump_tree(tree)
    vis_helper = DrakeVisualizerHelper(tree)

    q0 = Conf(tree)
    set_pose(tree, q0, block, Pose(Point(y=0.5, z=0.09)))
    vis_helper.draw(q0)

    grasp_info = GRASP_NAMES['top']


if __name__ == '__main__':
    main()
