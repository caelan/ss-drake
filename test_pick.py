import time

import numpy as np
import random

from pr2_self_collision import PR2_COLLISION_PAIRS, INITIAL_COLLISION_PAIRS
from utils import get_drake_file, get_enabled_collision_filter, \
    get_disabled_collision_filter, add_model,get_position_name, DrakeVisualizerHelper, Conf, \
    get_position_ids, get_position_limits, plan_motion, sample_placement, set_pose, get_body_from_name, \
    are_colliding, other_collision_filter, inverse_kinematics, get_world_pose
from pr2_utils import PR2_URDF, TABLE_SDF, BLOCK_URDF, PR2_GROUPS, PR2_LIMITS, REST_LEFT_ARM, \
    rightarm_from_leftarm, open_pr2_gripper, GRASP_NAMES, gripper_from_object, PR2_TOOL_FRAMES
from create_ir_database import uniform_pose_generator, learned_pose_generator

from pydrake.multibody.rigid_body_tree import RigidBodyTree, AddFlatTerrainToWorld

def sample_base_pose(tree, robot_id, q0, object_pose, grasp_pose, arm_name, grasp_name):
    gripper_id = get_body_from_name(tree, PR2_TOOL_FRAMES['{}_gripper'.format(arm_name)],
                                 robot_id).get_body_index()
    position_ids = get_position_ids(tree, PR2_GROUPS['{}_arm'.format(arm_name)], robot_id)

    gripper_pose = gripper_from_object(object_pose, grasp_pose)
    base_generator = learned_pose_generator(gripper_pose, arm_name, grasp_name)
    while True:
        q_approach = q0.copy()
        q_approach[position_ids] = GRASP_NAMES[grasp_name].carry_values
        base_pose2d = next(base_generator)
        if base_pose2d is None:
            continue
        q_approach[get_position_ids(tree, PR2_GROUPS['base'], robot_id)] = base_pose2d
        if are_colliding(tree, tree.doKinematics(q_approach), collision_filter=other_collision_filter):
            continue
        q_grasp = inverse_kinematics(tree, gripper_id, gripper_pose,
                                     position_ids=position_ids, q_seed=q_approach)
        if q_grasp is None:
            continue
        kin_cache = tree.doKinematics(q_grasp)
        gripper_pose = get_world_pose(tree, kin_cache, gripper_id)
        if not np.allclose(gripper_pose, gripper_pose, atol=1e-4):
            continue
        if are_colliding(tree, kin_cache, collision_filter=other_collision_filter):
            continue
        yield

def main():
    pr2_file = get_drake_file(PR2_URDF)
    table_file = get_drake_file(TABLE_SDF)
    block_file = get_drake_file(BLOCK_URDF)

    enabled_collision_filter = get_enabled_collision_filter(PR2_COLLISION_PAIRS)
    #disabled_collision_filter = get_disabled_collision_filter(load_disabled_collisions('pr2.srdf'))
    disabled_collision_filter = get_disabled_collision_filter(INITIAL_COLLISION_PAIRS)

    tree = RigidBodyTree()
    pr2 = add_model(tree, pr2_file, fixed_base=True)
    table1 = add_model(tree, table_file, fixed_base=True)
    block1 = add_model(tree, block_file, fixed_base=False)
    AddFlatTerrainToWorld(tree, box_size=10, box_depth=.1) # Adds visual & collision
    vis_helper = DrakeVisualizerHelper(tree)

    q0 = Conf(tree)
    q0[get_position_ids(tree, PR2_GROUPS['base'], pr2)] = [0, 0, 0]
    q0[get_position_ids(tree, PR2_GROUPS['torso'], pr2)] = [0.2]
    q0[get_position_ids(tree, PR2_GROUPS['left_arm'], pr2)] = REST_LEFT_ARM
    q0[get_position_ids(tree, PR2_GROUPS['right_arm'], pr2)] = rightarm_from_leftarm(REST_LEFT_ARM)
    open_pr2_gripper(tree, q0, pr2, 'left_gripper')
    open_pr2_gripper(tree, q0, pr2, 'right_gripper')

    grasp_name = 'top'
    grasp_info = GRASP_NAMES[grasp_name]
    grasps = grasp_info.get_grasps(tree, block1)
    while True:
        q = q0.copy()
        block_pose = sample_placement(tree, block1, table1)
        set_pose(tree, q, block1, block_pose)
        grasp_pose = random.choice(grasps)



if __name__ == '__main__':
    main()
