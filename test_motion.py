import time

import numpy as np

from pr2_self_collision import PR2_COLLISION_PAIRS, INITIAL_COLLISION_PAIRS
from utils import get_drake_file, get_enabled_collision_filter, \
    get_disabled_collision_filter, add_model,get_position_name, DrakeVisualizerHelper, Conf, \
    get_position_ids, get_position_limits, plan_motion, other_collision_filter
from pr2_utils import PR2_URDF, TABLE_SDF, PR2_GROUPS, PR2_LIMITS, REST_LEFT_ARM, \
    rightarm_from_leftarm, open_pr2_gripper, get_pr2_limits
from test_pick import step_path, execute_path, convert_path

from pydrake.multibody.rigid_body_tree import RigidBodyTree, AddFlatTerrainToWorld

def main():
    enabled_collision_filter = get_enabled_collision_filter(PR2_COLLISION_PAIRS)
    #disabled_collision_filter = get_disabled_collision_filter(load_disabled_collisions('pr2.srdf'))
    disabled_collision_filter = get_disabled_collision_filter(INITIAL_COLLISION_PAIRS)

    tree = RigidBodyTree()
    pr2 = add_model(tree, get_drake_file(PR2_URDF), fixed_base=True)
    table = add_model(tree, get_drake_file(TABLE_SDF), fixed_base=True) # TODO: table leg collisions
    AddFlatTerrainToWorld(tree, box_size=10, box_depth=.1) # Adds visual & collision
    vis_helper = DrakeVisualizerHelper(tree)

    start_pose2d = [-2, 0, 0]
    end_pose2d = [+2, 0, 3*np.pi/2]

    q = Conf(tree)
    q[get_position_ids(tree, PR2_GROUPS['torso'], pr2)] = [0.2]
    q[get_position_ids(tree, PR2_GROUPS['left_arm'], pr2)] = REST_LEFT_ARM
    q[get_position_ids(tree, PR2_GROUPS['right_arm'], pr2)] = rightarm_from_leftarm(REST_LEFT_ARM)
    open_pr2_gripper(tree, q, pr2, 'left_gripper')
    open_pr2_gripper(tree, q, pr2, 'right_gripper')

    base_ids = get_position_ids(tree, PR2_GROUPS['base'], pr2)
    q[base_ids] = start_pose2d
    start_time = time.time()
    base_path = plan_motion(tree, q, base_ids, end_pose2d,
                       position_limits=get_pr2_limits(tree),
                       collision_filter=other_collision_filter)

    print(time.time()-start_time)
    if base_path is None:
        print('Failed to find a path')
        return
    print('Found a path of length {}'.format(len(base_path)))
    full_path = convert_path(q, base_ids, base_path)
    execute_path(vis_helper, full_path, time_step=0.05)
    step_path(vis_helper, full_path)


if __name__ == '__main__':
    main()
