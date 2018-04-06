import time

import numpy as np

from pr2_self_collision import PR2_COLLISION_PAIRS, INITIAL_COLLISION_PAIRS
from utils import get_drake_file, PR2_URDF, TABLE_SDF, BLOCK_URDF, get_enabled_collision_filter, \
    get_disabled_collision_filter, add_model, Pose, Point, get_num_models, get_model_name, get_bodies, \
    get_position_name, get_min_position, get_max_position, DrakeVisualizerHelper, Conf, get_position_ids, PR2_GROUPS, \
    TOP_HOLDING_LEFT_ARM, rightarm_from_leftarm, REST_LEFT_ARM, set_max_positions, set_pose, sample_placement, \
    is_fixed_base, get_top_grasps, get_body_from_name, PR2_TOOL_FRAMES, get_world_pose, get_pose, gripper_from_object, \
    get_frame_from_name, inverse_kinematics, PR2_LIMITS, get_position_limits, plan_motion, are_colliding, \
    other_collision_filter, set_random_positions, POSE_POSITIONS

from pydrake.multibody.rigid_body_tree import RigidBodyTree, AddFlatTerrainToWorld

def main():
    #pr2_file = 'pr2_simplified.urdf'
    pr2_file = get_drake_file(PR2_URDF)
    table_file = get_drake_file(TABLE_SDF)
    block_file = get_drake_file(BLOCK_URDF)

    enabled_collision_filter = get_enabled_collision_filter(PR2_COLLISION_PAIRS)
    #disabled_collision_filter = get_disabled_collision_filter(load_disabled_collisions('pr2.srdf'))
    disabled_collision_filter = get_disabled_collision_filter(INITIAL_COLLISION_PAIRS)

    tree = RigidBodyTree()
    pr2 = add_model(tree, pr2_file, fixed_base=True)

    table1 = add_model(tree, table_file, pose=Pose(Point(2, 0, 0)), fixed_base=True)
    table2 = add_model(tree, table_file, pose=Pose(Point(-2, 0, 0)), fixed_base=True)
    block1 = add_model(tree, block_file, fixed_base=False)
    block2 = add_model(tree, block_file, fixed_base=False)
    AddFlatTerrainToWorld(tree, box_size=10, box_depth=.1) # Adds visual & collision

    print("Models:", get_num_models(tree))
    for model_id in range(get_num_models(tree)):
        print(model_id, get_model_name(tree, model_id))

    print("Bodies:", tree.get_num_bodies())
    print("Frames:", tree.get_num_frames())
    for body in get_bodies(tree):
        print(body.get_body_index(), body.get_name(), body.get_group_to_collision_ids_map().keys())

    print("Positions:", tree.get_num_positions())
    print("Velocities:", tree.get_num_velocities())
    for position_id in range(tree.get_num_positions()):
        print(position_id, get_position_name(tree, position_id),
            get_min_position(tree, position_id), get_max_position(tree, position_id))

    print("Actuators:", tree.get_num_actuators())

    vis_helper = DrakeVisualizerHelper(tree)

    q = Conf(tree)
    q[get_position_ids(tree, PR2_GROUPS['base'], pr2)] = [0, 0, 0]
    q[get_position_ids(tree, PR2_GROUPS['torso'], pr2)] = [0.2]
    #q[get_position_ids(tree, PR2_GROUPS['left_arm'], pr2)] = REST_LEFT_ARM
    q[get_position_ids(tree, PR2_GROUPS['left_arm'], pr2)] = TOP_HOLDING_LEFT_ARM

    q[get_position_ids(tree, PR2_GROUPS['right_arm'], pr2)] = rightarm_from_leftarm(REST_LEFT_ARM)
    set_max_positions(tree, q, get_position_ids(tree, PR2_GROUPS['left_gripper'], pr2))
    set_max_positions(tree, q, get_position_ids(tree, PR2_GROUPS['right_gripper'], pr2))

    set_pose(tree, q, block1, sample_placement(tree, block1, table1))
    set_pose(tree, q, block2, sample_placement(tree, block2, table2))
    #set_random_positions(tree, q, get_position_ids(tree, POSE_POSITIONS, block1))
    #set_random_positions(tree, q, get_position_ids(tree, POSE_POSITIONS, block2))
    print(table1, is_fixed_base(tree, table1))
    print(block1, is_fixed_base(tree, block1))


    create_inverse_reachability(tree, pr2, block1, table1, 'left', 'top', num_samples=500)
    return

    grasps = get_top_grasps(tree, block1)
    #grasps = get_side_grasps(tree, block1)

    #gripper_id = get_frame_from_name(tree, PR2_TOOL_FRAMES['left_gripper'], pr2).get_frame_index()
    gripper_id = get_body_from_name(tree, PR2_TOOL_FRAMES['left_gripper'], pr2).get_body_index()
    gripper_pose = get_world_pose(tree, tree.doKinematics(q), gripper_id)

    #for grasp in grasps:
    #    set_pose(tree, q, block1, object_from_gripper(gripper_pose, grasp))
    #    vis_helper.draw(q)
    #    raw_input('Continue?')
    #return

    grasp = grasps[0]
    block1_pose = get_pose(tree, q, block1)
    print(grasp)
    print(block1_pose)
    target_gripper_pose = gripper_from_object(block1_pose, grasp)
    print(gripper_pose)

    frame_id = get_frame_from_name(tree, PR2_TOOL_FRAMES['left_gripper'], pr2).get_frame_index()
    start_time = time.time()
    #position_ids = get_position_ids(tree, PR2_GROUPS['base'], pr2)
    #position_ids = get_position_ids(tree, PR2_GROUPS['left_arm'], pr2)
    position_ids = get_position_ids(tree, PR2_GROUPS['base'] + PR2_GROUPS['left_arm'], pr2)
    #position_ids = None
    solution = inverse_kinematics(tree, frame_id, target_gripper_pose, position_ids=position_ids, q_seed=q)
    print(time.time()-start_time)
    if solution is None:
        return
    vis_helper.draw(solution)
    return

    position_limits = [PR2_LIMITS.get(get_position_name(tree, position_id),
                                      get_position_limits(tree, position_id))
                       for position_id in range(tree.get_num_positions())]


    print(position_limits)

    position_ids = get_position_ids(tree, PR2_GROUPS['base'], pr2)
    goal_values = [4, 0, 3*np.pi/2]
    #goal_values = [1, 0, 3*np.pi/2]
    start_time = time.time()
    path = plan_motion(tree, q, position_ids, goal_values, position_limits=position_limits,
                       collision_filter=disabled_collision_filter, model_ids=[pr2, table1])
    print(path)
    print(time.time()-start_time)
    if path is not None:
        for values in path:
            q[position_ids] = values
            vis_helper.draw(q)
            raw_input('Continue?')

    # ComputeMaximumDepthCollisionPoints
    # Use ComputeMaximumDepthCollisionPoints instead


    # body.ComputeWorldFixedPose()
    # CreateKinematicsCacheFromBodiesVector

    # TODO: problem that doesn't take into account frame

    #print('Colliding:', sorted(map(lambda (i, j): (str(get_body(tree, i).get_name()), str(get_body(tree, j).get_name())),
    #    colliding_bodies(tree, tree.doKinematics(Conf(tree)), model_ids=[pr2]))))

    #aabb = get_model_visual_aabb(tree, kin_cache, table1)
    #print(aabb)
    #print(get_aabb_min(aabb))

    while True:
        vis_helper.draw(q)
        #print('Positions:', q)
        print(
            #are_colliding(tree, tree.doKinematics(q), collision_filter=all_collision_filter),
            are_colliding(tree, tree.doKinematics(q), collision_filter=disabled_collision_filter),
            are_colliding(tree, tree.doKinematics(q), collision_filter=enabled_collision_filter),
            are_colliding(tree, tree.doKinematics(q), collision_filter=other_collision_filter))

        raw_input('Continue?')
        print(set_random_positions(tree, q, get_position_ids(tree, PR2_GROUPS['base'], pr2)))
        q[get_position_ids(tree, POSE_POSITIONS, block1)] = sample_placement(tree, block1, table1)
        q[get_position_ids(tree, POSE_POSITIONS, block2)] = sample_placement(tree, block2, table2)

if __name__ == '__main__':
    main()
