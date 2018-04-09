import random
import time

from utils import get_drake_file, get_enabled_collision_filter, \
    get_disabled_collision_filter, add_model,get_position_name, DrakeVisualizerHelper, Conf, \
    get_position_ids, get_position_limits, plan_motion, sample_placement, set_pose, get_body_from_name, \
    are_colliding, other_collision_filter, inverse_kinematics, get_world_pose, get_pose, plan_motion, \
    Pose, Point, multiply_poses
from pr2_utils import PR2_URDF, TABLE_SDF, BLOCK_URDF, PR2_GROUPS, PR2_LIMITS, REST_LEFT_ARM, \
    rightarm_from_leftarm, open_pr2_gripper, GRASP_NAMES, gripper_from_object, PR2_TOOL_FRAMES, get_pr2_limits
from create_ir_database import uniform_pose_generator, learned_pose_generator

from pydrake.multibody.rigid_body_tree import RigidBodyTree, AddFlatTerrainToWorld

def step_path(vis_helper, path):
    for i, q in enumerate(path):
        vis_helper.draw(q)
        raw_input('{}) step?'.format(i))

def execute_path(vis_helper, path):
    for i, q in enumerate(path):
        vis_helper.draw(q)
        time.sleep(0.1)

def convert_path(q0, position_ids, path):
    for values in path:
        q = q0.copy()
        q[position_ids] = values
        yield q

def sample_base_pose(tree, q0, robot_id, object_id, arm_name, grasp_name):
    grasp_info = GRASP_NAMES[grasp_name]
    grasps = grasp_info.get_grasps(tree, object_id)
    gripper_id = get_body_from_name(tree, PR2_TOOL_FRAMES['{}_gripper'.format(arm_name)],
                                 robot_id).get_body_index()
    arm_ids = get_position_ids(tree, PR2_GROUPS['{}_arm'.format(arm_name)], robot_id)
    base_ids = get_position_ids(tree, PR2_GROUPS['base'], robot_id)

    object_pose = get_pose(tree, q0, object_id)
    collision_filter = other_collision_filter
    while True:
        grasp_pose = random.choice(grasps)
        gripper_pose = gripper_from_object(object_pose, grasp_pose)
        approach_pose = multiply_poses(grasp_info.approach_pose, gripper_pose)

        base_generator = learned_pose_generator(gripper_pose, arm_name, grasp_name)

        q_approach = q0.copy()
        q_approach[arm_ids] = grasp_info.carry_values
        base_pose2d = next(base_generator)
        if base_pose2d is None:
            continue
        q_approach[base_ids] = base_pose2d
        if are_colliding(tree, tree.doKinematics(q_approach), collision_filter=collision_filter):
            continue
        q_grasp = inverse_kinematics(tree, gripper_id, gripper_pose,
                                     position_ids=arm_ids, q_seed=q_approach)
        if q_grasp is None:
            continue
        if are_colliding(tree, tree.doKinematics(q_grasp), collision_filter=collision_filter):
            continue
        #base_path = []
        base_path = plan_motion(tree, q0, base_ids, base_pose2d,
                           position_limits=get_pr2_limits(tree), collision_filter=collision_filter)
        if base_path is None:
            continue
        arm_path = plan_motion(tree, q_approach, arm_ids, q_grasp[arm_ids],
                           position_limits=get_pr2_limits(tree), collision_filter=collision_filter)

        yield base_path

def main():
    pr2_file = get_drake_file(PR2_URDF)
    table_file = get_drake_file(TABLE_SDF)
    block_file = get_drake_file(BLOCK_URDF)

    tree = RigidBodyTree()
    pr2 = add_model(tree, pr2_file, fixed_base=True)
    table1 = add_model(tree, table_file, pose=Pose(Point(2, 0, 0)), fixed_base=True)
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

    iteration = 0
    while True:
        iteration += 1
        print('Iteration: {}'.format(iteration))
        q = q0.copy()
        set_pose(tree, q, block1, sample_placement(tree, block1, table1))
        path = next(sample_base_pose(tree, q, pr2, block1, 'left', 'top'))
        print(len(path))

if __name__ == '__main__':
    main()
