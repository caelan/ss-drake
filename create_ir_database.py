from utils import *
import pickle

DATABASES_DIR = 'databases'
IR_FILENAME = '{}_{}_ir.pickle'

def write_pickle(filename, data):  # NOTE - cannot pickle lambda or nested functions
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

##################################################

def create_inverse_reachability(tree, robot_id, object_id, table_id, arm_name, grasp_name, default_q=None, num_samples=500):
    # TODO: maybe hash all the first arguments
    assert(arm_name in ('left', 'right'))
    if default_q is None:
        default_q = Conf(tree)
    vis_helper = DrakeVisualizerHelper(tree)

    #gripper_id = get_frame_from_name(tree, PR2_TOOL_FRAMES['left_gripper'], robot_id).get_frame_index()
    gripper_id = get_body_from_name(tree, PR2_TOOL_FRAMES['{}_gripper'.format(arm_name)], robot_id).get_body_index()

    position_names = PR2_GROUPS['{}_arm'.format(arm_name)] #+ PR2_GROUPS['torso'] + PR2_GROUPS['base']
    position_ids = get_position_ids(tree, position_names, robot_id)

    grasp_info = GRASP_NAMES[grasp_name] 
    grasps = grasp_info.get_grasps(tree, object_id)
    gripper_from_base_list = []
    while len(gripper_from_base_list) < num_samples:
        q_approach = default_q.copy()
        object_pose = sample_placement(tree, object_id, table_id)
        if object_pose is None:
            continue
        set_pose(tree, q_approach, object_id, object_pose)
        grasp_pose = random.choice(grasps)
        target_gripper_pose = gripper_from_object(object_pose, grasp_pose)
        q_approach[position_ids] = grasp_info.carry_values
        base_pose2d = sample_nearby_pose2d(point_from_pose(target_gripper_pose))
        if base_pose2d is None:
            continue
        q_approach[get_position_ids(tree, PR2_GROUPS['base'], robot_id)] = base_pose2d
        #if are_colliding(tree, tree.doKinematics(q_approach), collision_filter=other_collision_filter):
        #    continue
        vis_helper.draw(q_approach)
        q_grasp = inverse_kinematics(tree, gripper_id, target_gripper_pose, position_ids=position_ids, q_seed=q_approach)
        # TODO: confirm that is a solution here
        if (q_grasp is None): # or pairwise_collision(robot, table):
            continue
        kin_cache = tree.doKinematics(q_grasp)
        gripper_pose = get_world_pose(tree, kin_cache, gripper_id)
        if not np.allclose(target_gripper_pose, gripper_pose, atol=1e-4):
            continue
        #if are_colliding(tree, kin_cache, collision_filter=other_collision_filter):
        #    continue
        vis_helper.draw(q_grasp)
        print(target_gripper_pose, gripper_pose)
        gripper_from_base = multiply_poses(invert_pose(gripper_pose), 
            pose_from_pose2d(base_pose2d))
        gripper_from_base_list.append(gripper_from_base)
        print('{} / {}'.format(len(gripper_from_base_list), num_samples))
        raw_input('Continue?')

    filename = IR_FILENAME.format(grasp_name, arm_name)
    path = os.path.join(DATABASES_DIR, filename)
    data = {
        'filename': filename,
        'robot': get_model_name(tree, robot_id),
        'object': get_model_name(tree, object_id),
        'table': get_model_name(tree, table_id),
        'grasp_name': grasp_name,
        'arm_name': arm_name,
        #'gripper_link': link,
        'gripper_from_base': gripper_from_base_list,
    }
    write_pickle(path, data)
    return path

##################################################

def main():
    pr2_file = get_drake_file(PR2_URDF)
    table_file = get_drake_file(TABLE_SDF)
    block_file = get_drake_file(BLOCK_URDF)

    tree = RigidBodyTree()
    pr2 = add_model(tree, pr2_file, fixed_base=True)
    table1 = add_model(tree, table_file, pose=Pose(Point(0, 0, 0)), fixed_base=True)
    block1 = add_model(tree, block_file, fixed_base=False)
    AddFlatTerrainToWorld(tree, box_size=10, box_depth=.1) # Adds visual & collision

    q = Conf(tree)
    q[get_position_ids(tree, PR2_GROUPS['base'], pr2)] = [0, 0, 0]
    q[get_position_ids(tree, PR2_GROUPS['torso'], pr2)] = [0.2]
    q[get_position_ids(tree, PR2_GROUPS['left_arm'], pr2)] = REST_LEFT_ARM
    q[get_position_ids(tree, PR2_GROUPS['right_arm'], pr2)] = rightarm_from_leftarm(REST_LEFT_ARM)
    set_max_positions(tree, q, get_position_ids(tree, PR2_GROUPS['left_gripper'], pr2))
    set_max_positions(tree, q, get_position_ids(tree, PR2_GROUPS['right_gripper'], pr2))
    set_pose(tree, q, block1, sample_placement(tree, block1, table1))

    create_inverse_reachability(tree, pr2, block1, table1, 'left', 'top', num_samples=5, default_q=q)
    
if __name__ == '__main__':
    main()