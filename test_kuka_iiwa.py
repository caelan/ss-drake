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

from test_pick import step_path, execute_path, convert_path
from pydrake.multibody.rigid_body_tree import RigidBodyTree, AddFlatTerrainToWorld

IIWA_URDF = "manipulation/models/iiwa_description/urdf/iiwa14_polytope_collision.urdf"
#IIWA_URDF = "manipulation/models/iiwa_description/urdf/iiwa14_spheres_collision.urdf"
#IIWA_URDF = "manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"

DUAL_IIWA_URDF = "manipulation/models/iiwa_description/urdf/dual_iiwa14_polytope_collision.urdf"
#DUAL_IIWA_URDF = "examples/kuka_iiwa_arm/dev/box_rotation/models/dual_iiwa14_primitive_sphere_visual_collision.urdf"
# dual_iiwa14_primitive_cylinder_collision_only.urdf
# dual_iiwa14_primitive_cylinder_visual_collision.urdf
# dual_iiwa14_primitive_sphere_collision_only.urdf
# dual_iiwa14_visual_only.urdf

VALKYRIE_URDF = "examples/valkyrie/urdf/urdf/valkyrie_A_sim_drake_one_neck_dof_wide_ankle_rom.urdf"

ATLAS_URDF = "examples/atlas/urdf/atlas_minimal_contact.urdf"
ATLAS_URDF = "examples/atlas/urdf/atlas_convex_hull.urdf"

JACO6_URDF = "manipulation/models/jaco_description/urdf/j2n6s300.urdf"
#JACO6_URDF = "manipulation/models/jaco_description/urdf/j2n6s300_col.urdf"
JACO7_URDF = "manipulation/models/jaco_description/urdf/j2s7s300.urdf"

WSG50_URDF = "manipulation/models/wsg_50_description/urdf/wsg_50_mesh_collision.urdf" # Gripper

#IRB140 = "examples/irb140/urdf/irb_140_robotiq_ati.urdf"
#IRB140 = "examples/irb140/urdf/irb_140_convhull.urdf"
#IRB140 = "examples/irb140/urdf/irb_140_robotiq_simple_ati.urdf"
#IRB140 = "examples/irb140/urdf/irb_140_robotiq.urdf"
IRB140 = "examples/irb140/urdf/irb_140.urdf"
# https://github.com/RobotLocomotion/drake/tree/c84bceb37a9fa9b01f23413733446495bf843725/manipulation/models
# https://github.com/caelan/drake/tree/master/examples/kuka_iiwa_arm/models/objects

FLAT_TERRAIN_URDF = "models/flat_terrain.urdf"
SHORT_FLOOR_URDF = "models/short_floor.urdf"

KUKA_TOOL_FRAME = 'iiwa_link_ee_kuka' # iiwa_link_ee_kuka | iiwa_link_ee

GRASP_NAMES = {
    'top': GraspInfo(lambda *args: get_top_grasps(*args, max_width=np.inf, tool_tform=np.eye(4), grasp_length=0), 
        None, 0.1*Pose(Point(z=1))),
}

KUKA_GROUPS = {
    'arm': [],
}

JACO6_GROUPS = {
    'arm': ['j2n6s300_joint_{}'.format(j) for j in range(1, 7)],
    'gripper': ['j2n6s300_joint_finger_{}'.format(j) for j in range(1, 4)],
}

JACO7_GROUPS = {
    'arm': ['j2s7s300_joint_{}'.format(j) for j in range(1, 8)],
    'gripper': ['j2s7s300_joint_finger_{}'.format(j) for j in range(1, 4)],
}

def sample_placements(tree, object_ids, surface_id, q_default=None):
    if q_default is None:
        q_default = Conf(tree)
    while True:
        q = q_default.copy()
        for object_id in object_ids:
            set_pose(tree, q, object_id, sample_placement(tree, object_id, surface_id))
        if not are_colliding(tree, tree.doKinematics(q)):
            return q

def sample_pick_path(tree, q_initial, robot_id, object_id, grasp_info):
    # TODO: method that produces these indepedent of the robot
    grasps = grasp_info.get_grasps(tree, object_id)
    gripper_id = get_body_from_name(tree, KUKA_TOOL_FRAME, robot_id).get_body_index()
    arm_ids = get_model_position_ids(tree, robot_id)

    object_pose = get_pose(tree, q_initial, object_id)
    grasp_pose = random.choice(grasps)
    gripper_pose = gripper_from_object(object_pose, grasp_pose)

    approach_pose = multiply_poses(grasp_info.approach_pose, gripper_pose)
    q_approach = inverse_kinematics(tree, gripper_id, approach_pose,
                                 position_ids=arm_ids, q_seed=q_initial)
    if (q_approach is None) or are_colliding(tree, tree.doKinematics(q_approach)):
        print('Failed approach inverse kinematics')
        return None
    #return [q_initial, q_approach]
    
    q_grasp = inverse_kinematics(tree, gripper_id, gripper_pose,
                                 position_ids=arm_ids, q_seed=q_approach)
    if (q_grasp is None) or are_colliding(tree, tree.doKinematics(q_grasp)):
        print('Failed grasp inverse kinematics')
        return None
    #return [q_initial, q_approach, q_grasp]

    approach_path = convert_path(q_initial, arm_ids, plan_motion(tree, q_initial, arm_ids, q_approach[arm_ids]))
    if approach_path is None:
        return None
    grasp_path = convert_path(approach_path[-1], arm_ids, plan_motion(tree, q_approach, arm_ids, q_grasp[arm_ids], linear_only=True))
    if grasp_path is None:
        return None
    print(len(approach_path), len(grasp_path))
    return (approach_path + grasp_path) # TODO: set the pose based on holding

def test_grasps(tree, vis_helper, robot, block, grasp_info):
    q0 = Conf(tree)
    end_effector = get_body_from_name(tree, KUKA_TOOL_FRAME, robot)
    end_effector_pose = get_world_pose(tree, tree.doKinematics(q0), end_effector.get_body_index())
    for grasp in grasp_info.get_grasps(tree, block):
        set_pose(tree, q0, block, object_from_gripper(end_effector_pose, grasp))
        vis_helper.draw(q0)
        raw_input('Continue?')

def main(num_blocks=1):
    tree = RigidBodyTree()
    robot = add_model(tree, get_drake_file(IIWA_URDF), fixed_base=True)
    ground = add_model(tree, SHORT_FLOOR_URDF, fixed_base=True)
    blocks = [add_model(tree, get_drake_file(BLOCK_URDF), fixed_base=False) for _ in range(num_blocks)]
    dump_tree(tree)
    vis_helper = DrakeVisualizerHelper(tree)

    #while True:
    #    q = tree.getRandomConfiguration()
    #    #q = Conf(tree)
    #    #set_min_positions(tree, q, get_position_ids(tree, JACO6_GROUPS['gripper'], robot)) # Open
    #    #set_max_positions(tree, q, get_position_ids(tree, JACO6_GROUPS['gripper'], robot)) # Closed
    #    #set_random_positions(tree, q, get_position_ids(tree, JACO6_GROUPS['gripper'], robot))
    #    vis_helper.draw(q)
    #    raw_input('Next?')
    #return

    block = blocks[0]
    grasp_info = GRASP_NAMES['top']
    #test_grasps(tree, vis_helper, robot, block, grasp_info)
    #return

    while True:
        q0 = sample_placements(tree, blocks, ground)
        vis_helper.draw(q0)
        full_path = sample_pick_path(tree, q0, robot, block, grasp_info)
        if full_path is None:
            continue
        print(len(full_path))
        raw_input('Execute?')
        execute_path(vis_helper, full_path)
        #step_path(vis_helper, full_path)


if __name__ == '__main__':
    main()
