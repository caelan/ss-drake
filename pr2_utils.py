from collections import namedtuple

import numpy as np

from utils import REVOLUTE_LIMITS, pose_from_tform, get_model_visual_aabb, Conf, get_aabb_center, get_aabb_extent, Pose, Euler, Point, \
    multiply_poses, invert_pose, set_max_positions, get_position_ids, set_min_positions, get_position_name, get_position_limits

PR2_URDF = "examples/pr2/models/pr2_description/urdf/pr2_simplified.urdf"
TABLE_SDF = "examples/kuka_iiwa_arm/models/table/extra_heavy_duty_table_surface_only_collision.sdf"
BLOCK_URDF = "examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place_mid_size.urdf"

ROBOT_TOY_URDER = "examples/kuka_iiwa_arm/models/objects/big_robot_toy.urdf"
FOLDING_TABLE_URDF = "examples/kuka_iiwa_arm/models/objects/folding_table.urdf"
ROUND_TABLE_URDF = "examples/kuka_iiwa_arm/models/objects/round_table.urdf"
SIMPLE_CYLINDER_URDF = "examples/kuka_iiwa_arm/models/objects/simple_cylinder.urdf"
SIMPLE_CUBOID_URDF = "examples/kuka_iiwa_arm/models/objects/simple_cuboid.urdf"
YELLOW_POST_URDF = "examples/kuka_iiwa_arm/models/objects/yellow_post.urdf"

BLACK_BOX_URDF = "examples/kuka_iiwa_arm/models/objects/black_box.urdf"
SMALL_BLOCK_URDF = "examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place.urdf"
LARGE_BLOCK_URDF = "examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place_large_size.urdf"

PR2_GROUPS = {
    'base': ['x', 'y', 'theta'],
    'torso': ['torso_lift_joint'],
    'head': ['head_pan_joint', 'head_tilt_joint'],
    'left_arm': ['l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_upper_arm_roll_joint',
                    'l_elbow_flex_joint', 'l_forearm_roll_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint'],
    'right_arm': ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint',
                     'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint'],
    'left_gripper': ['l_gripper_l_finger_joint', 'l_gripper_r_finger_joint',
        'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_tip_joint'],
    'right_gripper': ['r_gripper_l_finger_joint', 'r_gripper_r_finger_joint',
        'r_gripper_l_finger_tip_joint', 'r_gripper_r_finger_tip_joint'],
}
PR2_TOOL_FRAMES = {
    'left_gripper': 'l_gripper_palm_link', # l_gripper_palm_link | l_gripper_tool_frame
    'right_gripper': 'r_gripper_palm_link', # r_gripper_palm_link | r_gripper_tool_frame
}

TRANSLATION_LIMITS = (-10, 10)

PR2_REVOLUTE = ['theta',
                'r_forearm_roll_joint', 'r_wrist_roll_joint',
                'l_forearm_roll_joint', 'l_wrist_roll_joint'] # TODO: obtain from joint info
PR2_LIMITS = {
    'x': TRANSLATION_LIMITS,
    'y': TRANSLATION_LIMITS,
}
for joint_name in PR2_REVOLUTE:
    PR2_LIMITS[joint_name] = REVOLUTE_LIMITS

PR2_TOOL_TFORM = np.array([[0., 0., 1., 0.18],
                           [0., 1., 0., 0.],
                           [-1., 0., 0., 0.],
                           [0., 0., 0., 1.]])
#PR2_TOOL_TFORM = np.eye(4)

PR2_TOOL_DIRECTION = np.array([0., 0., 1.])

TOP_HOLDING_LEFT_ARM = [0.67717021, -0.34313199, 1.2, -1.46688405, 1.24223229, -1.95442826, 2.22254125]
SIDE_HOLDING_LEFT_ARM = [0.39277395, 0.33330058, 0., -1.52238431, 2.72170996, -1.21946936, -2.98914779]
REST_LEFT_ARM = [2.13539289, 1.29629967, 3.74999698, -0.15000005, 10000., -0.10000004, 10000.]
WIDE_LEFT_ARM = [1.5806603449288885, -0.14239066980481405, 1.4484623937179126, -1.4851759349218694, 1.3911839347271555, -1.6531320011389408, -2.978586584568441]
CENTER_LEFT_ARM = [-0.07133691252641006, -0.052973836083405494, 1.5741805775919033, -1.4481146328076862, 1.571782540186805, -1.4891468812835686, -9.413338322697955]


def rightarm_from_leftarm(config):
  right_from_left = np.array([-1, 1, -1, 1, -1, 1, -1])
  return config*right_from_left

def get_pr2_limits(tree):
    return [PR2_LIMITS.get(get_position_name(tree, position_id), get_position_limits(tree, position_id))
                       for position_id in range(tree.get_num_positions())]


GRASP_LENGTH = 0.04 # 0
#GRASP_LENGTH = 0.0
MAX_GRASP_WIDTH = 0.07

def get_top_grasps(tree, model_id, under=False, max_width=MAX_GRASP_WIDTH, 
        tool_tform=PR2_TOOL_TFORM, grasp_length=GRASP_LENGTH):
    tool_pose = pose_from_tform(tool_tform)
    #tool_pose = Pose()

    aabb = get_model_visual_aabb(tree, tree.doKinematics(Conf(tree)), model_id)
    w, l, h = 2*get_aabb_extent(aabb)
    #print(get_aabb_center(aabb)) # TODO: incorporate
    assert(np.allclose(get_aabb_center(aabb), np.zeros(3)))
    reflect_z = Pose(euler=Euler(pitch=np.pi))
    translate = Pose(point=Point(z=(h / 2 - grasp_length)))
    grasps = []
    if w <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=Euler(yaw=(np.pi / 2 + i * np.pi)))
            grasps += [multiply_poses(tool_pose, translate, rotate_z, reflect_z)]
    if l <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=Euler(yaw=(i*np.pi)))
            grasps += [multiply_poses(tool_pose, translate, rotate_z, reflect_z)]
    return grasps


def get_side_grasps(tree, model_id, under=False, limits=False, grasp_length=GRASP_LENGTH):
    tool_pose = pose_from_tform(PR2_TOOL_TFORM)
    aabb = get_model_visual_aabb(tree, tree.doKinematics(Conf(tree)), model_id)
    w, l, h = 2 * get_aabb_extent(aabb)
    grasps = []
    for j in range(1 + under):
        swap_xz = Pose(euler=Euler(pitch=(-np.pi / 2 + j * np.pi)))
        if not limits or (w <= MAX_GRASP_WIDTH):
            translate = Pose(point=Point(z=(l / 2 - grasp_length)))
            for i in range(2):
                rotate_z = Pose(euler=Euler(roll=(np.pi / 2 + i * np.pi)))
                grasps += [multiply_poses(tool_pose, translate, rotate_z, swap_xz)]
        if not limits or (l <= MAX_GRASP_WIDTH):
            translate = Pose(point=Point(z=(w / 2 - grasp_length)))
            for i in range(2):
                rotate_z = Pose(euler=Euler(roll=(i * np.pi)))
                grasps += [multiply_poses(tool_pose, translate, rotate_z, swap_xz)]
    return grasps


# def get_x_presses(body, max_orientations=1): # g_f_o
#   pose = get_pose(body)
#   set_pose(body, unit_pose())
#   center, (w, l, h) = get_center_extent(body)
#   press_poses = []
#   for j in xrange(max_orientations):
#       swap_xz = (unit_point(), quat_from_euler([0, -np.pi/2 + j*np.pi, 0]))
#       translate = ([0, 0, w / 2], unit_quat())
#       press_poses += [multiply(TOOL_POSE, translate, swap_xz)]
#   set_pose(body, pose)
#   return press_poses

APPROACH_DISTANCE = 0.1

GraspInfo = namedtuple('GraspInfo', ['get_grasps', 'carry_values', 'approach_pose'])
GRASP_NAMES = {
    'top': GraspInfo(get_top_grasps, TOP_HOLDING_LEFT_ARM,
                     APPROACH_DISTANCE*Pose(Point(z=1))),
    'side': GraspInfo(get_side_grasps, SIDE_HOLDING_LEFT_ARM,
                      APPROACH_DISTANCE * Pose(Point(z=1))),
}


def object_from_gripper(gripper_pose, grasp_pose):
    return multiply_poses(gripper_pose, grasp_pose)


def gripper_from_object(object_pose, grasp_pose):
    return multiply_poses(object_pose, invert_pose(grasp_pose))


def open_pr2_gripper(tree, q, model_id, gripper_name):
    set_max_positions(tree, q, get_position_ids(tree, PR2_GROUPS[gripper_name], model_id))


def close_pr2_gripper(tree, q, model_id, gripper_name):
    set_min_positions(tree, q, get_position_ids(tree, PR2_GROUPS[gripper_name], model_id))