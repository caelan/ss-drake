from __future__ import absolute_import, division, print_function

import re
import os
import numpy as np
import pydrake
import time
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

from pydrake.multibody.rigid_body_plant import RigidBodyPlant, DrakeVisualizer, \
  CompliantMaterial, CompliantContactModelParameters
from pydrake.lcm import DrakeMockLcm, DrakeLcm, DrakeLcmInterface
from pydrake.systems.framework import BasicVector, DiagramBuilder
# TODO(eric.cousineau): Use `unittest` (after moving `ik` into `multibody`),
# declaring this as a drake_py_unittest in the BUILD.bazel file.
from motion_planners.rrt_connect import birrt, direct_path
from pydrake.solvers import ik

from pr2_self_collision import PR2_COLLISION_PAIRS, INITIAL_COLLISION_PAIRS

X_POSITION = 'base_x' # 'weld_x'
Y_POSITION = 'base_y'
Z_POSITION = 'base_z'
ROLL_POSITION = 'base_roll'
PITCH_POSITION = 'base_pitch'
YAW_POSITION = 'base_yaw'

POINT_POSITIONS = [X_POSITION, Y_POSITION, Z_POSITION]
EULER_POSITIONS = [ROLL_POSITION, PITCH_POSITION, YAW_POSITION]
POSE_POSITIONS = POINT_POSITIONS + EULER_POSITIONS
POSE2D_POSITIONS = [X_POSITION, Y_POSITION, YAW_POSITION]

def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])

def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])

def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return np.concatenate([point, euler])

def point_from_pose(pose):
    return pose[:3]

def euler_from_pose(pose):
    return pose[3:]

def point_euler_from_pose(pose):
    return point_from_pose(pose), euler_from_pose(pose)

from transformations import euler_matrix, euler_from_matrix

def rot_from_euler(euler):
    return euler_matrix(*euler, axes='rxyz')[:3,:3] # TODO: confirm this is what I want

def euler_from_rot(rot):
    return np.array(euler_from_matrix(rot, axes='rxyz'))

def tform_from_pose(pose):
    tform = np.eye(4)
    tform[:3, 3] = point_from_pose(pose)
    tform[:3, :3] = rot_from_euler(euler_from_pose(pose))
    return tform

def point_from_tform(tform):
    return tform[:3, 3]

def rot_from_tform(tform):
    return tform[:3, :3]

def pose_from_tform(tform):
    return Pose(point_from_tform(tform), euler_from_rot(rot_from_tform(tform)))

def multiply_poses(*poses):
    tform = np.eye(4)
    for pose in poses:
        tform = tform.dot(tform_from_pose(pose))
    return pose_from_tform(tform)

def invert_pose(pose):
    return pose_from_tform(np.linalg.inv(tform_from_pose(pose)))

##################################################

PR2_URDF = "examples/pr2/models/pr2_description/urdf/pr2_simplified.urdf"
TABLE_SDF = "examples/kuka_iiwa_arm/models/table/extra_heavy_duty_table_surface_only_collision.sdf"
BLOCK_URDF = "examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place_mid_size.urdf"

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
REVOLUTE_LIMITS = (-np.pi, np.pi)

PR2_LIMITS = {
    'x': TRANSLATION_LIMITS,
    'y': TRANSLATION_LIMITS,
    'theta': REVOLUTE_LIMITS,
    'r_forearm_roll_joint': REVOLUTE_LIMITS,
    'r_wrist_roll_joint': REVOLUTE_LIMITS,
    'l_forearm_roll_joint': REVOLUTE_LIMITS,
    'l_wrist_roll_joint': REVOLUTE_LIMITS,
}

PR2_TOOL_TFORM = np.array([[0., 0., 1., 0.18],
                           [0., 1., 0., 0.],
                           [-1., 0., 0., 0.],
                           [0., 0., 0., 1.]])
PR2_TOOL_DIRECTION = np.array([0., 0., 1.])

TOP_HOLDING_LEFT_ARM = [0.67717021, -0.34313199, 1.2, -1.46688405, 1.24223229, -1.95442826, 2.22254125]
SIDE_HOLDING_LEFT_ARM = [0.39277395, 0.33330058, 0., -1.52238431, 2.72170996, -1.21946936, -2.98914779]
REST_LEFT_ARM = [2.13539289, 1.29629967, 3.74999698, -0.15000005, 10000., -0.10000004, 10000.]
WIDE_LEFT_ARM = [1.5806603449288885, -0.14239066980481405, 1.4484623937179126, -1.4851759349218694, 1.3911839347271555, -1.6531320011389408, -2.978586584568441]
CENTER_LEFT_ARM = [-0.07133691252641006, -0.052973836083405494, 1.5741805775919033, -1.4481146328076862, 1.571782540186805, -1.4891468812835686, -9.413338322697955]

def rightarm_from_leftarm(config):
  right_from_left = np.array([-1, 1, -1, 1, -1, 1, 1])
  return config*right_from_left

##################################################

# TODO: distinguish between joints and positions better

def get_position_name(tree, position_id):
    return str(tree.get_position_name(position_id))

def get_position_names(tree, position_ids):
    return [get_position_name(tree, position_id) for position_id in position_ids]

def get_position_id(tree, position_name, model_id=-1):
    for body in get_bodies(tree, model_id):
        for position_id in range(body.get_position_start_index(),
                                 body.get_position_start_index() + body.get_num_positions()):
            if get_position_name(tree, position_id) == position_name:
                return position_id
    raise ValueError(position_name)

def get_position_ids(tree, position_names, model_id=-1):
    return [get_position_id(tree, position_name, model_id) for position_name in position_names]

#def get_num_joints(tree):
#    return tree.get_num_positions() # number_of_positions

#def get_joint_id(tree, joint_name, model_id=-1):
#    return tree.findJointId(joint_name, model_id=model_id)

#def get_joint_ids(tree, joint_names, model_id=-1):
#    return [get_joint_id(tree, joint_name, model_id) for joint_name in joint_names]

def get_min_position(tree, position_id):
    return tree.joint_limit_min[position_id]

def get_max_position(tree, position_id):
    return tree.joint_limit_max[position_id]

def get_position_limits(tree, position_id):
    return get_min_position(tree, position_id), \
           get_max_position(tree, position_id)

def has_position_name(tree, position_name, model_id=-1):
    try:
        get_position_id(tree, position_name, model_id)
    except ValueError:
        return False
    return True

def has_position_names(tree, position_names, model_id=-1):
    return any(has_position_name(tree, position_name, model_id) for position_name in position_names)

def is_fixed_base(tree, model_id):
    return not has_position_names(tree, POSE_POSITIONS, model_id)

##################################################

def get_world(tree):
    #assert(tree.world() is tree.get_body(0))
    return tree.world()

def get_num_models(tree):
    return tree.get_num_model_instances()

def get_body_name(tree, body_id):
    return tree.getBodyOrFrameName(body_id)

def get_frame_name(tree, frame_id):
    return tree.getBodyOrFrameName(frame_id)

def get_body(tree, body_id):
    return tree.get_body(body_id)

def get_bodies(tree, model_id=-1):
    #if model_id != -1:
    #    return tree.FindModelInstanceBodies(model_id) # TODO: not sure this works...
    bodies = [get_body(tree, body_id) for body_id in range(tree.get_num_bodies())]
    if model_id == -1:
        return bodies
    return filter(lambda b: b.get_model_instance_id() == model_id, bodies)

def get_base_bodies(tree, model_id=-1):
    return [get_body(tree, body_id) for body_id in tree.FindBaseBodies(model_id=model_id)]

#def get_frame(tree, frame_id):
#    return get_body(tree, frame_id).???

def get_body_from_name(tree, body_name, model_id=-1):
    return tree.FindBody(body_name, model_id=model_id)

def get_frame_from_name(tree, frame_name, model_id=-1):
    return tree.findFrame(frame_name, model_id=model_id)

def get_base_body(tree, model_id):
    base_bodies = get_base_bodies(tree, model_id)
    #assert(len(base_bodies) == 1) # TODO: should make formal assumption that one root per URDF
    return base_bodies[0]

def get_model_name(tree, model_id):
    #return str(get_base_body(tree, model_id).get_model_name())
    return get_base_body(tree, model_id).get_model_name()

#def is_fixed(tree, model_index):
#    base_body = base_bodies_from_model_index(tree, model_index)[0]
#    return base_body.IsRigidlyFixedToWorld()

##################################################

def get_world_pose(tree, kin_cache, body_id):
    #base_index = 0
    #return tree.relativeTransform(kin_cache, base_index, body_index)
    return pose_from_tform(tree.CalcBodyPoseInWorldFrame(kin_cache, get_body(tree, body_id)))

def get_drake_file(rel_path):
    return os.path.join(pydrake.getDrakePath(), rel_path)

def Conf(tree):
    return tree.getZeroConfiguration()

def set_zero_positions(tree, q, position_ids):
    q[position_ids] = tree.getZeroConfigurations()[position_ids]
    return q[position_ids]

def set_random_positions(tree, q, position_ids):
    q[position_ids] = tree.getRandomConfiguration()[position_ids] # TODO: switches to gaussian when not defined
    return q[position_ids]

def set_min_positions(tree, q, position_ids):
    q[position_ids] = [get_min_position(tree, position_id) for position_id in position_ids]
    return q[position_ids]

def set_max_positions(tree, q, position_ids):
    q[position_ids] = [get_max_position(tree, position_id) for position_id in position_ids]
    return q[position_ids]

#def set_positions(tree, q, model_id, position_names, values):
#    q[get_position_ids(tree, POSE_POSITIONS, model_id)] = values
#
#def set_positions(tree, q, model_id, position_names,values):
#    q[get_position_ids(tree, POSE_POSITIONS, model_id)] = values

def set_pose(tree, q, model_id, values):
    q[get_position_ids(tree, POSE_POSITIONS, model_id)] = values

def get_pose(tree, q, model_id):
    return q[get_position_ids(tree, POSE_POSITIONS, model_id)]

##################################################

def add_model(tree, model_file, pose=None, fixed_base=True):
    model_string = open(model_file).read()
    package_map = PackageMap()
    base_type = FloatingBaseType.kFixed if fixed_base else FloatingBaseType.kRollPitchYaw

    name = 'frame' # TODO: body name is always frame
    weld_frame = None if pose is None else RigidBodyFrame(name, get_world(tree), pose) # point, euler
    #weld_frame = None
    #tree.addFrame(weld_frame)

    if model_file.endswith('.urdf'):
        base_dir = os.path.dirname(model_file)
        #base_dir = pydrake.getDrakePath()
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
    model_index = tree.get_num_model_instances() - 1
    return model_index

##################################################

class DrakeVisualizerHelper:
    def __init__(self, tree, default_q=None):
        lcm = DrakeLcm()
        self.tree = tree
        self.visualizer = DrakeVisualizer(tree=self.tree, lcm=lcm, enable_playback=True)
        if default_q is None:
            default_q = Conf(self.tree)
        self.x = np.concatenate([default_q, np.zeros(tree.get_num_velocities())])
        self.visualizer.PublishLoadRobot()
        self.draw()

    def update(self, q):
        self.x[:self.tree.get_num_positions()] = q

    def draw(self, q=None):
        if q is not None:
            self.update(q)
        context = self.visualizer.CreateDefaultContext()
        context.FixInputPort(0, BasicVector(self.x))
        self.visualizer.Publish(context)

#####################################

def homogeneous_from_points(points):
    return np.vstack([points, np.ones(points.shape[1])])

def points_from_homogenous(homogenous):
    return homogenous[:3,:]

def transform_points(tform, points):
    return points_from_homogenous(tform.dot(homogeneous_from_points(points)))

def aabb_from_points(points):
    return np.column_stack([np.min(points, axis=1), np.max(points, axis=1)])

def get_aabb_min(aabb):
    return aabb[:,0]

def get_aabb_max(aabb):
    return aabb[:,1]

def get_aabb_center(aabb):
    return (get_aabb_min(aabb) + get_aabb_max(aabb)) / 2

def get_aabb_extent(aabb):
    return (get_aabb_max(aabb) - get_aabb_min(aabb)) / 2

def aabb_union(aabbs):
    return aabb_from_points(np.hstack(aabbs))

#def aabb2d_from_aabb(aabb):
#    lower, upper = aabb
#    return lower[:2], upper[:2]

def aabb_contains(container, contained):
    return np.all(get_aabb_min(container) <= get_aabb_min(contained)) and \
           np.all(get_aabb_max(contained) <= get_aabb_max(container))

##################################################

def get_local_visual_points(body):
    return np.hstack([transform_points(element.getLocalTransform(),
                                       element.getGeometry().getPoints())
                        for element in body.get_visual_elements()])

def get_local_visual_box_points(body): # getBoundingBoxPoints returns 8 points
    return np.hstack([transform_points(element.getLocalTransform(),
                                       element.getGeometry().getBoundingBoxPoints())
                        for element in body.get_visual_elements()])

def get_body_visual_points(tree, kin_cache, body): # ComputeWorldFixedPose | IsRigidlyFixedToWorld
    #return transform_points(tree.CalcBodyPoseInWorldFrame(kin_cache, body), get_local_body_visual_points(body))
    return tree.transformPoints(kin_cache, get_local_visual_points(body),
                                body.get_body_index(), get_world(tree).get_body_index())

def get_model_visual_points(tree, kin_cache, model_id):
    return np.hstack([get_body_visual_points(tree, kin_cache, body)
                      for body in get_bodies(tree, model_id)])

def get_body_visual_aabb(tree, kin_cache, body):
    #return aabb_from_points(transform_points(tree.CalcBodyPoseInWorldFrame(kin_cache, body),
    #                                         get_local_body_visual_box_points(body)))
    return aabb_from_points(tree.transformPoints(kin_cache, get_local_visual_box_points(body),
                                                 body.get_body_index(), get_world(tree).get_body_index()))

def get_model_visual_aabb(tree, kin_cache, model_id):
    return aabb_union([get_body_visual_aabb(tree, kin_cache, body)
                      for body in get_bodies(tree, model_id)])

##################################################

def supports_body(top_body, bottom_body, epsilon=1e-2): # TODO: above / below
    top_aabb = get_lower_upper(top_body)
    bottom_aabb = get_lower_upper(bottom_body)
    top_z_min = top_aabb[0][2]
    bottom_z_max = bottom_aabb[1][2]
    return (bottom_z_max <= top_z_min <= (bottom_z_max + epsilon)) and \
           (aabb_contains(aabb2d_from_aabb(top_aabb), aabb2d_from_aabb(bottom_aabb)))

def sample_placement(tree, object_id, surface_id, max_attempts=50, epsilon=1e-3):
    # TODO: could also just do with center of mass
    #com = tree.centerOfMass(tree.doKinematics(Conf()), object_id)
    for _ in xrange(max_attempts):
        q = Conf(tree)
        [yaw] = set_random_positions(tree, q, get_position_ids(tree, [YAW_POSITION], object_id))
        euler = Euler(yaw=yaw)
        kin_cache = tree.doKinematics(q)

        object_aabb = get_model_visual_aabb(tree, kin_cache, object_id)
        surface_aabb = get_model_visual_aabb(tree, kin_cache, surface_id)
        lower = (get_aabb_min(surface_aabb) + get_aabb_extent(object_aabb))[:2]
        upper = (get_aabb_max(surface_aabb)  - get_aabb_extent(object_aabb))[:2]
        if np.any(upper < lower):
          continue
        [x, y] = np.random.uniform(lower, upper)
        z = (get_aabb_max(surface_aabb) + get_aabb_extent(object_aabb))[2] + epsilon
        point = Point(x, y, z) - get_aabb_center(object_aabb)
        return Pose(point, euler)
    return None


def colliding_bodies(tree, kin_cache, model_ids=None, min_distance=0):
    if model_ids is None:
        model_ids = range(get_num_models(tree))
    bodies = []
    for model_id in model_ids:
        bodies += get_bodies(tree, model_id)
    #body_ids = [body.get_body_index() for body in bodies]
    #phi, normal, xA, xB, bodyA_idx, bodyB_idx = tree.collisionDetect(kin_cache, body_ids, False)
    #return {(body_id1, body_id2) for distance, body_id1, body_id2 in
    #        zip(phi, bodyA_idx, bodyB_idx) if distance < min_distance}
    xA, xB, bodyA_idx, bodyB_idx = tree.allCollisions(kin_cache, False)
    return zip(bodyA_idx, bodyB_idx)

all_collision_filter = lambda b1, b2: True
none_collision_filter = lambda b1, b2: False
nonadjacent_collision_filter = lambda b1, b2: not b1.adjacentTo(b2) and not b2.adjacentTo(b1)
self_collision_filter = lambda b1, b2: b1.get_model_instance_id() == b2.get_model_instance_id()
other_collision_filter = lambda b1, b2: b1.get_model_instance_id() != b2.get_model_instance_id()

def get_disabled_collision_filter(disabled_pairs):
    def fn(b1, b2):
        if other_collision_filter(b1, b2):
            return True
        return ((b1.get_name(), b2.get_name()) not in disabled_pairs) and \
            ((b2.get_name(), b1.get_name()) not in disabled_pairs)
    return fn

def get_enabled_collision_filter(enabled_pairs):
    def fn(b1, b2):
        if other_collision_filter(b1, b2):
            return True
        return ((b1.get_name(), b2.get_name()) in enabled_pairs) or \
            ((b2.get_name(), b1.get_name()) in enabled_pairs)
    return fn

def are_colliding(tree, kin_cache, collision_filter=all_collision_filter, model_ids=None):
    # TODO: could make a separate tree for these
    if model_ids is None:
        model_ids = range(get_num_models(tree))
    for body_id1, body_id2 in colliding_bodies(tree, kin_cache, model_ids):
        body1 = get_body(tree, body_id1)
        body2 = get_body(tree, body_id2)
        if (body1.get_model_instance_id() in model_ids) and \
                (body2.get_model_instance_id() in model_ids) and \
                collision_filter(body1, body2):
            #print(get_body(tree, body_id1).get_name(), get_body(tree, body_id2).get_name())
            return True
    return False

def violates_limits(tree, q):
    return np.any(q < tree.joint_limit_min) or np.any(tree.joint_limit_max < q)

##################################################

def plan_motion(tree, initial_conf, position_ids, end_values, position_limits=None,
        collision_filter=all_collision_filter, model_ids=None, linear_only=False, **kwargs):
    assert len(position_ids) == len(end_values)
    if position_limits is None:
        position_limits = zip(tree.joint_limit_min, tree.joint_limit_max)
    assert(len(initial_conf) == len(position_limits))

    def sample_fn():
        sample = np.array([np.random.uniform(*position_limits[position_id])
                         for position_id in position_ids])
        #sample = tree.getRandomConfiguration()[position_ids]
        print(sample)
        return sample
        #return tree.getRandomConfiguration()[position_ids]

    def difference_fn(q2, q1):
        # TODO: revolute joints
        difference = []
        for joint, value2, value1 in zip(position_ids, q2, q1):
            #difference.append((value2 - value1) if is_circular(body, joint)
            #                  else circular_difference(value2, value1))
            difference.append(value2 - value1)
        return tuple(difference)

    # TODO: custom weights and step sizes
    weights = 1*np.ones(len(position_ids))
    def distance_fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))

    resolutions = 0.05*np.ones(len(position_ids))
    def extend_fn(q1, q2):
        steps = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        num_steps = int(np.max(steps)) + 1
        q = q1
        for i in xrange(num_steps):
            q = (1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            yield q
            # TODO: should wrap these joints

    def collision_fn(q):
        # Need to pass the pr2 in to this
        current_conf = initial_conf.copy()
        current_conf[position_ids] = q
        if violates_limits(tree, current_conf):
            return True
        tree.doKinematics(current_conf)
        return are_colliding(tree, tree.doKinematics(current_conf), 
            collision_filter=collision_filter, model_ids=model_ids)
        
    start_values = initial_conf[position_ids]
    if linear_only:
        return direct_path(start_values, end_values, extend_fn, collision_fn)
    return birrt(start_values, end_values, distance_fn,
                 sample_fn, extend_fn, collision_fn, **kwargs)

def load_disabled_collisions(srdf_file):
    srdf_string = open(srdf_file).read()
    regex = r'<\s*disable_collisions\s+link1="(\w+)"\s+link2="(\w+)"\s+reason="(\w+)"\s*/>'
    disabled_collisions = set()
    for link1, link2, reason in re.findall(regex, srdf_string):
        disabled_collisions.update([(link1, link2), (link2, link1)])
    return disabled_collisions

##################################################

GRASP_LENGTH = 0.04
#GRASP_LENGTH = 0.
MAX_GRASP_WIDTH = 0.07

def get_top_grasps(tree, model_id, under=False, limits=False, grasp_length=GRASP_LENGTH):
    tool_pose = pose_from_tform(PR2_TOOL_TFORM)
    #tool_pose = Pose()

    aabb = get_model_visual_aabb(tree, tree.doKinematics(Conf(tree)), model_id)
    w, l, h = 2*get_aabb_extent(aabb)
    #print(get_aabb_center(aabb)) # TODO: incorporate
    reflect_z = Pose(euler=Euler(pitch=np.pi))
    translate = Pose(point=Point(z=(h / 2 - grasp_length)))
    grasps = []
    if not limits or (w <= MAX_GRASP_WIDTH):
        for i in range(1 + under):
            rotate_z = Pose(euler=Euler(yaw=(np.pi / 2 + i * np.pi)))
            grasps += [multiply_poses(tool_pose, translate, rotate_z, reflect_z)]
    if not limits or (l <= MAX_GRASP_WIDTH):
        for i in range(1 + under):
            rotate_z = Pose(euler=Euler(yaw=(i*np.pi)))
            grasps += [multiply_poses(tool_pose, translate, rotate_z, reflect_z)]
    return grasps

def get_side_grasps(tree, model_id, under=False, limits=False, grasp_length=GRASP_LENGTH):
  tool_pose = pose_from_tform(PR2_TOOL_TFORM)
  aabb = get_model_visual_aabb(tree, tree.doKinematics(Conf(tree)), model_id)
  w, l, h = 2*get_aabb_extent(aabb)
  grasps = []
  for j in range(1 + under):
    swap_xz = Pose(euler=Euler(pitch=(-np.pi/2 + j*np.pi)))
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

def get_x_presses(body, max_orientations=1): # g_f_o
  pose = get_pose(body)
  set_pose(body, unit_pose())
  center, (w, l, h) = get_center_extent(body)
  press_poses = []
  for j in xrange(max_orientations):
      swap_xz = (unit_point(), quat_from_euler([0, -np.pi/2 + j*np.pi, 0]))
      translate = ([0, 0, w / 2], unit_quat())
      press_poses += [multiply(TOOL_POSE, translate, swap_xz)]
  set_pose(body, pose)
  return press_poses


def object_from_gripper(gripper_pose, grasp_pose):
    return multiply_poses(gripper_pose, grasp_pose)

def gripper_from_object(object_pose, grasp_pose):
    return multiply_poses(object_pose, invert_pose(grasp_pose))

def open_pr2_gripper(tree, q, model_id, gripper_name):
    set_max_positions(tree, q, get_position_ids(tree, PR2_GROUPS[gripper_name], model_id))

def close_pr2_gripper(tree, q, model_id, gripper_name):
    set_min_positions(tree, q, get_position_ids(tree, PR2_GROUPS[gripper_name], model_id))

##################################################

def inverse_kinematics(tree, frame_id, pose, q_seed=None):
    print(point_from_pose(pose))
    print(euler_from_pose(pose))
    print(frame_id)

    constraints = [
        # These three constraints ensure that the base of the robot is
        # at z = 0 and has no pitch or roll. Instead of directly
        # constraining orientation, we just require that the points at
        # [0, 0, 0], [1, 0, 0], and [0, 1, 0] in the robot's base's
        # frame must all be at z = 0 in world frame.
        # We don't care about the x or y position of the robot's base,
        # so we use NaN values to tell the IK solver not to apply a
        # constraint along those dimensions. This is equivalent to
        # placing a lower bound of -Inf and an upper bound of +Inf along
        # those axes.
        # ik.WorldPositionConstraint(robot, base_body_id,
        #                            np.array([0.0, 0.0, 0.0]),
        #                            np.array([np.nan, np.nan, 0.0]),
        #                            np.array([np.nan, np.nan, 0.0])),
        # ik.WorldPositionConstraint(robot, base_body_id,
        #                            np.array([1.0, 0.0, 0.0]),
        #                            np.array([np.nan, np.nan, 0.0]),
        #                            np.array([np.nan, np.nan, 0.0])),
        # ik.WorldPositionConstraint(robot, base_body_id,
        #                            np.array([0.0, 1.0, 0.0]),
        #                            np.array([np.nan, np.nan, 0.0]),
        #                            np.array([np.nan, np.nan, 0.0])),

        # This constraint exactly specifies the desired position of the
        # hand frame we defined earlier.


        ik.WorldPositionConstraint(tree, frame_id,
                                   np.array([0.0, 0.0, 0.0]),
                                   point_from_pose(pose),  # lower bound
                                   point_from_pose(pose)), # upper bound

        # And this specifies the orientation of that frame
        ik.WorldEulerConstraint(tree, frame_id,
                                euler_from_pose(pose),  # lower bound
                                euler_from_pose(pose)) # upper bound

        #ik.MinDistanceConstraint(robot, hand_frame_id,
        #                        min_distance,
        #                         active_bodies_idx,
        #                         active_group_names)  # upper bound
        # default

        # TODO: make a group for each unique object

    ]
    # Constraints are hard (i.e. cannot violate)

    if q_seed is None:
        q_seed = tree.getZeroConfiguration()
    options = ik.IKoptions(tree)
    #print(options.getQ()) # TODO: implement and use to set weight function
    results = ik.InverseKin(tree, q_seed, q_seed, constraints, options)

    # Each entry (only one is present in this case, since InverseKin()
    # only returns a single result) in results.info gives the output
    # status of SNOPT. info = 1 is good, anything less than 10 is OK, and
    # any info >= 10 indicates an infeasibility or failure of the
    # optimizer.
    #assert results.info[0] == 1
    print(results.q_sol)
    print(results.info) # http://drake.mit.edu/doxygen_cxx/rigid__body__ik_8h_source.html
    print(results.infeasible_constraints)
    print(q_seed.shape)
    print(results.q_sol[0].shape)

    # TODO: use Na on the configurations to constrain
    # TODO: can constraint Q weight by modifying the option

    if not results.q_sol:
        return None
    return results.q_sol[0]

    """
    IKResults,
    IKoptions,
    InverseKin,
    InverseKinTraj,
    InverseKinPointwise,
    PostureConstraint,
    RelativePositionConstraint,
    RelativeQuatConstraint,
    WorldEulerConstraint,
    WorldQuatConstraint,
    WorldGazeDirConstraint,
    WorldGazeTargetConstraint,
    RelativeGazeDirConstraint,
    MinDistanceConstraint,
    QuasiStaticConstraint,
    """
    # http://drake.mit.edu/doxygen_cxx/inverse__kinematics_8cc.html
    # http://drake.mit.edu/doxygen_cxx/rigid__body__ik_8h.html

    # IKoptions
    # InverseKinPointwise
    # InverseKinTraj
    # IKResults

##################################################

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

    grasps = get_top_grasps(tree, block1)
    #grasps = get_side_grasps(tree, block1)

    #gripper_id = get_frame_from_name(tree, PR2_TOOL_FRAMES['left_gripper'], pr2).get_frame_index()
    gripper_id = get_body_from_name(tree, PR2_TOOL_FRAMES['left_gripper'], pr2).get_body_index()
    gripper_pose = get_world_pose(tree, tree.doKinematics(q), gripper_id)

    for grasp in grasps:
        set_pose(tree, q, block1, object_from_gripper(gripper_pose, grasp))
        vis_helper.draw(q)
        raw_input('Continue?')
    return


    block1_pose = get_pose(tree, q, block1)
    print(grasp)
    print(block1_pose)
    target_gripper_pose = gripper_from_object(block1_pose, grasp)
    print(gripper_pose)

    frame_id = get_frame_from_name(tree, PR2_TOOL_FRAMES['left_gripper'], pr2).get_frame_index()
    solution = inverse_kinematics(tree, frame_id, target_gripper_pose, q_seed=q)
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
