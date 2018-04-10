from __future__ import absolute_import, division, print_function

import os
import re
import numpy as np
import pydrake
import time

from motion_planners.rrt_connect import birrt, direct_path
from pydrake.lcm import DrakeLcm
from pydrake.multibody.parsers import PackageMap
from pydrake.multibody.rigid_body_plant import DrakeVisualizer
from pydrake.multibody.rigid_body_tree import (
    AddModelInstanceFromUrdfStringSearchingInRosPackages,
    AddModelInstancesFromSdfStringSearchingInRosPackages,
    FloatingBaseType,
    RigidBodyFrame,
)
from pydrake.solvers import ik
from pydrake.systems.framework import BasicVector

from transformations import euler_matrix, euler_from_matrix

X_POSITION = 'base_x' # base_x | weld_x
Y_POSITION = 'base_y'
Z_POSITION = 'base_z'
ROLL_POSITION = 'base_roll'
PITCH_POSITION = 'base_pitch'
YAW_POSITION = 'base_yaw'

POINT_POSITIONS = [X_POSITION, Y_POSITION, Z_POSITION]
EULER_POSITIONS = [ROLL_POSITION, PITCH_POSITION, YAW_POSITION]
POSE_POSITIONS = POINT_POSITIONS + EULER_POSITIONS
POSE2D_POSITIONS = [X_POSITION, Y_POSITION, YAW_POSITION]

EULER_AXES = 'sxyz' # sxyz | rxyz
REVOLUTE_LIMITS = (-np.pi, np.pi)

def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])

def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])

def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return np.concatenate([point, euler])

def Pose2d(x=0., y=0., yaw=0.):
    return np.array([x, y, yaw])

def point_from_pose(pose):
    return pose[:3]

def euler_from_pose(pose):
    return pose[3:]

def point_euler_from_pose(pose):
    return point_from_pose(pose), euler_from_pose(pose)

def rot_from_euler(euler):
    return euler_matrix(*euler, axes=EULER_AXES)[:3,:3]

def euler_from_rot(rot):
    return np.array(euler_from_matrix(rot, axes=EULER_AXES))

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

def unit_from_theta(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)

def pose2d_from_pose(pose):
    x, y, z = point_from_pose(pose)
    roll, pitch, yaw = euler_from_pose(pose)
    assert (abs(roll) < 1e-3) and (abs(pitch) < 1e-3)
    return np.array([x, y, yaw])

def pose_from_pose2d(pose2d, default_pose=None):
    if default_pose is None:
        default_pose = Pose()
    x, y, yaw = pose2d
    _, _, z = point_from_pose(default_pose)
    roll, pitch, _ = euler_from_pose(default_pose)
    return Pose(Point(x, y, z), Euler(roll, pitch, yaw))

##################################################

def get_element_color(element):
    return element.getMaterial()

def set_element_color(element, color):
    return element.setMaterial(color)

def set_body_color(body, color):
    for element in body.get_visual_elements():
        set_element_color(element, color)

# TODO: bind new methods for creating bodies

##################################################

# TODO: distinguish between joints and positions better

def get_position_name(tree, position_id):
    return str(tree.get_position_name(position_id))

def get_position_names(tree, position_ids):
    return [get_position_name(tree, position_id) for position_id in position_ids]

def get_body_position_ids(body):
    return range(body.get_position_start_index(), body.get_position_start_index() + body.get_num_positions())

def get_model_position_ids(tree, model_id=-1):
    position_ids = []
    for body in get_bodies(tree, model_id):
        position_ids += get_body_position_ids(body)
    return position_ids

def get_model_joint_ids(tree, model_id=-1):
    return filter(lambda i: get_position_name(tree, i) not in POSE_POSITIONS, get_model_position_ids(tree, model_id))

def get_position_id(tree, position_name, model_id=-1):
    for position_id in get_model_position_ids(tree, model_id):
        if get_position_name(tree, position_id) == position_name:
            return position_id
    raise ValueError(position_name)

def get_position_ids(tree, position_names, model_id=-1):
    return [get_position_id(tree, position_name, model_id) for position_name in position_names]

def get_position_body(tree, position_id):
    for body in get_bodies(tree):
        if position_id in get_body_position_ids(body):
            return body
    raise ValueError(position_id)

def get_position_bodies(tree, position_ids):
    return [get_position_body(tree, position_id) for position_id in position_ids]

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

def wrap_position(tree, position_id, value, revolute_names=set()):
    if get_position_name(tree, position_id) in revolute_names:
        return wrap_angle(value)
    return value

def wrap_positions(tree, q, revolute_names=set()):
    return np.array([wrap_position(tree, position_id, value, revolute_names)
                     for position_id, value in enumerate(q)])

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
    # Indexing at 1 to exclude the world
    bodies = [get_body(tree, body_id) for body_id in range(1, tree.get_num_bodies())]
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

def is_fixed_model(tree, model_id):
    return get_base_body(tree, model_id).IsRigidlyFixedToWorld()

def is_connected_model(tree, model_id):
    return len(get_base_bodies(tree, model_id)) <= 1

def is_rigid_model(tree, model_id):
    return not get_model_position_ids(tree, model_id)

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

def set_center_positions(tree, q, position_ids):
    q[position_ids] = [(get_min_position(tree, position_id) + get_max_position(tree, position_id))/2. for position_id in position_ids]
    return q[position_ids]

def set_max_positions(tree, q, position_ids):
    q[position_ids] = [get_max_position(tree, position_id) for position_id in position_ids]
    return q[position_ids]

#def set_positions(tree, q, model_id, position_names, values):
#    q[get_position_ids(tree, POSE_POSITIONS, model_id)] = values
#
#def set_positions(tree, q, model_id, position_names,values):
#    q[get_position_ids(tree, POSE_POSITIONS, model_id)] = values

##################################################

def has_pose(tree, model_id):
    return has_position_names(tree, POSE_POSITIONS, model_id)

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

def load_disabled_collisions(srdf_file):
    srdf_string = open(srdf_file).read()
    regex = r'<\s*disable_collisions\s+link1="(\w+)"\s+link2="(\w+)"\s+reason="(\w+)"\s*/>'
    disabled_collisions = set()
    for link1, link2, reason in re.findall(regex, srdf_string):
        disabled_collisions.update([(link1, link2), (link2, link1)])
    return disabled_collisions

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

    def step_sequence(self, sequence):
        for i, q in enumerate(sequence):
            self.draw(q)
            raw_input('{}) step?'.format(i))

    def execute_sequence(self, sequence, time_step=0.05):
        for i, q in enumerate(sequence):
            self.draw(q)
            time.sleep(time_step)

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

def aabb2d_from_aabb(aabb):
    return aabb[:2,:]

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

def sample_nearby_pose2d(target_point, radius_range=(0.25, 1.0)):
    radius = np.random.uniform(*radius_range)
    theta = np.random.uniform(*REVOLUTE_LIMITS)
    x, y = radius*unit_from_theta(theta) + target_point[:2]
    yaw = np.random.uniform(*REVOLUTE_LIMITS)
    return Pose2d(x, y, yaw)

def is_placement(tree, kin_cache, object_id, surface_id, epsilon=1e-2): # TODO: above / below
    object_aabb = get_model_visual_aabb(tree, kin_cache, object_id)
    surface_aabb = get_model_visual_aabb(tree, kin_cache, surface_id)
    surface_z_max = get_aabb_max(surface_aabb)[2]
    return (surface_z_max <= get_aabb_min(object_aabb)[2] <= (surface_z_max + epsilon)) and \
           (aabb_contains(aabb2d_from_aabb(surface_aabb), aabb2d_from_aabb(object_aabb)))

PLACEMENT_OFFSET = 1e-3

def stable_z(tree, object_id, surface_id, q=None, epsilon=PLACEMENT_OFFSET):
    if q is None:
        q = Conf(tree)
    kin_cache = tree.doKinematics(q)
    object_aabb = get_model_visual_aabb(tree, kin_cache, object_id)
    surface_aabb = get_model_visual_aabb(tree, kin_cache, surface_id)
    return (get_aabb_max(surface_aabb) + get_aabb_extent(object_aabb))[2] + epsilon

def sample_placement(tree, object_id, surface_id, max_attempts=50, epsilon=PLACEMENT_OFFSET):
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

##################################################

def get_sample_fn(position_ids, position_limits):
    def fn():
        return np.array([np.random.uniform(*position_limits[position_id])
                         for position_id in position_ids])
        #return tree.getRandomConfiguration()[position_ids]
    return fn

def get_difference_fn(position_ids):
    def fn(q2, q1):
        # TODO: revolute joints
        difference = []
        for joint, value2, value1 in zip(position_ids, q2, q1):
            #difference.append((value2 - value1) if is_circular(body, joint)
            #                  else circular_difference(value2, value1))
            difference.append(value2 - value1)
        return tuple(difference)
    return fn

def get_distance_fn(position_ids, weights=None):
    if weights is None:
        weights = 1*np.ones(len(position_ids))
    # TODO: custom weights and step sizes
    difference_fn = get_difference_fn(position_ids)
    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
    return fn

def get_refine_fn(position_ids, num_steps=0):
    difference_fn = get_difference_fn(position_ids)
    num_steps = num_steps + 1
    def fn(q1, q2):
        q = q1
        for i in range(num_steps):
            q = (1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            yield q
            # TODO: should wrap these joints
    return fn

def get_extend_fn(position_ids, resolutions=None):
    if resolutions is None:
        resolutions = 0.05*np.ones(len(position_ids))
    difference_fn = get_difference_fn(position_ids)
    def fn(q1, q2):
        steps = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        num_steps = int(np.max(steps)) + 1
        refine_fn = get_refine_fn(position_ids, num_steps=int(np.max(steps)))
        return refine_fn(q1, q2)
    return fn

def plan_motion(tree, initial_conf, position_ids, end_values, position_limits=None,
        collision_filter=all_collision_filter, model_ids=None, linear_only=False, **kwargs):
    assert(len(position_ids) == len(end_values))
    if position_limits is None:
        position_limits = zip(tree.joint_limit_min, tree.joint_limit_max)
    assert(len(initial_conf) == len(position_limits))

    sample_fn = get_sample_fn(position_ids, position_limits)
    distance_fn = get_distance_fn(position_ids)
    extend_fn = get_extend_fn(position_ids)

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

##################################################

def inverse_kinematics(tree, frame_id, pose, position_ids=None, q_seed=None, epsilon=0, tolerance=1e-3):
    if q_seed is None:
        q_seed = tree.getZeroConfiguration()
    if position_ids is None:
        position_ids = range(tree.get_num_positions())
    position_ids = set(position_ids)

    indices = range(tree.get_num_positions())
    lower_limits = [get_min_position(tree, position_id) if (position_id in position_ids) else q_seed[position_id]
        for position_id in indices]
    upper_limits = [get_max_position(tree, position_id) if (position_id in position_ids) else q_seed[position_id] 
        for position_id in indices]
    #lower_limits = [REVOLUTE_LIMITS[0] if value == -np.inf else value for value in lower_limits]
    #upper_limits = [REVOLUTE_LIMITS[1] if value == np.inf else value for value in upper_limits]
    posture_constraint = ik.PostureConstraint(tree)
    posture_constraint.setJointLimits(indices, lower_limits, upper_limits)

    min_distance = None
    active_bodies_idx = []
    active_group_names = set()

    constraints = [
        posture_constraint,

        # so we use NaN values to tell the IK solver not to apply a
        # constraint along those dimensions. This is equivalent to
        # placing a lower bound of -Inf and an upper bound of +Inf along
        # those axes.
        ik.WorldPositionConstraint(tree, frame_id,
                                   np.array([0.0, 0.0, 0.0]),
                                   point_from_pose(pose) - epsilon*np.zeros(3),  # lower bound
                                   point_from_pose(pose) + epsilon*np.zeros(3)), # upper bound
        ik.WorldEulerConstraint(tree, frame_id,
                                euler_from_pose(pose) - epsilon*np.zeros(3),  # lower bound
                                euler_from_pose(pose) + epsilon*np.zeros(3)), # upper bound
        # TODO: make a group for each unique object
    ]
    if min_distance is not None:
        constraints += [
            ik.MinDistanceConstraint(tree, # TODO: doesn't seem to work yet
                                    min_distance,
                                    active_bodies_idx,
                                    active_group_names)]

    options = ik.IKoptions(tree)
    #print(options.getQ()) # TODO: implement and use to set weight function
    results = ik.InverseKin(tree, q_seed, q_seed, constraints, options)

    # info = 1 is good, anything less than 10 is OK, and any info >= 10 indicates an infeasibility or failure of the optimizer.
    # TODO: can constraint Q weight by modifying the option
    # TODO: check that the solution is close enough here

    [info] = results.info # http://drake.mit.edu/doxygen_cxx/rigid__body__ik_8h_source.html
    if 10 <= info:
        return None
    #print('Success: {} | Info: {} | Infeasible: {}'.format(success, info, len(results.infeasible_constraints)))
    q_solution = results.q_sol[0]
    frame_pose = get_world_pose(tree, tree.doKinematics(q_solution), frame_id)
    #print(frame_pose - pose)
    #print(np.isclose(frame_pose, pose, atol=tolerance, rtol=0))
    if not np.allclose(frame_pose, pose, atol=tolerance, rtol=0):
        return None
    return q_solution

##################################################

def dump_tree(tree):
    print("Models:", get_num_models(tree))
    for model_id in range(get_num_models(tree)):
        print(model_id, get_model_name(tree, model_id))

    print("Bodies:", tree.get_num_bodies())
    print("Frames:", tree.get_num_frames())
    for body in get_bodies(tree):
        print(body.get_body_index(), body.get_name(), body.IsRigidlyFixedToWorld()) #body.get_group_to_collision_ids_map().keys())

    print("Positions:", tree.get_num_positions())
    print("Velocities:", tree.get_num_velocities())
    for position_id in range(tree.get_num_positions()):
        print(position_id, get_position_name(tree, position_id),
            get_min_position(tree, position_id), get_max_position(tree, position_id))

    print("Actuators:", tree.get_num_actuators())