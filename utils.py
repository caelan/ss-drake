from __future__ import absolute_import, division, print_function

import re
import os
import numpy as np
import pydrake
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

def Point(x=0, y=0, z=0):
    return np.array([x, y, z])

def Euler(roll=0, pitch=0, yaw=0):
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
    'left_gripper': ['l_gripper_l_finger_joint', 'l_gripper_r_finger_joint'],
    'right_gripper': ['r_gripper_l_finger_joint', 'r_gripper_r_finger_joint'],

}

PR2_TOOL_POSE = ([0.18, 0., 0.], [0., 0.70710678, 0., 0.70710678])
PR2_TOOL_DIRECTION = [ 0., 0., 1.]

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

def get_num_joints(tree):
    return tree.get_num_positions() # number_of_positions

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

#def get_joint_id(tree, joint_name, model_id=-1):
#    return tree.findJointId(joint_name, model_id=model_id)

#def get_joint_ids(tree, joint_names, model_id=-1):
#    return [get_joint_id(tree, joint_name, model_id) for joint_name in joint_names]

def get_min_position(tree, position_id):
    return tree.joint_limit_min[position_id]

def get_max_position(tree, position_id):
    return tree.joint_limit_max[position_id]

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
    return tree.CalcBodyPoseInWorldFrame(kin_cache, get_body(tree, body_id))

def get_drake_file(rel_path):
    return os.path.join(pydrake.getDrakePath(), rel_path)

def Conf(tree):
    return tree.getZeroConfiguration()

def set_zero_positions(tree, q, position_ids):
    values = tree.getZeroConfigurations()[position_ids]
    q[position_ids] = values
    return values

def set_random_positions(tree, q, position_ids):
    values = tree.getRandomConfiguration()[position_ids]
    q[position_ids] = values
    return values

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


def colliding_bodies(tree, kin_cache, model_ids=None, min_distance=1e-3):
    if model_ids is None:
        model_ids = range(get_num_models(tree))
    bodies = []
    for model_id in model_ids:
        bodies += get_bodies(tree, model_id)
    body_ids = [body.get_body_index() for body in bodies]
    phi, normal, xA, xB, bodyA_idx, bodyB_idx = tree.collisionDetect(kin_cache, body_ids, True)
    return {(body_id1, body_id2) for distance, body_id1, body_id2 in zip(phi, bodyA_idx, bodyB_idx) if distance < min_distance}

def are_colliding(tree, kin_cache, disabled_self_collisions=set(), **kwargs):
    # TODO: colliding models
    print()
    for body_id1, body_id2 in colliding_bodies(tree, kin_cache, **kwargs):
        body1 = get_body(tree, body_id1)
        body2 = get_body(tree, body_id2)
        if body1.get_model_instance_id() != body2.get_model_instance_id():
            return True
        if disabled_self_collisions is True:
            continue
        if (body1.get_name(), body2.get_name()) not in disabled_self_collisions:
            #print(body1.get_name(), body2.get_name())
            #return True
            pass
        else:
            print(body1.get_name(), body2.get_name())
    return False

##################################################

def load_disabled_collisions(srdf_file):
    srdf_string = open(srdf_file).read()
    regex = r'<\s*disable_collisions\s+link1="(\w+)"\s+link2="(\w+)"\s+reason="(\w+)"\s*/>'
    disabled_collisions = set()
    for link1, link2, reason in re.findall(regex, srdf_string):
        disabled_collisions.update([(link1, link2), (link2, link1)])
    return disabled_collisions

def main():
    #pr2_file = 'pr2_simplified.urdf'
    pr2_file = get_drake_file(PR2_URDF)
    table_file = get_drake_file(TABLE_SDF)
    block_file = get_drake_file(BLOCK_URDF)
    disabled_collisions = load_disabled_collisions('pr2.srdf')

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
    q[get_position_ids(tree, PR2_GROUPS['base'], pr2)] = [0, 0, np.pi/2]
    q[get_position_ids(tree, PR2_GROUPS['torso'], pr2)] = [0.2]
    #q[get_position_ids(tree, PR2_GROUPS['left_arm'], pr2)] = REST_LEFT_ARM
    #q[get_position_ids(tree, PR2_GROUPS['right_arm'], pr2)] = rightarm_from_leftarm(REST_LEFT_ARM)

    set_random_positions(tree, q, get_position_ids(tree, POSE_POSITIONS, block1))
    set_random_positions(tree, q, get_position_ids(tree, POSE_POSITIONS, block2))
    print(table1, is_fixed_base(tree, table1))
    print(block1, is_fixed_base(tree, block1))


    # body.ComputeWorldFixedPose()
    # CreateKinematicsCacheFromBodiesVector

    # TODO: problem that doesn't take into account frame

    kin_cache = tree.doKinematics(q)
    print('Colliding:', len(colliding_bodies(tree, kin_cache)))

    aabb = get_model_visual_aabb(tree, kin_cache, table1)
    print(aabb)
    print(get_aabb_min(aabb))

    while True:
        vis_helper.draw(q)
        #print('Positions:', q)
        print(are_colliding(tree, tree.doKinematics(q), disabled_self_collisions=disabled_collisions, model_ids=[pr2]))
        #print(are_colliding(tree, tree.doKinematics(q), disabled_self_collisions=set()), 
        #    are_colliding(tree, tree.doKinematics(q), disabled_self_collisions=disabled_collisions),
        #    are_colliding(tree, tree.doKinematics(q), disabled_self_collisions=True))

        raw_input('Continue?')
        print(set_random_positions(tree, q, get_position_ids(tree, PR2_GROUPS['base'], pr2)))
        q[get_position_ids(tree, POSE_POSITIONS, block1)] = sample_placement(tree, block1, table1)
        q[get_position_ids(tree, POSE_POSITIONS, block2)] = sample_placement(tree, block2, table2)



if __name__ == '__main__':
    main()
