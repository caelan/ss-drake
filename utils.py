from __future__ import absolute_import, division, print_function

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

X_POSITION = 'base_x'
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

##################################################

def get_num_joints(tree):
    return tree.get_num_positions() # number_of_positions

def get_joint_name(tree, joint_id):
    return str(tree.get_position_name(joint_id))

def get_joint_names(tree, joint_ids):
    return [get_joint_name(tree, joint_id) for joint_id in joint_ids]

def get_joint_id(tree, joint_name, model_id=-1):
    return tree.findJointId(joint_name, model_id=model_id)

def get_joint_ids(tree, joint_names, model_id=-1):
    return [get_joint_id(tree, joint_name, model_id) for joint_name in joint_names]

def get_min_position(tree, joint_id):
    return tree.joint_limit_min[joint_id]

def get_max_position(tree, joint_id):
    return tree.joint_limit_max[joint_id]

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

def set_zero_positions(tree, q, joint_ids):
    q[joint_ids] = tree.getZeroConfigurations()[joint_ids]

def set_random_positions(tree, q, joint_ids):
    q[joint_ids] = tree.getRandomConfiguration()[joint_ids]

##################################################

def add_model(tree, model_file, pose=None, fixed_base=True):
    model_string = open(model_file).read()
    package_map = PackageMap()
    base_type = FloatingBaseType.kFixed if fixed_base else FloatingBaseType.kRollPitchYaw

    #name = 'frame' # TODO: body name is always frame
    #pose = Pose() if pose is None else pose
    #weld_frame = RigidBodyFrame(name, get_world(tree), pose) # point, euler
    weld_frame = None

    if model_file.endswith('.urdf'):
        base_dir = os.path.dirname(model_file)
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
            self.update(self.q)
        context = self.visualizer.CreateDefaultContext()
        context.FixInputPort(0, BasicVector(self.x))
        self.visualizer.Publish(context)

##################################################

def main():
    pr2_file = get_drake_file(PR2_URDF)
    table_file = get_drake_file(TABLE_SDF)
    block_file = get_drake_file(BLOCK_URDF)

    tree = RigidBodyTree()
    pr2 = add_model(tree, pr2_file, pose=None, fixed_base=True)

    #table1 = add_model(tree, table_file, pose=Pose(Point(2, 0, 0)), fixed_base=True)
    #table2 = add_model(tree, table_file, pose=Pose(Point(-2, 0, 0)), fixed_base=True)
    block1 = add_model(tree, block_file, pose=Pose(), fixed_base=False)
    #block2 = add_model(tree, block_file, pose=Pose(), fixed_base=False)
    AddFlatTerrainToWorld(tree, box_size=10, box_depth=.1) # Adds visual & collision

    print("Models:", get_num_models(tree))
    for model_id in range(get_num_models(tree)):
        print(model_id, get_model_name(tree, model_id))
        print(get_bodies(tree, model_id)) # TODO: this screws things up?

    print("Bodies:", tree.get_num_bodies())
    print("Frames:", tree.get_num_frames())
    for body_id in range(tree.get_num_bodies()):
        print(body_id, get_body_name(tree, body_id))
    #for body in get_bodies(tree):
    #    print(body.get_name())

    print("Positions:", tree.get_num_positions())
    print("Velocities:", tree.get_num_velocities())
    for joint_id in range(get_num_joints(tree)):
        print(joint_id, get_joint_name(tree, joint_id))
        print(joint_id, get_joint_id(tree, get_joint_name(tree, joint_id)))

    print("Actuators:", tree.get_num_actuators())

    vis_helper = DrakeVisualizerHelper(tree)

    q = Conf(tree)
    set_random_positions(tree, q, get_joint_ids(tree, POSE_POSITIONS))


    vis_helper.draw(q)
    print('Positions:', q)


if __name__ == '__main__':
    main()
