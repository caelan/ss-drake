#!/usr/bin/env python2.7

import time
import random
import numpy as np

from pr2_self_collision import PR2_COLLISION_PAIRS, INITIAL_COLLISION_PAIRS
from utils import get_drake_file, get_enabled_collision_filter, \
    get_disabled_collision_filter, add_model,get_position_name, DrakeVisualizerHelper, Conf, \
    get_position_ids, get_position_limits, plan_motion, dump_tree, set_pose, sample_placement, are_colliding, get_body_from_name, \
    get_model_position_ids, get_pose, multiply_poses, inverse_kinematics, get_world_pose, Pose, Point, set_min_positions, set_random_positions, \
    set_max_positions, set_center_positions, get_num_models, is_fixed_model, has_pose, POSE_POSITIONS
from pr2_utils import PR2_URDF, TABLE_SDF, PR2_GROUPS, PR2_LIMITS, REST_LEFT_ARM, \
    rightarm_from_leftarm, open_pr2_gripper, get_pr2_limits, BLOCK_URDF, gripper_from_object, object_from_gripper, \
    GraspInfo, get_top_grasps

from test_kuka_iiwa import IIWA_URDF, SHORT_FLOOR_URDF, GRASP_NAMES, KUKA_TOOL_FRAME
from test_pick import step_path, execute_path, convert_path
from pydrake.multibody.rigid_body_tree import RigidBodyTree, AddFlatTerrainToWorld


from ss.algorithms.dual_focused import dual_focused
from ss.algorithms.incremental import incremental
from ss.model.functions import Predicate, NonNegFunction, rename_functions, initialize, TotalCost, Increase
from ss.model.problem import Problem, get_length, get_cost
from ss.model.operators import Action, Axiom
from ss.model.streams import Stream, ListStream, GenStream, FnStream, TestStream

SINK_URDF = 'models/sink.urdf'
STOVE_URDF = 'models/sink.urdf'

# TODO: stacking

O = '?o'; O2 = '?o2'
P = '?p'; P2 = '?p2'
G = '?g'
Q = '?q'; Q2 = '?q2'
T = '?t'

IsMovable = Predicate([O])
Stackable = Predicate([O, O2])
Cleaned = Predicate([O])
Cooked = Predicate([O])
Washer = Predicate([O])
Stove = Predicate([O])

IsPose = Predicate([P])
IsGrasp = Predicate([G])
IsConf = Predicate([Q])
IsTraj = Predicate([T])

ValidPose = Predicate([O, P])
IsSupported = Predicate([P, O2])
ValidGrasp = Predicate([O, G])
IsKin = Predicate([O, P, G, Q, T])
IsMotion = Predicate([Q, Q2, T])

AtPose = Predicate([O, P])
AtConf = Predicate([Q])
HandEmpty = Predicate([])
HasGrasp = Predicate([O, G])
CanMove = Predicate([])

Holding = Predicate([O])
On = Predicate([O, O2])
Unsafe = Predicate([T])

#IsButton = Predicate([O])
#IsPress = Predicate([A, B, BQ, AT])
#IsConnected = Predicate([O, O2])

rename_functions(locals())

class PartialConf(object):
    def __init__(self, tree, positions, values):
        self.tree = tree
        self.positions = positions
        self.values = values
    def assign(self, q):
        q[self.positions] = self.values
    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)

class PartialPath(object):
    # TODO: store transforms relative to part
    def __init__(self, *args):
        self.tree, self.positions, self.path = args
    def reverse(self):
        return self.__class__(self.tree, self.positions, self.path)
    def __repr__(self):
        return 't{}'.format(id(self) % 1000)

class Commands(object):
    def __init__(self, paths):
        self.paths = paths
    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)

class Grasp(object):
    def __init__(self, model, grasp_pose, approach_pose):
        self.model = model
        self.grasp_pose = grasp_pose
        self.approach_pose = approach_pose
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

def get_motion_gen(tree, robot, fixed_models=[]):
    collision_models = [robot] + fixed_models
    def fn(conf1, conf2):
        q = Conf(tree)
        conf1.assign(q)
        sequence = [conf1.values, conf2.values]
        #sequence = plan_motion(tree, q, conf2.positions, conf2.values, model_ids=collision_models)
        #if sequence is None:
        #    return None
        path = PartialPath(tree, conf2.positions, conf2.values)
        return [path]
    return fn

def get_grasp_gen(tree, grasp_name):
    grasp_info = GRASP_NAMES[grasp_name]
    def gen(model):
        grasp_poses = grasp_info.get_grasps(tree, model)
        for grasp_pose in grasp_poses:
            grasp = Grasp(model, grasp_pose, grasp_info.approach_pose)
            yield [grasp]
    return gen

def get_stable_gen(tree):
    #q = Conf(tree)
    def gen(model, surface):
        model_positions = get_position_ids(tree, POSE_POSITIONS, model)
        while True:
            pose = sample_placement(tree, model, surface)
            if pose is None:
                continue
            model_conf = PartialConf(tree, model_positions, pose)
            yield [model_conf]
            #q[model_positions] = values
            #set_pose(tree, q, model, pose)
            # TODO: check collisions
    return gen

def get_ir_gen(tree, robot_id):
    q_default = Conf(tree)
    arm_ids = get_model_position_ids(tree, robot_id)
    def fn(model, pose, grasp):
        gripper_id = get_body_from_name(tree, KUKA_TOOL_FRAME, robot_id).get_body_index()
        gripper_pose = gripper_from_object(pose.values, grasp.grasp_pose)
        approach_pose = multiply_poses(grasp.approach_pose, gripper_pose)
        q_approach = inverse_kinematics(tree, gripper_id, approach_pose, position_ids=arm_ids)
        if (q_approach is None): # or are_colliding(tree, tree.doKinematics(q_approach)):
            return None
        conf = PartialConf(tree, arm_ids, q_approach[arm_ids])
        q_grasp = inverse_kinematics(tree, gripper_id, gripper_pose,
                                     position_ids=arm_ids, q_seed=q_approach)
        if (q_grasp is None): # or are_colliding(tree, tree.doKinematics(q_grasp)):
            return None
        path = PartialPath(tree, arm_ids, [q_approach[arm_ids], q_grasp[arm_ids]])
        return (conf, path)


        grasp_path = plan_motion(tree, q_approach, arm_ids, q_grasp[arm_ids], linear_only=True)
        if grasp_path is None:
            return None
        return (conf, path)
    return fn

def ss_from_problem(tree, q0, robot_id, bound='shared', movable_collisions=False, grasp_name='top'):
    #robot = problem.robot
    models = range(get_num_models(tree))
    movable = filter(lambda m: (m != robot_id) and has_pose(tree, m), models)
    fixed_models = filter(lambda m: (m != robot_id) and not has_pose(tree, m), models)
    print(movable)

    robot_positions = get_model_position_ids(tree, robot_id)
    conf = PartialConf(tree, robot_positions, q0[robot_positions])
    initial_atoms = [
        CanMove(),
        IsConf(conf), AtConf(conf),
        initialize(TotalCost(), 0),
    ]
    for model in movable:
        model_positions = get_position_ids(tree, POSE_POSITIONS, model)
        pose = PartialConf(tree, model_positions, q0[model_positions])
        initial_atoms += [IsMovable(model), IsPose(pose),
                          ValidPose(model, pose), AtPose(model, pose)]
        #for surface in problem.surfaces:
        #    initial_atoms += [Stackable(body, surface)]
        #    if supports_body(body, surface):
        #        initial_atoms += [IsSupported(pose, surface)]

    #initial_atoms += map(Washer, problem.sinks)
    #initial_atoms += map(Stove, problem.stoves)
    #initial_atoms += [IsConnected(*pair) for pair in problem.buttons]
    #initial_atoms += [IsButton(body) for body, _ in problem.buttons]

    goal_literals = []
    #if problem.goal_conf is not None:
    #    goal_conf = Pose(robot, problem.goal_conf)
    #    initial_atoms += [IsConf(goal_conf)]
    #    goal_literals += [AtConf(goal_conf)]
    #goal_literals += [Holding(*pair) for pair in problem.goal_holding]
    goal_literals += [Holding(model) for model in movable]

    #goal_literals += [On(*pair) for pair in problem.goal_on]
    #goal_literals += map(Cleaned, problem.goal_cleaned)
    #goal_literals += map(Cooked, problem.goal_cooked)

    #GraspedCollision = Predicate([A, G, AT], domain=[IsArm(A), POSE(G), IsArmTraj(AT)],
    #                             fn=lambda a, p, at: False, bound=False)

    PlacedCollision = Predicate([T, P], domain=[IsTraj(T), IsPose(P)],
                                fn=lambda at, p: False,
                                bound=False)

    actions = [
        Action(name='pick', param=[O, P, G, Q, T],
               pre=[IsKin(O, P, G, Q, T),
                    HandEmpty(), AtPose(O, P), AtConf(Q), ~Unsafe(T)],
               eff=[HasGrasp(O, G), CanMove(), ~HandEmpty(), ~AtPose(O, P)]),

        Action(name='place', param=[O, P, G, Q, T],
               pre=[IsKin(O, P, G, Q, T),
                    HasGrasp(O, G), AtConf(Q), ~Unsafe(T)],
               eff=[HandEmpty(), CanMove(), AtPose(O, P), ~HasGrasp(O, G)]),

        Action(name='move', param=[Q, Q2, T],
               pre=[IsMotion(Q, Q2, T),
                    CanMove(), AtConf(Q), ~Unsafe(T)],
               eff=[AtConf(Q2), ~CanMove(), ~AtConf(Q)]),

            #Action(name='clean', param=[O, O2],  # Wirelessly communicates to clean
            #      pre=[Stackable(O, O2), Washer(O2),
            #           ~Cooked(O), On(O, O2)],
            #      eff=[Cleaned(O)]),
            #
            #Action(name='cook', param=[O, O2],  # Wirelessly communicates to cook
            #      pre=[Stackable(O, O2), Stove(O2),
            #           Cleaned(O), On(O, O2)],
            #      eff=[Cooked(O), ~Cleaned(O)]),
    ]

    axioms = [
        Axiom(param=[O, G],
              pre=[ValidGrasp(O, G),
                   HasGrasp(O, G)],
              eff=Holding(O)),
        Axiom(param=[O, P, O2],
              pre=[ValidPose(O, P), IsSupported(P, O2),
                   AtPose(O, P)],
              eff=On(O, O2)),
    ]
    if movable_collisions:
        axioms += [
            Axiom(param=[O, P, T],
                  pre=[ValidPose(O, P), PlacedCollision(T, P),
                       AtPose(O, P)],
                  eff=Unsafe(T)),
        ]

    streams = [
        FnStream(name='motion', inp=[Q, Q2], domain=[IsConf(Q), IsConf(Q2)],
                 fn=get_motion_gen(tree, robot_id), out=[T],
                 graph=[IsMotion(Q, Q2, T), IsTraj(T)], bound=bound),

        Stream(name='grasp', inp=[O], domain=[IsMovable(O)],
               fn=get_grasp_gen(tree, grasp_name), out=[G],
               graph=[ValidGrasp(O, G), IsGrasp(G)], bound=bound),

        # TODO: test_support
        Stream(name='support', inp=[O, O2], domain=[Stackable(O, O2)],
               fn=get_stable_gen(tree), out=[P],
               graph=[IsPose(P), ValidPose(O, P), IsSupported(P, O2)], bound=bound),

        FnStream(name='ik', inp=[O, P, G], domain=[ValidPose(O, P), ValidGrasp(O, G)],
                  fn=get_ir_gen(tree, robot_id), out=[Q, T],
                  graph=[IsKin(O, P, G, Q, T), IsConf(Q), IsTraj(T)], bound=bound),
    ]

    return Problem(initial_atoms, goal_literals, actions, axioms, streams,
                   objective=TotalCost())


def main():
    tree = RigidBodyTree()
    robot = add_model(tree, get_drake_file(IIWA_URDF), fixed_base=True)
    sink = add_model(tree, SINK_URDF, pose=Pose(Point(x=-0.5)), fixed_base=True)
    stove = add_model(tree, STOVE_URDF, pose=Pose(Point(x=+0.5)), fixed_base=True)
    ground = add_model(tree, SHORT_FLOOR_URDF, fixed_base=True)
    block = add_model(tree, get_drake_file(BLOCK_URDF), fixed_base=False)
    dump_tree(tree)
    vis_helper = DrakeVisualizerHelper(tree)

    q0 = Conf(tree)
    set_pose(tree, q0, block, Pose(Point(y=0.5, z=0.09)))
    vis_helper.draw(q0)

    grasp_info = GRASP_NAMES['top']

    ss_problem = ss_from_problem(tree, q0, robot)
    print(ss_problem)

    plan, evaluations = dual_focused(ss_problem)
    #plan, evaluations = incremental(ss_problem, verbose=True)
    print(plan)



if __name__ == '__main__':
    main()