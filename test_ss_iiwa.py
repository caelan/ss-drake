#!/usr/bin/env python2.7

from pydrake.multibody.rigid_body_tree import RigidBodyTree
from ss.algorithms.dual_focused import dual_focused
from ss.model.functions import Predicate, rename_functions, initialize, TotalCost
from ss.model.operators import Action, Axiom
from ss.model.plan import print_plan
from ss.model.problem import Problem
from ss.model.streams import GenStream, FnStream

from pr2_utils import BLOCK_URDF, gripper_from_object, object_from_gripper
from test_kuka_iiwa import IIWA_URDF, SHORT_FLOOR_URDF, GRASP_NAMES, KUKA_TOOL_FRAME
from utils import get_drake_file, add_model, DrakeVisualizerHelper, Conf, \
    get_position_ids, plan_motion, dump_tree, set_pose, sample_placement, are_colliding, \
    get_body_from_name, \
    get_model_position_ids, multiply_poses, inverse_kinematics, get_world_pose, Pose, Point, \
    get_num_models, has_pose, POSE_POSITIONS, stable_z, \
    is_placement, get_model_name, get_position_bodies, get_refine_fn, SINK_URDF, STOVE_URDF, HoldingInfo

# TODO: stacking

#######################################################

O = '?o'; O2 = '?o2'
P = '?p'; P2 = '?p2'
G = '?g'
Q = '?q'; Q2 = '?q2'
T = '?t'

IsMovable = Predicate([O])
Stackable = Predicate([O, O2])
Cleaned = Predicate([O])
Cooked = Predicate([O])
Sink = Predicate([O])
Stove = Predicate([O])

IsPose = Predicate([O, P])
IsGrasp = Predicate([O, G])
IsConf = Predicate([Q])
IsTraj = Predicate([T])

IsSupported = Predicate([P, O2])
IsKin = Predicate([O, P, G, Q, T])
IsFreeMotion = Predicate([Q, Q2, T])
IsHoldingMotion = Predicate([Q, Q2, O, G, T])

AtPose = Predicate([O, P])
AtConf = Predicate([Q])
HandEmpty = Predicate([])
HasGrasp = Predicate([O, G])
CanMove = Predicate([])

Holding = Predicate([O])
On = Predicate([O, O2])
Unsafe = Predicate([T])

rename_functions(locals())

#######################################################

class Grasp(object):
    def __init__(self, model, grasp_pose, approach_pose):
        self.model = model
        self.grasp_pose = grasp_pose
        self.approach_pose = approach_pose
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

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
    def __init__(self, tree, positions, sequence, holding=[]):
        self.tree = tree
        self.positions = positions
        self.sequence = sequence
        self.holding = holding
    def model_ids(self):
        controlled_bodies = get_position_bodies(self.tree, self.positions)
        actuated_ids = [body.get_model_instance_id() for body in controlled_bodies]
        holding_ids = [info.model_id for info in self.holding]
        return list(set(actuated_ids + holding_ids))
    #def partial_confs(self): # TODO: holding
    #    return [PartialConf(self.tree, self.positions, values) for values in self.sequence]
    def full_path(self, q0=None):
        if q0 is None:
            q0 = Conf(self.tree)
        new_path = []
        for values in self.sequence:
            q = q0.copy()
            q[self.positions] = values
            if self.holding: # TODO: cache this
                kin_cache = self.tree.doKinematics(q)
                for body_id, grasp_pose, model_id in self.holding:
                    body_pose = get_world_pose(self.tree, kin_cache, body_id)
                    model_pose = object_from_gripper(body_pose, grasp_pose)
                    set_pose(self.tree, q, model_id, model_pose)
            new_path.append(q)
        return new_path
    def refine(self, num_steps=0):
        refine_fn = get_refine_fn(self.positions, num_steps)
        new_sequence = []
        for v1, v2 in zip(self.sequence, self.sequence[1:]):
            new_sequence += list(refine_fn(v1, v2))
        return self.__class__(self.tree, self.positions, new_sequence, self.holding)
    def reverse(self):
        return self.__class__(self.tree, self.positions, self.sequence[::-1], self.holding)
    def __repr__(self):
        return 't{}'.format(id(self) % 1000)

class Command(object):
    def __init__(self, partial_paths):
        self.partial_paths = partial_paths
    def full_path(self, q0=None):
        if q0 is None:
            q0 = Conf(self.tree)
        new_path = [q0]
        for partial_path in self.partial_paths:
            new_path += partial_path.full_path(new_path[-1])[1:]
        return new_path
    def refine(self, num_steps=0):
        return self.__class__([partial_path.refine(num_steps) for partial_path in self.partial_paths])
    def reverse(self):
        return self.__class__([partial_path.reverse() for partial_path in reversed(self.partial_paths)])
    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)

#######################################################

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
            yield (model_conf,)
            #q[model_positions] = values
            #set_pose(tree, q, model, pose)
            # TODO: check collisions
    return gen

def get_ik_fn(tree, robot_id, fixed_ids=[], teleport=True):
    #q_default = Conf(tree)
    position_ids = get_model_position_ids(tree, robot_id)
    model_ids = ([robot_id] + fixed_ids)
    gripper_id = get_body_from_name(tree, KUKA_TOOL_FRAME, robot_id).get_body_index()
    def fn(model, pose, grasp):
        gripper_pose = gripper_from_object(pose.values, grasp.grasp_pose)
        approach_pose = multiply_poses(grasp.approach_pose, gripper_pose)
        q_approach = inverse_kinematics(tree, gripper_id, approach_pose, position_ids=position_ids)
        if (q_approach is None) or are_colliding(tree, tree.doKinematics(q_approach), model_ids=model_ids):
            return None
        conf = PartialConf(tree, position_ids, q_approach[position_ids])
        q_grasp = inverse_kinematics(tree, gripper_id, gripper_pose, position_ids=position_ids, q_seed=q_approach)
        if (q_grasp is None) or are_colliding(tree, tree.doKinematics(q_grasp), model_ids=model_ids):
            return None
        holding_info = HoldingInfo(gripper_id, grasp.grasp_pose, model)
        if teleport:
            sequence = [q_approach[position_ids], q_grasp[position_ids]]
        else:
            sequence = plan_motion(tree, q_approach, position_ids, q_grasp[position_ids], model_ids=model_ids, linear_only=True)
            if sequence is None:
                raw_input('Approach motion failed')
                return None
        command = Command([PartialPath(tree, position_ids, sequence),
                        PartialPath(tree, position_ids, sequence[::-1], holding=[holding_info])])
        return (conf, command)
        # TODO: holding collisions
    return fn

def get_motion_gen(tree, robot_id, fixed_ids=[], teleport=True):
    model_ids = ([robot_id] + fixed_ids)
    def fn(conf1, conf2):
        assert(conf1.positions == conf2.positions)
        if teleport:
            sequence = [conf1.values, conf2.values]
        else:
            q = Conf(tree)
            conf1.assign(q)
            sequence = plan_motion(tree, q, conf2.positions, conf2.values, model_ids=model_ids)
            if sequence is None:
                raw_input('Free motion failed')
                return None
        command = Command([PartialPath(tree, conf2.positions, sequence)])
        return (command,)
    return fn

def get_holding_motion_gen(tree, robot_id, fixed_ids=[], teleport=True):
    model_ids = ([robot_id] + fixed_ids)
    gripper_id = get_body_from_name(tree, KUKA_TOOL_FRAME, robot_id).get_body_index()
    def fn(conf1, conf2, model, grasp):
        assert(conf1.positions == conf2.positions)
        holding_info = HoldingInfo(gripper_id, grasp.grasp_pose, model)
        if teleport:
            sequence = [conf1.values, conf2.values]
        else:
            q = Conf(tree)
            conf1.assign(q)
            sequence = plan_motion(tree, q, conf2.positions, conf2.values, model_ids=model_ids)
            if sequence is None:
                raw_input('Holding motion failed')
                return None
        command = Command([PartialPath(tree, conf2.positions, sequence, holding=[holding_info])])
        return (command,)
    return fn


def get_movable_collision_test(tree):
    def test(command, model, pose):
        for partial_path in command.partial_paths:
            if any(info.model_id == model for info in partial_path.holding):
                continue # Cannot collide with itself
            # TODO: cache the KinematicsCaches
            q = Conf(tree)
            pose.assign(q) # TODO: compute kinematics trees just for pairs/triplets of objects
            model_ids = partial_path.model_ids() + [model]
            for q in command.full_path(q):
                if are_colliding(tree, tree.doKinematics(q), model_ids=model_ids):
                    raw_input('Movable collision')
                    return True
        return False
    return test

#######################################################

def ss_from_problem(tree, q0, robot_id, bound='shared',
                    teleport=False, movable_collisions=False, grasp_name='top'):
    kin_cache = tree.doKinematics(q0)
    assert(not are_colliding(tree, kin_cache))

    rigid_ids = [m for m in range(get_num_models(tree)) if m != robot_id]
    movable_ids = filter(lambda m: has_pose(tree, m), rigid_ids)
    fixed_ids = filter(lambda m: m not in movable_ids, rigid_ids)
    print('Robot:', robot_id)
    print('Movable:', movable_ids)
    print('Fixed:', fixed_ids)

    robot_positions = get_model_position_ids(tree, robot_id)
    conf = PartialConf(tree, robot_positions, q0[robot_positions])
    initial_atoms = [
        HandEmpty(), CanMove(),
        IsConf(conf), AtConf(conf),
        initialize(TotalCost(), 0),
    ]
    for model in movable_ids:
        model_positions = get_position_ids(tree, POSE_POSITIONS, model)
        pose = PartialConf(tree, model_positions, q0[model_positions])
        initial_atoms += [IsMovable(model), IsPose(model, pose), AtPose(model, pose)]
        for surface in fixed_ids:
            initial_atoms += [Stackable(model, surface)]
            if is_placement(tree, kin_cache, model, surface):
                initial_atoms += [IsSupported(pose, surface)]

    for model in rigid_ids:
        name = get_model_name(tree, model)
        if 'sink' in name:
            initial_atoms.append(Sink(model))
        if 'stove' in name:
            initial_atoms.append(Stove(model))

    model = movable_ids[0]
    goal_literals = [
        AtConf(conf),
        #Holding(model),
        #On(model, fixed_ids[0]),
        #On(model, fixed_ids[2]),
        #Cleaned(model),
        Cooked(model),
    ]
 
    PlacedCollision = Predicate([T, O, P], domain=[IsTraj(T), IsPose(O, P)],
                                fn=get_movable_collision_test(tree),
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

        # Can just do move if we check holding collisions
        Action(name='move_free', param=[Q, Q2, T],
               pre=[IsFreeMotion(Q, Q2, T),
                    HandEmpty(), CanMove(), AtConf(Q), ~Unsafe(T)],
               eff=[AtConf(Q2), ~CanMove(), ~AtConf(Q)]),

        Action(name='move_holding', param=[Q, Q2, O, G, T],
               pre=[IsHoldingMotion(Q, Q2, O, G, T),
                    HasGrasp(O, G), CanMove(), AtConf(Q), ~Unsafe(T)],
               eff=[AtConf(Q2), ~CanMove(), ~AtConf(Q)]),

        Action(name='clean', param=[O, O2],  # Wirelessly communicates to clean
             pre=[Stackable(O, O2), Sink(O2),
                  ~Cooked(O), On(O, O2)],
             eff=[Cleaned(O)]),
        
        Action(name='cook', param=[O, O2],  # Wirelessly communicates to cook
             pre=[Stackable(O, O2), Stove(O2),
                  Cleaned(O), On(O, O2)],
             eff=[Cooked(O), ~Cleaned(O)]),
    ]

    axioms = [
        Axiom(param=[O, G],
              pre=[IsGrasp(O, G),
                   HasGrasp(O, G)],
              eff=Holding(O)),
        Axiom(param=[O, P, O2],
              pre=[IsPose(O, P), IsSupported(P, O2),
                   AtPose(O, P)],
              eff=On(O, O2)),
    ]
    if movable_collisions:
        axioms += [
            Axiom(param=[T, O, P],
                  pre=[IsPose(O, P), PlacedCollision(T, O, P),
                       AtPose(O, P)],
                  eff=Unsafe(T)),
        ]

    streams = [
        GenStream(name='grasp', inp=[O], domain=[IsMovable(O)],
                  fn=get_grasp_gen(tree, grasp_name), out=[G],
                  graph=[IsGrasp(O, G)], bound=bound),

        # TODO: test_support
        GenStream(name='support', inp=[O, O2], domain=[Stackable(O, O2)],
                  fn=get_stable_gen(tree), out=[P],
                  graph=[IsPose(O, P), IsSupported(P, O2)], bound=bound),

        FnStream(name='inverse_kin', inp=[O, P, G], domain=[IsPose(O, P), IsGrasp(O, G)],
                 fn=get_ik_fn(tree, robot_id, fixed_ids=fixed_ids, teleport=teleport), out=[Q, T],
                 graph=[IsKin(O, P, G, Q, T), IsConf(Q), IsTraj(T)], bound=bound),

        FnStream(name='free_motion', inp=[Q, Q2], domain=[IsConf(Q), IsConf(Q2)],
                 fn=get_motion_gen(tree, robot_id, teleport=teleport), out=[T],
                 graph=[IsFreeMotion(Q, Q2, T), IsTraj(T)], bound=bound),
    
        FnStream(name='holding_motion', inp=[Q, Q2, O, G], domain=[IsConf(Q), IsConf(Q2), IsGrasp(O, G)],
                 fn=get_holding_motion_gen(tree, robot_id, teleport=teleport), out=[T],
                 graph=[IsHoldingMotion(Q, Q2, O, G, T), IsTraj(T)], bound=bound),
    ]

    return Problem(initial_atoms, goal_literals, actions, axioms, streams,
                   objective=TotalCost())

#######################################################

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
    set_pose(tree, q0, block, Pose(Point(y=0.5, z=stable_z(tree, block, ground))))
    vis_helper.draw(q0)

    ss_problem = ss_from_problem(tree, q0, robot, teleport=False, movable_collisions=True)
    print(ss_problem)
    #ss_problem = ss_problem.teleport_problem()
    #print(ss_problem)

    plan, evaluations = dual_focused(ss_problem, verbose=True)
    #plan, evaluations = incremental(ss_problem, verbose=True)
    print_plan(plan, evaluations)
    if plan is None:
        return

    commands = []
    for action, args in plan:
        command = args[-1]
        if action.name == 'place':
            commands.append(command.reverse())
        elif action.name in ['move_free', 'move_holding', 'pick']:
            commands.append(command)
    print(commands)

    full_path = Command(commands).refine(num_steps=10).full_path(q0) # TODO: generator?
    raw_input('Execute?')
    vis_helper.execute_sequence(full_path, time_step=0.005)
    vis_helper.step_sequence(full_path)


if __name__ == '__main__':
    main()
