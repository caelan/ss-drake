from __future__ import absolute_import, division, print_function

import os
import numpy as np
import pydrake
from pydrake.multibody.parsers import PackageMap
from pydrake.multibody.rigid_body_tree import (
    AddModelInstanceFromUrdfStringSearchingInRosPackages,
    FloatingBaseType,
    RigidBodyFrame,
    RigidBodyTree,
    AddFlatTerrainToWorld,
)
from pydrake.solvers import ik
from pydrake.systems.analysis import Simulator
from pydrake.multibody.rigid_body_plant import RigidBodyPlant, DrakeVisualizer, \
  CompliantMaterial, CompliantContactModelParameters
from pydrake.lcm import DrakeMockLcm, DrakeLcm, DrakeLcmInterface
from pydrake.multibody.shapes import Box, Cylinder, Mesh
from pydrake.systems.framework import BasicVector, DiagramBuilder
# TODO(eric.cousineau): Use `unittest` (after moving `ik` into `multibody`),
# declaring this as a drake_py_unittest in the BUILD.bazel file.


def load_robot_from_urdf(urdf_file):
    """
    This function demonstrates how to pass a complete
    set of arguments to Drake's URDF parser.  It is also
    possible to load a robot with a much simpler syntax
    that uses default values, such as:

      robot = RigidBodyTree(urdf_file)

    """
    urdf_string = open(urdf_file).read()
    base_dir = os.path.dirname(urdf_file)
    package_map = PackageMap()
    weld_frame = None
    floating_base_type = FloatingBaseType.kRollPitchYaw

    # Load our model from URDF
    robot = RigidBodyTree()

    AddModelInstanceFromUrdfStringSearchingInRosPackages(
        urdf_string,
        package_map,
        base_dir,
        floating_base_type,
        weld_frame,
        robot)

    return robot


urdf_file = os.path.join(
    pydrake.getDrakePath(),
    "examples/pr2/models/pr2_description/urdf/pr2_simplified.urdf")

# Load our model from URDF
robot = load_robot_from_urdf(urdf_file)

"""
# Add a convenient frame, positioned 0.1m away from the r_gripper_palm_link
# along that link's x axis
robot.addFrame(RigidBodyFrame(
    "r_hand_frame", robot.FindBody("r_gripper_palm_link"),
    np.array([0.1, 0, 0]), np.array([0., 0, 0])))

# Make sure attribute access works on bodies
assert robot.world().get_name() == "world"

hand_frame_id = robot.findFrame("r_hand_frame").get_frame_index()
base_body_id = robot.FindBody('base_footprint').get_body_index()

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
    ik.WorldPositionConstraint(robot, base_body_id,
                               np.array([0.0, 0.0, 0.0]),
                               np.array([np.nan, np.nan, 0.0]),
                               np.array([np.nan, np.nan, 0.0])),
    ik.WorldPositionConstraint(robot, base_body_id,
                               np.array([1.0, 0.0, 0.0]),
                               np.array([np.nan, np.nan, 0.0]),
                               np.array([np.nan, np.nan, 0.0])),
    ik.WorldPositionConstraint(robot, base_body_id,
                               np.array([0.0, 1.0, 0.0]),
                               np.array([np.nan, np.nan, 0.0]),
                               np.array([np.nan, np.nan, 0.0])),

    # This constraint exactly specifies the desired position of the
    # hand frame we defined earlier.
    ik.WorldPositionConstraint(robot, hand_frame_id,
                               np.array([0.0, 0.0, 0.0]),
                               np.array([0.5, 0.0, 0.6]),
                               np.array([0.5, 0.0, 0.6])),
    # And this specifies the orientation of that frame
    ik.WorldEulerConstraint(robot, hand_frame_id,
                            np.array([0.0, 0.0, 0.0]),
                            np.array([0.0, 0.0, 0.0]))
]

q_seed = robot.getZeroConfiguration()
options = ik.IKoptions(robot)
results = ik.InverseKin(robot, q_seed, q_seed, constraints, options)

# Each entry (only one is present in this case, since InverseKin()
# only returns a single result) in results.info gives the output
# status of SNOPT. info = 1 is good, anything less than 10 is OK, and
# any info >= 10 indicates an infeasibility or failure of the
# optimizer.
assert results.info[0] == 1
print("Solution")
print(repr(results.q_sol[0]))
"""

AddFlatTerrainToWorld(robot)
print("Positions:", robot.get_num_positions()) # 34 | number_of_positions
print("Velocities:", robot.get_num_velocities()) # 34 | number_of_velocities
print("Actuators:", robot.get_num_actuators()) # 28
print("Frames:", robot.get_num_frames())
print("Bodies:", robot.get_num_bodies())
world = robot.world()
print(world.get_name(), world.get_visual_elements())

plant = RigidBodyPlant(robot)
plant.set_contact_model_parameters(
                CompliantContactModelParameters())
plant.set_default_compliant_material(CompliantMaterial())
#plant = RigidBodyPlant(robot, timestep=0.1)
print(plant.GetStateVector(plant.CreateDefaultContext()).shape)
print(plant.actuator_command_input_port())
print(plant.torque_output_port())
print(plant.kinematics_results_output_port())
print(plant.contact_results_output_port())
print(plant.is_state_discrete())
print(plant.get_time_step())


"""
context = plant.CreateDefaultContext()
state = plant.GetStateVector(context)
print(state)
print(len(state))
state_vector = context.get_mutable_continuous_state_vector()
print(state_vector.CopyToVector())
print(context.get_mutable_state())
"""

print("States:", plant.get_num_states())
print("Controls:", plant.get_num_actuators())

# TODO: terrain

#box = Box(size=[0.07, 0.05, 0.15])
#element = Element(box, ...)
#visual_element = Element(box, ..., material)
#print(box.getBoundingBoxPoints())
#tree.get_visual_elements()

lcm = DrakeMockLcm()
#lcm = DrakeLcmInterface() # Just an interface
#lcm = DrakeLcm()

viz = DrakeVisualizer(tree=robot, lcm=lcm, enable_playback=False)
viz.set_publish_period(period=0.01)
viz_context = viz.CreateDefaultContext()
u0 = np.zeros(plant.get_num_actuators())
x0 = np.zeros(robot.get_num_positions() + robot.get_num_velocities())
#viz_context.FixInputPort(0, BasicVector(u0)) # This is an input to the size
viz_context.FixInputPort(0, BasicVector(x0)) # This is an input to the viewer
viz_context.set_time(0.)

lcm.StartReceiveThread()
while True:
  viz.Publish(viz_context)
  #viz.ReplayCachedSimulation()
  print('Done!')

quit()


#x0 = np.zeros(robot.get_num_positions() + robot.get_num_velocities())
#context.FixInputPort(0, BasicVector(x0))
#context.set_time(0.)
#plant.SetDefaultState(context, state)
#plant.set_state_vector(context, ...) # Or state vector directly
# https://github.com/RobotLocomotion/drake/blob/master/bindings/pydrake/multibody/rigid_body_plant_py.cc

builder = DiagramBuilder()
builder.Connect(plant.get_output_port(0), viz.get_input_port(0))
diagram = builder.Build()

#simulator = Simulator(plant)
simulator = Simulator(diagram)
integrator = simulator.get_mutable_integrator() # get_integrator
#integrator.set_maximum_step_size(1e-4)
# https://github.com/RobotLocomotion/drake/blob/master/bindings/pydrake/systems/analysis_py.cc

#simulator.set_target_realtime_rate(0)
#simulator.set_publish_every_time_step(True)
sim_context = simulator.get_mutable_context()
#sim_context = simulator.get_context()
print(sim_context)
sim_state = sim_context.get_mutable_continuous_state_vector()
#sim_state = sim_context.get_continuous_state_vector()
sim_state.SetFromVector([0.] * sim_state.size())
print(sim_context.get_state())
#print(context.get_discrete_state_vector())
#print(context.get_abstract_state())


print(sim_context.get_num_input_ports())
port_index = 0
input_port = plant.get_input_port(port_index)
print(input_port.size(), input_port.get_data_type(), input_port.get_index())

output_port = plant.get_output_port(0)
print(output_port.size(), output_port.get_index())


u0 = np.zeros(input_port.size())
#u0 = np.zeros(robot.get_num_positions() + robot.get_num_velocities())
#u0 = np.zeros(robot.get_num_positions() + robot.get_num_velocities() + 1) # No error if larger
#print(u0.shape)
# RuntimeError: RigidBodyPlant::EvaluateActuatorInputs(): ERROR: Actuator command input port for model instance 0 is not connected. All 1 actuator command input ports must be connected

sim_context.FixInputPort(port_index, BasicVector(u0))
sim_context.set_time(0.)
u = plant.EvalVectorInput(sim_context, port_index).CopyToVector()
print('U:', u.shape) # 68

# Context
# https://github.com/RobotLocomotion/drake/blob/master/bindings/pydrake/systems/framework_py.cc

# https://github.com/RobotLocomotion/drake/blob/28e5d1060e8bffa084e9db7fdabc4ef6ebd1ff7f/examples/pr2/pr2_passive_simulation.cc

lcm.StartReceiveThread()
simulator.Initialize()
print('Stepping')
#simulator.StepTo(0.001)
print('Done!')
#raw_input('Finish?')

#lcm.StopReceiveThread()