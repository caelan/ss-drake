import time

import numpy as np

from pr2_self_collision import PR2_COLLISION_PAIRS, INITIAL_COLLISION_PAIRS
from utils import get_drake_file, get_enabled_collision_filter, \
    get_disabled_collision_filter, add_model,get_position_name, DrakeVisualizerHelper, Conf, \
    get_position_ids, get_position_limits, plan_motion, dump_tree
from pr2_utils import PR2_URDF, TABLE_SDF, PR2_GROUPS, PR2_LIMITS, REST_LEFT_ARM, \
    rightarm_from_leftarm, open_pr2_gripper, get_pr2_limits

from pydrake.multibody.rigid_body_tree import RigidBodyTree, AddFlatTerrainToWorld

IIWA_URDF = "examples/kuka_iiwa_arm/dev/box_rotation/models/dual_iiwa14_primitive_sphere_visual_collision.urdf"
IIWA_URDF = "manipulation/models/iiwa_description/urdf/iiwa14_polytope_collision.urdf"
VALKYRIE_URDF = "examples/valkyrie/urdf/urdf/valkyrie_A_sim_drake_one_neck_dof_wide_ankle_rom.urdf"
ATLAS_URDF = "examples/atlas/urdf/atlas_minimal_contact.urdf"
ATLAS_URDF = "examples/atlas/urdf/atlas_convex_hull.urdf"
JACO_URDF = "manipulation/models/jaco_description/urdf/j2n6s300.urdf"


FLAT_TERRAIN_URDF = "models/flat_terrain.urdf"

# https://github.com/RobotLocomotion/drake/tree/c84bceb37a9fa9b01f23413733446495bf843725/manipulation/models
# https://github.com/caelan/drake/tree/master/examples/kuka_iiwa_arm/models/objects

def main():
    tree = RigidBodyTree()
    robot = add_model(tree, get_drake_file(IIWA_URDF), fixed_base=True)
    ground = add_model(tree, FLAT_TERRAIN_URDF, fixed_base=True)
    #AddFlatTerrainToWorld(tree, box_size=10, box_depth=.1) # Adds visual & collision
    vis_helper = DrakeVisualizerHelper(tree)
    #vis_helper.draw()
    dump_tree(tree)

if __name__ == '__main__':
    main()
