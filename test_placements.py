from utils import get_drake_file, add_model, DrakeVisualizerHelper, Conf, set_pose, \
    sample_placement, are_colliding, set_random_positions, get_position_ids, POSE_POSITIONS, dump_tree
from pr2_utils import TABLE_SDF, BLOCK_URDF

from pydrake.multibody.rigid_body_tree import RigidBodyTree, AddFlatTerrainToWorld

def main(num_blocks=5):
    tree = RigidBodyTree()
    table = add_model(tree, get_drake_file(TABLE_SDF), fixed_base=True)
    floor = add_model(tree, "models/flat_terrain.urdf", fixed_base=True)
    blocks = [add_model(tree, get_drake_file(BLOCK_URDF), fixed_base=False) for _ in range(num_blocks)]
    #AddFlatTerrainToWorld(tree, box_size=10, box_depth=.1) # Adds visual & collision
    dump_tree(tree)

    vis_helper = DrakeVisualizerHelper(tree)
    q = Conf(tree)
    while True:
        for block in blocks:
            pose = sample_placement(tree, block, table)
            #pose = sample_placement(tree, block, floor)
            set_pose(tree, q, block, pose)
            #position_ids =  get_position_ids(tree, POSE_POSITIONS, block)
            #set_random_positions(tree, q, position_ids)

        kin_cache = tree.doKinematics(q)
        if not are_colliding(tree, kin_cache):
            vis_helper.draw(q)
            raw_input('Next?')

if __name__ == '__main__':
    main()
