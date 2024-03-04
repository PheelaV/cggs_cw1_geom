import os
import polyscope as ps
import numpy as np
from skimage import measure
from ReconstructionFunctions import load_off_file, compute_RBF_weights, evaluate_RBF, polyharmonic, biharmonic, Wendland
from functools import partial


if __name__ == '__main__':
    ps.init()

    points_init = 0.05
    points_end = 0.15
    file_name, epsilon, RBF_function = 'bunny-500.off', 0.05, polyharmonic

    # points_init = 0.1
    # points_end = 0.15
    # file_name, epsilon, RBF_function = 'dragon-3000.off', 0.05, polyharmonic
    # file_name, epsilon, RBF_function = 'fertility-2500.off', 0.05, polyharmonic

    # inputPointNormals, _ = load_off_file(os.path.join('..', 'data', 'fertility-2500.off'))
    inputPointNormals, _ = load_off_file(os.path.join('..', 'data', file_name))
    inputPoints = inputPointNormals[:, 0:3]
    inputNormals = inputPointNormals[:, 3:6]

    # normalizing point cloud to be centered on [0,0,0] and between [-0.9, 0.9]
    inputPoints -= np.mean(inputPoints, axis=0)
    min_coords = np.min(inputPoints, axis=0)
    max_coords = np.max(inputPoints, axis=0)
    scale_factor = 0.9 / np.max(np.abs(inputPoints))
    inputPoints = inputPoints * scale_factor

    ps_cloud = ps.register_point_cloud("Input points", inputPoints)
    ps_cloud.add_vector_quantity("Input Normals", inputNormals)

    # Parameters
    gridExtent = 1 #the dimensions of the evaluation grid for marching cubes
    res = 50 #the resolution of the grid (number of nodes)

    # Generating and registering the grid
    gridDims = (res, res, res)
    bound_low = (-gridExtent, -gridExtent, -gridExtent)
    bound_high = (gridExtent, gridExtent, gridExtent)
    ps_grid = ps.register_volume_grid("Sampled Grid", gridDims, bound_low, bound_high, enabled=False)

    X, Y, Z = np.meshgrid(np.linspace(-gridExtent, gridExtent, res),
                          np.linspace(-gridExtent, gridExtent, res),
                          np.linspace(-gridExtent, gridExtent, res), indexing='ij')

    #the list of points to be fed into evaluate_RBF
    xyz = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    ##########################
    ## you code of computation and evaluation goes here

    no_points = inputPoints.shape[0]
    use_off_points = True
    
    w_golden, RBF_centers_golden, _ = compute_RBF_weights(inputPoints.copy(), inputNormals, RBF_function, epsilon, useOffPoints=use_off_points, l=-1)
    rbf_iter_golden = evaluate_RBF(inputPoints, RBF_centers_golden, RBF_function, w_golden)

    no_points_init = int(no_points * points_init)
    no_points_end = int(no_points * points_end)


    included = [x for x in np.random.choice(no_points, no_points_init, replace=False)]

    while len(included) <= no_points_end:
        rbf_renctre_indices = np.array(included)
        w, RBF_centers, _ = compute_RBF_weights(inputPoints.copy(), inputNormals, RBF_function, epsilon, useOffPoints=use_off_points, RBFCentreIndices=rbf_renctre_indices, l=-1)
        rbf_iter_error = evaluate_RBF(inputPoints, RBF_centers, RBF_function, w)

        # select the largest error point that is not included
        # Create a boolean mask initialized to True
        mask = np.ones(rbf_iter_error.shape, dtype=bool)
        # Set selected indices to False
        mask[included] = False
        # Use the mask to access elements
        error_index = np.argmax(np.abs(rbf_iter_golden[mask] - rbf_iter_error[mask]))
        error = np.abs(rbf_iter_golden[mask][error_index] - rbf_iter_error[mask][error_index])
        included.append(np.arange(rbf_iter_golden.shape[0])[mask][error_index])
        print(f"Error: {error:.4f}, no points: {len(included)}, error index: {error_index}")
    print(f"Final size: {len(included)}")
    #########################

    w, RBF_centers, _ = compute_RBF_weights(inputPoints, inputNormals, RBF_function, epsilon, useOffPoints=use_off_points, RBFCentreIndices=included, l=-1)
    RBFValues = evaluate_RBF(xyz, RBF_centers, RBF_function, w)

    #fitting to grid shape again
    RBFValues = np.reshape(RBFValues, X.shape)

    # Registering the grid representing the implicit function
    ps_grid.add_scalar_quantity("Implicit Function", RBFValues, defined_on='nodes',
                                datatype="standard", enabled=True)

    # Computing marching cubes and realigning result to sit on point cloud exactly
    vertices, faces, _, _ = measure.marching_cubes(RBFValues, spacing=(
        2.0 * gridExtent / float(res - 1), 2.0 * gridExtent / float(res - 1), 2.0 * gridExtent / float(res - 1)),
                                                   level=0.0)
    vertices -= gridExtent
    ps.register_surface_mesh("Marching-Cubes Surface", vertices, faces)

    ps.show()

    # for currEpsilonIndex in range(len(epsilonRange)):
    #     root, old_extension = os.path.splitext(off_file_path)
    #     pickle_file_path = root + '-basic-version-eps-' + str(currEpsilonIndex) + '.data'
    #     with open(pickle_file_path, 'rb') as pickle_file:
    #         loaded_data = pickle.load(pickle_file)

    #     w, RBFCentres, _ = compute_RBF_weights(loaded_data['inputPoints'], loaded_data['inputNormals'],
    #                                             polyharmonic,
    #                                             loaded_data['currEpsilon'])

    #     RBFValues = evaluate_RBF(loaded_data['xyz'], RBFCentres, polyharmonic, w)

    #     print("w error: ", np.amax(loaded_data['w'] - w))
    #     print("RBFCentres error: ", np.amax(loaded_data['RBFCentres'] - RBFCentres))
    #     print("RBFValues error: ", np.amax(loaded_data['RBFValues'] - RBFValues))
    #     print("")