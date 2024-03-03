import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lu_factor, lu_solve

def get_triplets(l):
    """
    Generate all index triplets (i, j, k) such that i + j + k <= l.

    :param l: The maximum degree of the polynomial.
    :return: A list of tuples representing the index triplets.
    """
    triplets = []
    for i in range(l + 1):
        for j in range(l + 1 - i):
            for k in range(l + 1 - (i + j)):
                triplets.append((i, j, k))
    return np.array(triplets)

def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces

def compute_RBF_weights(inputPoints, inputNormals, RBFFunction, epsilon, RBFCentreIndices=[], useOffPoints=True,
                        sparsify=False, l=-1):
    """
    Compute the weights for the RBF interpolation
    :param inputPoints: the input points
    :param inputNormals: the normals at the input points
    :param RBFFunction: the RBF function to use
    :param epsilon: the epsilon parameter
    :param RBFCentreIndices: the indices of the points to use as RBF centres
    :param useOffPoints: whether to use off-surface points
    :param sparsify: whether to sparsify the RBF matrix
    :param l: polynomial degree 

    :return: the weights, the RBF centres, and the polynomial coefficients
    """

    if l > -1 and len(RBFCentreIndices) > 0:
        raise ValueError("Polynomial terms are not supported with centre reduction. Please set l to -1 or use all points.")

    # rbf_centers = inputPoints if no_center_indices else inputPoints[RBFCentreIndices]
    rbf_centers = inputPoints[RBFCentreIndices] if len(RBFCentreIndices) > 0 else inputPoints
    rbf_normals = inputNormals[RBFCentreIndices] if len(RBFCentreIndices) > 0 else inputNormals
    # rbf_normals = inputNormals if no_center_indices else inputNormals[RBFCentreIndices]

    rbf_inputs_len = rbf_centers.shape[0]

    # we always have +- epsilon points for either N or 3N points
    inputPoints = np.vstack((
        inputPoints,
        inputPoints + epsilon * inputNormals,
        inputPoints - epsilon * inputNormals))

    # we only have +- epsilon points depending on the useOffPoints flag and the RBFCentreIndices
    # either M or 3M points where if RBFCentreIndices is not specified, M = N
    if useOffPoints:
        rbf_centers = np.vstack((
            rbf_centers,
            rbf_centers + epsilon * rbf_normals,
            rbf_centers - epsilon * rbf_normals))
        
    # Compute the RBF matrix either 3Mx3N or Mx3N depending on useOffPoints where
    # if RBFCentreIndices is not specified, M = N
    distance_matrix = cdist(rbf_centers, inputPoints, 'euclidean')
    A = RBFFunction(distance_matrix)
    # Initialize b to zeros for all points
    b = np.hstack((np.zeros(rbf_inputs_len), np.repeat(epsilon, rbf_inputs_len), np.repeat(-epsilon, rbf_inputs_len))) if useOffPoints else np.zeros(rbf_inputs_len)

    if l >= 0:
        triplets = get_triplets(l)
        Q = np.prod(rbf_centers[:, None, :] ** triplets[None, :, :], axis=2)
        A = np.block([[A, Q], [Q.T, np.zeros((len(triplets), len(triplets)))]])
        b = np.hstack((b, np.zeros(len(triplets))))
    
    # Do we have an over-determined system?
    if A.shape[0] != A.shape[1]:
        # Overdetermined
        # Solve the least squares system A^T A w = A^T b
        ATA = np.dot(A.T, A)
        ATb = np.dot(A.T, b)
        w, _, _, _ = np.linalg.lstsq(ATA, ATb, rcond=None)
    else:
        # Single solution
        # Solve the linear system using LU decomposition Aw = b => LUw = b
        lu, piv = lu_factor(A)
        w = lu_solve((lu, piv), b)

    if l >= 0:
        a = w[-len(triplets):]
        w = w[:-len(triplets)]
    else:
        a = []
    
    return w, rbf_centers, a

def evaluate_RBF(xyz, centres, RBFFunction, w, l=-1, a=[]):
    """
    Evaluate the RBF at a set of evaluation points.

    :param xyz: The evaluation points.
    :param centres: The RBF centres.
    :param RBFFunction: The chosen RBF function.
    :param w: The computed weights.
    :param l: The degree of polynomial (unused if a is empty).
    :param a: The polynomial coefficients (unused if empty).

    :return: The evaluated implicit function values.
    """
    # Compute the pairwise distance matrix between evaluation points and centers
    distance_matrix = cdist(xyz, centres, 'euclidean')

    # Apply the RBF to the distance matrix
    rbf_values = RBFFunction(distance_matrix)

    # Multiply the RBF values by the weights and sum to get the function value at each evaluation point
    values = np.dot(rbf_values, w)

    # If polynomial terms are used, add their contribution (this part is problem-specific and may vary)
    # Here's a placeholder for polynomial term evaluation (if needed):
    if l >= 0 and len(a) > 0:
        triplets = get_triplets(l)
        poly_terms = np.prod(xyz[:, None, :] ** triplets[None, :, :], axis=2)
        poly_values = np.dot(poly_terms, a)
        values += poly_values

    return values

def biharmonic(r):
    return r

def polyharmonic(r):
    return r**3

def Wendland(r, beta = 0.5):
    return (1 / 12) * (np.maximum(1 - beta*r, 0) ** 3) * (1 - 3*beta*r)