import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lu_factor, lu_solve

from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from scipy.sparse import hstack, vstack, csr_matrix

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

def compute_sparse_A(inputPoints, centres, RBFFunction, epsilon, support_radius):
    """
    Efficiently creates a sparse A matrix using local RBF support.
    :param inputPoints: All input points (including off-surface points).
    :param centres: RBF centres.
    :param RBFFunction: The radial basis function.
    :param epsilon: Small value used for off-surface points.
    :param support_radius: The radius within which the RBF has effect.
    :return: Sparse matrix A.
    """
    tree = cKDTree(centres)
    n_points = inputPoints.shape[0]
    n_centres = centres.shape[0]
    A = lil_matrix((n_points, n_centres), dtype=np.float64)

    for i, point in enumerate(inputPoints):
        # Find indices of centres within the support radius of the point
        indices = tree.query_ball_point(point, support_radius)
        distances = np.linalg.norm(point - centres[indices], axis=1)
        A[i, indices] = RBFFunction(distances)
    
    return A.tocsr()

def compute_sparse_Q(points, l):
    """
    Generate polynomial basis terms up to degree l for a set of points.
    
    :param points: An array of shape (N, 3) containing N points in 3D space.
    :param l: The maximum degree of the polynomial.
    :return: A sparse matrix of shape (N, M) containing polynomial terms,
             where M is the number of terms up to degree l.
    """
    n = points.shape[0]
    triplets = get_triplets(l)
    m = len(triplets)
    Q = lil_matrix((n, m))
    
    for idx, (i, j, k) in enumerate(triplets):
        # Compute the polynomial term x^i * y^j * z^k for each point
        Q[:, idx] = np.prod(np.power(points, [i, j, k]), axis=1)
    
    return Q.tocsr()


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


    inputs_len = inputPoints.shape[0]
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
    if not sparsify:
        A = RBFFunction(cdist(inputPoints, rbf_centers, 'euclidean'))
    else:
        # support_radius = 10 * epsilon
        support_radius = 5
        A = compute_sparse_A(inputPoints, rbf_centers, RBFFunction, epsilon, support_radius)

    # Initialize b to zeros for all points
    # b = np.hstack((np.zeros(inputs_len), np.repeat(epsilon, inputs_len), np.repeat(-epsilon, inputs_len))) if useOffPoints else np.zeros(inputs_len)
    b = np.hstack((np.zeros(inputs_len), np.repeat(epsilon, inputs_len), np.repeat(-epsilon, inputs_len)))
    if l >= 0 and not sparsify:
        # Generate polynomial terms for input points
        triplets = get_triplets(l)
        Q = np.prod(rbf_centers[:, None, :] ** triplets[None, :, :], axis=2)
        # Assemble the full A matrix
        A = np.block([[A, Q], [Q.T, np.zeros((len(triplets), len(triplets)))]])
        b = np.hstack((b, np.zeros(len(triplets))))
    elif l >= 0 and sparsify:
        # Generate polynomial terms for input points
        Q = compute_sparse_Q(inputPoints, l)
        zero_block = csr_matrix((Q.shape[1], Q.shape[1]))
        # Assemble the full A matrix
        A_upper = hstack([A, Q])
        A_lower = hstack([Q.transpose(), zero_block])
        A = vstack([A_upper, A_lower])
        b = np.hstack([b, np.zeros(Q.shape[1])])
    
    # Do we have an over-determined system?
    if sparsify:
        w = lsqr(A, b)[0]
    elif A.shape[0] != A.shape[1]:
        # Overdetermined
        # Solve the least squares system A^T A w = A^T b
        w, _, _, _ = np.linalg.lstsq(A, b)
    else:
        # Single solution
        # Solve the linear system using LU decomposition Aw = b => LUw = b
        lu, piv = lu_factor(A)
        w = lu_solve((lu, piv), b)

    if l >= 0:
        a = w[3*inputs_len:]
        w = w[:3*inputs_len]
        # a = w[-len(triplets):]
        # w = w[:-len(triplets)]
    else:
        a = []
    
    return w, rbf_centers, a

def evaluate_RBF(xyz, centres, RBFFunction, w, l=-1, a=[]):
    """
    Evaluate the RBF at a set of evaluation points, in batches to conserve memory.

    :param xyz: The evaluation points.
    :param centres: The RBF centres.
    :param RBFFunction: The chosen RBF function.
    :param w: The computed weights.
    :param l: The degree of polynomial (unused if a is empty).
    :param a: The polynomial coefficients (unused if empty).
    :param batch_size: The number of points to process in each batch.

    :return: The evaluated implicit function values.
    """
    batch_size = 2000
    values = np.zeros(xyz.shape[0])
    for i in range(0, xyz.shape[0], batch_size):
        end = i + batch_size
        batch_xyz = xyz[i:end]
        
        # Compute the pairwise distance matrix between batch evaluation points and centers
        distance_matrix = cdist(batch_xyz, centres, 'euclidean')

        # Apply the RBF to the distance matrix
        rbf_values = RBFFunction(distance_matrix)

        # Multiply the RBF values by the weights and sum to get the function value at each batch evaluation point
        values_batch = np.dot(rbf_values, w)
        
        # If polynomial terms are used, add their contribution
        if l >= 0 and len(a) > 0:
            triplets = get_triplets(l)
            poly_terms_batch = np.prod(batch_xyz[:, None, :] ** triplets[None, :, :], axis=2)
            poly_values_batch = np.dot(poly_terms_batch, a)
            values_batch += poly_values_batch
        
        # Store the computed values for this batch
        values[i:end] = values_batch

    return values

def biharmonic(r):
    return r

def polyharmonic(r):
    return r**3

def Wendland(r, beta = 0.5):
    return (1 / 12) * (np.maximum(1 - beta*r, 0) ** 3) * (1 - 3*beta*r)