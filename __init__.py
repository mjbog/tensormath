""" Contains tensor math operations, some of which have been implemented in Fortran

print_n_by_n(mat) - reproduce a given matrix in a slightly more readable form
convert_strain(e) - return strain as either 3x3 matrix or 6x1 vector opposite of what is given
convert_stress(s) - return stress as either 3x3 matrix or 6x1 vector opposite of what is given
compute_transformation_matrix(theta) - return 3x3 transformation matrix for given angle in degrees
rotate_stiffness(A,stiffness) - return the rotated 6x6 stiffness tensor after applying rotation matrix A
rotate_stress(A,s) - return the rotated 6x1 stress vector after applying rotation matrix A
rotate_strain(A,e) - return the rotated 6x1 strain vector after applying rotation matrix A
idxchange6_3 - change from 6x1 notation to 3x3 notation
idxchange3_6 - change from 3x3 notation to 6x1 notation
"""
import numpy as np
from numpy import zeros, array, pi, sin, cos
from numba import jit


def print_n_by_n(mat):
    n = len(mat)
    m = len(mat[0])
    print_out = ''
    for i in range(n):
        for j in range(m):
            # try:
            #     print("{0:10.3e}".format(mat[i, j]), end=' ')
            # except:
            print_out += "{0:10.3e}".format(mat[i, j])
        print_out += "\n"
    print(print_out)


def convert_strain(e, return_vector=False, return_matrix=False):
    """
    Convert the vector form of strain to a matrix or vice versa.

    Default behavior is to return the opposite form as the input structure. If a vector is given and return_vector is
    True, the result is unchanged

    :param e: 6x1 contracted notation [11, 22, 33, 12, 13, 23] or 3x3 tensor notation strain
    :param return_vector: Force method to return vector form
    :param return_matrix: Force method to return matrix form
    :return: 6x1 or 3x3  numpy array of strain
    """

    e = array(e)
    if e.size == 6:
        if return_vector:
            new_e = e.copy().reshape(-1, 1)
        else:
            e = e.reshape(-1)
            new_e = _strain_to_matrix(e)
    else:
        if return_matrix:
            new_e = e.copy()
        else:
            new_e = _strain_to_vector(e)
    return new_e


@jit(nopython=True)
def _strain_to_matrix(strain):
    """ Return the 3x3 strain matrix with tensorial shear strain
    
    :param strain: 6x1 contracted notation strain [11, 22, 33, 12, 13, 23] with engineering shear strain
    :type strain: np.ndarray
    :return: 3x3 strain tensor
    :rtype: np.ndarray
    """

    strain = strain.reshape(-1)
    new_strain = zeros((3, 3))
    new_strain[0, 0] = strain[0]
    new_strain[1, 1] = strain[1]
    new_strain[2, 2] = strain[2]
    new_strain[0, 1] = strain[3] * 0.5
    new_strain[0, 2] = strain[4] * 0.5
    new_strain[1, 2] = strain[5] * 0.5
    new_strain[1, 0] = strain[3] * 0.5
    new_strain[2, 0] = strain[4] * 0.5
    new_strain[2, 1] = strain[5] * 0.5
    return new_strain


@jit(nopython=True)
def _strain_to_vector(strain):
    """ Return the 6x1 strain vector with engineering shear strain [11, 22, 33, 12, 13, 23] indexing
    
    :param strain: 3x3 tensorial strain
    :type strain: np.ndarray
    :return: 6x1 contracted notation strain [11, 22, 33, 12, 13, 23] with engineering shear strain
    :rtype: np.ndarray
    """

    new_strain = zeros((6, 1))
    new_strain[0, 0] = strain[0, 0]
    new_strain[1, 0] = strain[1, 1]
    new_strain[2, 0] = strain[2, 2]
    new_strain[3, 0] = strain[0, 1] * 2.
    new_strain[4, 0] = strain[0, 2] * 2.
    new_strain[5, 0] = strain[1, 2] * 2.
    return new_strain


def convert_stress(s, return_vector=False, return_matrix=False):
    """ Convert the vector form of strain to a matrix or vice convert_strain """
    s = array(s)
    if s.size == 6:
        if return_vector:
            new_s = s.copy().reshape(-1)
        else:
            new_s = _stress_to_matrix(s)
    else:
        if return_matrix:
            new_s = s.copy()
        else:
            new_s = _stress_to_vector(s)
    return new_s


@jit(nopython=True)
def _stress_to_matrix(stress):
    """ Return the 3x3 stress tensor
    
    :param stress: 6x1 contracted notation stress [11, 22, 33, 12, 13, 23]
    :type stress: np.ndarray
    :return: 3x3 stress tensor
    :rtype: np.ndarray
    """

    stress = stress.reshape(-1)
    new_stress = zeros((3, 3))
    new_stress[0, 0] = stress[0]
    new_stress[1, 1] = stress[1]
    new_stress[2, 2] = stress[2]
    new_stress[0, 1] = stress[3]
    new_stress[0, 2] = stress[4]
    new_stress[1, 2] = stress[5]
    new_stress[1, 0] = stress[3]
    new_stress[2, 0] = stress[4]
    new_stress[2, 1] = stress[5]
    return new_stress


@jit(nopython=True)
def _stress_to_vector(stress):
    """ Return the 6x1 stress vector
    
    :param stress: 3x3 tensorial stress
    :type stress: np.ndarray
    :return: 6x1 contracted notation stress [11, 22, 33, 12, 13, 23]
    :rtype: np.ndarray
    """

    new_stress = zeros((6, 1))
    new_stress[0] = stress[0, 0]
    new_stress[1] = stress[1, 1]
    new_stress[2] = stress[2, 2]
    new_stress[3] = stress[0, 1]
    new_stress[4] = stress[0, 2]
    new_stress[5] = stress[1, 2]
    return new_stress


# Wrappers for Fortran functions
@jit(nopython=True)
def compute_transformation_matrix(theta):
    """
    Return the 3x3 transformation matrix for given in plane rotation angle

    :param theta: rotation in degrees from the X axis in the XY plane about the Z axis
    :return: 3x3 transformation matrix
    """
    theta *= pi / 180.

    cos_t = cos(theta)
    sin_t = sin(theta)

    return array([[cos_t, sin_t, 0.],
                  [-sin_t, cos_t, 0.],
                  [0., 0., 1.]])


def rotate_stiffness(transformation_matrix, stiffness_tensor):
    """ Return the rotated 6x6 stiffness tensor for the given 3x3 transformation matrix and initial 6x6 stiffness tensor

    :param transformation_matrix: 3x3 transformation matrix
    :param stiffness_tensor: 6x6 initial stiffness tensor
    :return: 6x6 rotated stiffness tensor
    """
    return _rotate_fourth_order(transformation_matrix, stiffness_tensor)


def rotate_stress(transformation_matrix, stress):
    """
    Return the rotated stress vector after applying given transformation
    :param transformation_matrix: 3x3 transformation matrix
    :param stress: 6x1 stress vector in [11, 22, 33, 12, 13, 23] notation or 3x3 stress tensor
    :return: 6x1 rotated stress vector or 3x3 rotated stress tensor matching input type
    """

    # Check the format of the input stress
    if stress.size == 6:
        # Convert to matrix
        stress_mat = convert_stress(stress)
        # Rotate
        stress_rot = _rotate_second_order(transformation_matrix, stress_mat)
        # Return in vector form
        return convert_stress(stress_rot)
    else:
        return _rotate_second_order(transformation_matrix, stress)


def rotate_strain(transformation_matrix, strain):
    """
    Return the rotated strain vector after applying given transformation
    :param transformation_matrix: 3x3 transformation matrix
    :param strain: 6x1 strain vector in [11, 22, 33, 12, 13, 23] notation or 3x3 strain tensor
    :return: 6x1 rotated strain vector or 3x3 rotated strain tensor matching input type
    """
    # Check the format of the input strain
    if strain.size == 6:
        # Convert to matrix
        strain_mat = convert_strain(strain)
        # Rotate
        strain_rot = _rotate_second_order(transformation_matrix, strain_mat)
        # Return in vector form
        return convert_strain(strain_rot)
    else:
        return _rotate_second_order(transformation_matrix, strain)


def rotate_thermal_expansion(transformation_matrix, thermal_expansion):
    """
    Return the rotated coefficient of thermal expansion matrix after applying given transformation
    :param transformation_matrix: 3x3 transformation matrix
    :param thermal_expansion: 3x3 initial thermal_expansion matrix
    :return: 3x3 rotated thermal_expansion vector
    """
    return _rotate_second_order(transformation_matrix, thermal_expansion)


@jit(nopython=True)
def _rotate_second_order(a, mat2):
    """
    Return the rotated second order tensor (stress, strain, CTE, etc.)

    :param a: 3x3 rotation matrix
    :param mat2: 3x3 matrix in original axes
    :return: 3x3 rotated matrix
    """
    mat2_out = zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for m in range(3):
                for n in range(3):
                    mat2_out[m, n] += a[i, m] * a[j, n] * mat2[i, j]
    return mat2_out


@jit(nopython=True)
def _rotate_fourth_order(a, mat4):
    """
    Return the rotated fourth order tensor represented in 6x6 contracted notation [11, 22, 33, 12, 13, 23]
    
    :param a: 3x3 rotation matrix
    :param mat4: 6x6 contracted notation tensor
    :return: 6x6 rotated tensor
    """

    expand = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

    contract = [[0, 3, 4],
                [3, 1, 5],
                [4, 5, 2]]

    mat_out = zeros((6, 6))

    for ij in range(6):
        i, j = expand[ij]
        for kl in range(ij, 6):
            k, l = expand[kl]
            value = 0.
            for m in range(3):
                for n in range(3):
                    mn = contract[m][n]
                    for o in range(3):
                        for p in range(3):
                            op = contract[o][p]
                            value += mat4[mn, op] * a[m, i] * a[n, j] * a[o, k] * a[p, l]
            mat_out[ij, kl] = value
            mat_out[kl, ij] = value

    return mat_out


@jit(nopython=True)
def idxchange6_3(ij):
    """
    Return the indices for the 3x3 representation and accompanying multiplier of the given 6x1 index
    :param ij: int index from 6x1 representation
    :return: int first index of 3x3 matrix
             int second index of 3x3 matrix
             float multiplier for shear terms
    """

    indexing = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    multipliers = [1., 1., 1., 2., 2., 2.]

    i, j = indexing[ij]
    multiplier = multipliers[ij]

    return i, j, multiplier


idxchange63 = idxchange6_3


@jit(nopython=True)
def idxchange3_6(i, j):
    """
    Return the index for the 6x1 representation and accompanying multiplier for a given 3x3 index pair
    :param i: int -- first index
    :param j: int -- second index
    :return: int -- reduced index
             float -- multiplier for shear terms
    """

    indexing = [[0, 3, 4],
                [3, 1, 5],
                [4, 5, 2]]
    multipliers = [[1., 2., 2.],
                   [2., 1., 2.],
                   [2., 2., 1.]]

    ij = indexing[i][j]
    multiplier = multipliers[i][j]

    return ij, multiplier


idxchange36 = idxchange3_6
