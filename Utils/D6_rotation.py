'''Reference code from paper'''
''' Implementation of https://arxiv.org/pdf/1812.07035.pdf '''

from cmath import pi
import numpy as np
import math

def expand(v):
    if len(v.shape) == 1:
        v = v[np.newaxis, :]
    return v

def repmat_norm(v, r=None):
    ''' calculate the 2-norm of v and repeat it by r  '''
    if r is None:
        v = expand(v)
        r = v.shape[1]
    norm = np.expand_dims(np.sqrt(np.sum(v ** 2, axis=1)), axis=1)
    rep_norm = np.tile(norm, (1, r))
    if len(rep_norm.shape) == 1:
        rep_norm = rep_norm[np.newaxis, :]
    return rep_norm


def quat_to_so3(q):
    ''' converts quaternion q [qw,qx,qy,qz] to SO3 representation [a0.T,a1.T,a2.T] '''
    if len(q.shape) == 2:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        w, x, y, z = q[np.newaxis, 0], q[np.newaxis, 1], q[np.newaxis, 2], q[np.newaxis, 3]

    # a_row_column
    a00 = 1 - 2 * y ** 2 - 2 * z ** 2
    a10 = 2 * x * y + 2 * z * w
    a20 = 2 * x * z - 2 * y * w
    a01 = 2 * x * y - 2 * z * w
    a11 = 1 - 2 * x ** 2 - 2 * z ** 2
    a21 = 2 * y * z + 2 * x * w
    a02 = 2 * x * z + 2 * y * w
    a12 = 2 * y * z - 2 * x * w
    a22 = 1 - 2 * x ** 2 - 2 * y ** 2

    R = np.vstack([a00, a10, a20, a01, a11, a21, a02, a12, a22]).T
    return expand(R)


def quat_to_d6(q):
    ''' converts quaternion q [qw,qx,qy,qz] to 6D representation [a0.T,a1.T] '''
    return quat_to_so3(q)[:, :6]


def _so3_to_quat(R):
    ''' converts single SO3 rotation matrix R [a0.T,a1.T,a2.T] to quaternion q [qw,qx,qy,qz] '''
    R = expand(R)

    a00, a10, a20 = R[:, 0], R[:, 1], R[:, 2]
    a01, a11, a21 = R[:, 3], R[:, 4], R[:, 5]
    a02, a12, a22 = R[:, 6], R[:, 7], R[:, 8]

    # convert
    tr = a00 + a11 + a22

    if (tr > 0.):
        S = np.sqrt(tr + 1.0) * 2.  # S=4*qw
        w = 0.25 * S
        x = (a21 - a12) / S
        y = (a02 - a20) / S
        z = (a10 - a01) / S
    elif (a00 > a11) and (a00 > a22):
        S = np.sqrt(1.0 + a00 - a11 - a22) * 2.  # S=4*qx
        w = (a21 - a12) / S
        x = 0.25 * S
        y = (a01 + a10) / S
        z = (a02 + a20) / S
    elif (a11 > a22):
        S = np.sqrt(1.0 + a11 - a00 - a22) * 2.  # S=4*qy
        w = (a02 - a20) / S
        x = (a01 + a10) / S
        y = 0.25 * S
        z = (a12 + a21) / S
    else:
        S = np.sqrt(1.0 + a22 - a00 - a11) * 2.  # S=4*qz
        w = (a10 - a01) / S
        x = (a02 + a20) / S
        y = (a12 + a21) / S
        z = 0.25 * S
    return np.hstack([w, x, y, z])

def so3_to_quat(R):
    ''' converts SO3 rotation matrix R [a0.T,a1.T,a2.T] to quaternion q [qw,qx,qy,qz] '''
    R = expand(R)
    q = np.zeros(shape=(R.shape[0], 4))

    # convert
    for row in range(R.shape[0]):
        q[row, :] = _so3_to_quat(R[row, :])

    # normalize
    q /= repmat_norm(q)

    return q


def d6_to_so3(R):
    ''' converts 6D representation [a1.T,a2.T] to rotation matrix R [a0.T,a1.T,a2.T] '''
    if len(R.shape) == 1:
        R = R[np.newaxis, :]

    a00, a10, a20 = R[:, 0], R[:, 1], R[:, 2]
    a01, a11, a21 = R[:, 3], R[:, 4], R[:, 5]

    a0 = np.vstack([a00, a10, a20]).T
    a1 = np.vstack([a01, a11, a21]).T

    b0 = a0 / repmat_norm(a0)
    b1 = np.zeros(shape=b0.shape)
    b2 = np.zeros(shape=b0.shape)
    for row in range(b0.shape[0]):
        b1[row,:] = a1[row,:] - (np.dot(b0[row,:], a1[row,:])) * b0[row,:]
        b2[row,:] = np.cross(b0[row,:], b1[row,:])


    return np.concatenate([b0, b1, b2], axis=1)


def d6_to_quat(R):
    return so3_to_quat(d6_to_so3(R))

def _so3_to_axis_angle(R):
    ''' Convert the rotation matrix into the axis-angle notation '''
    R = expand(R)
    a00, a10, a20 = R[:, 0][0], R[:, 1][0], R[:, 2][0]
    a01, a11, a21 = R[:, 3][0], R[:, 4][0], R[:, 5][0]
    a02, a12, a22 = R[:, 6][0], R[:, 7][0], R[:, 8][0]

    matrix = np.matrix([
    [a00, a01, a02],
    [a10, a11, a12],
    [a20, a21, a22]
    ])

    # Axes.
    axis = np.zeros(3)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = math.atan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return np.hstack([axis, theta])

def so3_to_axis_angle(R):
    ''' Convert the rotation matrix into the axis-angle notation '''
    aa = np.zeros(shape=(R.shape[0],4))
    for row in range(R.shape[0]):
        aa[row, :] = _so3_to_axis_angle(R[row,:])
    return aa

def rpy_to_so3(roll, pitch, yaw):
    ''' converts RPY to rotation matrix R [a0.T,a1.T,a2.T] '''
    yaw_matrix = np.matrix([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 1]
    ])

    pitch_matrix = np.matrix([
    [np.cos(pitch), 0, np.sin(pitch)],
    [0, 1, 0],
    [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    roll_matrix = np.matrix([
    [1, 0, 0],
    [0, np.cos(roll), -np.sin(roll)],
    [0, np.sin(roll), np.cos(roll)]
    ])

    R = yaw_matrix * pitch_matrix * roll_matrix
    return np.hstack([R[:,0].T, R[:,1].T, R[:,2].T])

def rpy_to_6d(roll,pitch,yaw):
    return quat_to_d6(so3_to_quat(rpy_to_so3(roll,pitch,yaw)))

def gaze_to_d6(face_gaze):
    #face_np = face_tensor.cpu().detach().numpy()
    roll = 0

    pitch = face_gaze[0] #* np.pi /2
    yaw = face_gaze[1] #* np.pi /2

    #pitch = face_gaze[0] * np.pi / 2
    #yaw = face_gaze[1] * np.pi / 2

    return rpy_to_6d(roll,pitch,yaw).flatten()




if __name__ == '__main__':
    value = [[0,np.pi/6,np.pi/2],
            [0,np.pi/3,np.pi/2],
            [0,np.pi/2,np.pi/2],
            [0,2*np.pi/3,np.pi/2]]

    for rpy in value:
        d6 = rpy_to_6d(*rpy)
        print(d6.flatten())