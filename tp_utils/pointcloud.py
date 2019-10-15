import numpy as np
import quaternion
from pyntcloud import PyntCloud
#  from pyntcloud.plot.pythreejs_backend import pythreejs
import pyntcloud
import pandas as pd
from scipy.spatial.transform import Rotation
import io
import base64
import open3d as o3
import pythreejs
import time
import sys
import trimesh
import copy
from functools import lru_cache
from IPython.display import display, clear_output
import json
from tqdm import tqdm_notebook
import os
from PIL import Image
from tensorflow.python.util import nest
from scipy.spatial import Delaunay
import ipywidgets
#  from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
from contextlib import contextmanager


# From kitti_util.py
def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


# From kitti_util.py
class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath, from_video=False, from_sun=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        elif from_sun:
            calibs = self.read_calib_from_sun(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                # KITTI tracking quickfix
                key, value = line.split(' ', 1)
                key = key.replace(':', '')
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        # KITTI tracking quickfix
        data['Tr_velo_to_cam'] = data['Tr_velo_cam']
        data['R0_rect'] = data['R_rect']
        return data

    def read_calib_from_sun(self, sun_image):
        P = np.array([0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((3, 4))
        P[:3, :3] = sun_image.K
        Tr_velo_to_cam = np.eye(4)[:3, :]
        R0_rect = np.eye(3)
        return {'P2': P, 'Tr_velo_to_cam': Tr_velo_to_cam, 'R0_rect': R0_rect}

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def np_to_str(arr, plaintext=True):
    output = io.BytesIO()
    if plaintext:
        np.savetxt(output, arr)
        s = output.getvalue()
        s = s.decode('ascii')
    else:
        np.savez_compressed(output, arr=arr)
        s = output.getvalue()
        s = base64.b64encode(s).decode('ascii')
    return s


def str_to_np(s, plaintext=True):
    if plaintext:
        return np.loadtxt(io.BytesIO(s.encode('ascii')))
    else:
        s = base64.b64decode(s)
        return np.load(io.BytesIO(s))['arr']


def get_mat(translation=None, rotation=None):
    assert False
    mat = np.eye(4)
    if translation is not None:
        mat[:3, 3] = translation
    if rotation is not None:
        assert False  # Does not work?
        mat[:3, :3] = quaternion.as_rotation_matrix(np.quaternion(*rotation))
    return mat


def get_mat_angle(translation=None, rotation=None, rotation_center=np.array([0., 0, 0])):
    mat1 = np.eye(4)
    mat2 = np.eye(4)
    mat3 = np.eye(4)
    mat1[:3, 3] = -rotation_center
    mat3[:3, 3] = rotation_center
    if translation is not None:
        mat3[:3, 3] += translation
    if rotation is not None:
        mat2[:3, :3] = Rotation.from_rotvec(np.array([0, 0, 1.]) * rotation).as_dcm()
    return np.matmul(np.matmul(mat3, mat2), mat1)


def transform_points(ps, mats):
    if type(mats) == list:
        for mat in mats:
            ps[:, :4] = np.matmul(ps[:, :4], np.transpose(mat))
    else:
        ps = np.matmul(ps[:, :4], np.transpose(mats))
    return ps


def heuristic_use_smaller_angle(pred_angles):
    pred_angles = np.mod(pred_angles, 2. * np.pi)
    large_pred_angle_mask = np.logical_and(pred_angles > 0.5 * np.pi, pred_angles < 1.5 * np.pi)
    pred_angles[large_pred_angle_mask] = np.mod(pred_angles[large_pred_angle_mask] + np.pi, 2. * np.pi)
    pred_angles = np.mod(pred_angles + np.pi, 2. * np.pi) - np.pi
    return pred_angles


def translate_transform_to_new_center_of_rotation(all_pred_translations, all_pred_angles, all_pred_centers, all_gt_pc1centers):
    new_all_pred_translations = np.zeros_like(all_pred_translations)
    for idx, (pred_translation, pred_angle, pred_center, gt_pc1center) in enumerate(zip(all_pred_translations, all_pred_angles, all_pred_centers, all_gt_pc1centers)):
        old_center, new_center = pred_center, gt_pc1center
        center_shift = new_center - old_center
        #  pred_transform = get_mat_angle(pred_translation, pred_angle, rotation_center=pred_center)
        #  new_pred_translation = pred_translation
        new_pred_translation = -center_shift + (np.matmul(get_mat_angle(rotation=pred_angle)[:3, :3], center_shift)) + pred_translation
        new_all_pred_translations[idx] = new_pred_translation
    return new_all_pred_translations


origin_lines = [{"color": "red", "vertices": [[0, 0, 0], [1.0, 0, 0]]}, {"color": "green", "vertices": [[0, 0, 0], [0, 1.0, 0]]}, {"color": "blue", "vertices": [[0, 0, 0], [0, 0, 1.0]]}]

bbox_pairs = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


def get_points_sun(pc, mask=None):
    m = np.empty((pc.shape[0], 6), dtype=np.float32)
    m[:, :3] = pc[:, :3]
    m[:, 3:6] = pc[:, 3:6] * 255.
    return pd.DataFrame(m, columns=['x', 'y', 'z', 'red', 'green', 'blue'])


def clean_polylines(polylines):
    for line in polylines:
        line['vertices'] = [list(l) for l in line['vertices']]
    return polylines


def get_center_dot(center, c, length=1.0):
    return [{"color": c, "vertices": [center, center + np.random.uniform(-0.5, 0.5, 3) * length]} for i in range(20)]


def get_bbox_lines(box_points, c):
    polylines = []
    for pair in bbox_pairs:
        p1, p2 = box_points[[pair]]
        polylines.append({"color": c, "vertices": [list(p1), list(p2)]})
    return polylines


def get_points(pc, color=[255, 0, 0], show_point_color=False):
    m = np.empty((pc.shape[0], 6), dtype=np.float32)
    if pc.shape[1] == 3:
        m[:, :3] = pc
    elif pc.shape[1] == 4:
        m[:, :3] = pc[:, :3] / pc[:, 3:]
    elif pc.shape[1] == 6:
        m[:, :] = pc[:, :]
    elif pc.shape[1] == 7:
        m[:, :3] = pc[:, :3] / pc[:, 3:4]
        m[:, 3:] = pc[:, 4:]
    else:
        assert False
    if pc.shape[1] < 6 or not show_point_color:
        m[:, 3:] = color
    return pd.DataFrame(m, columns=['x', 'y', 'z', 'red', 'green', 'blue'])


def get_arrow(scene, predicted_transform):
    origin = scene.transform.transform_start[:3, 3]
    target = scene.transform.transform_end[:3, 3]
    #  end_point = origin + predicted_transform[:3]
    #  print(scene.transform.rel_transform)
    reltr = np.matmul(scene.transform.rel_transform, np.array([0, 0, 0, 1]))
    #  print(reltr)
    reltr_3 = reltr[:3] / reltr[3]
    #  print(reltr_3)
    end_point = origin + reltr_3
    end_point = np.matmul(scene.transform.rel_transform, np.array([*origin, 1]))
    #  end_point2 = np.matmul(predicted_transform, np.array([*origin,1]))
    end_point2 = predicted_transform[:3]
    end_point = scene.transform.transform_end[:3, 3]
    end_point = origin + scene.transform.rel_transform[:3, 3]
    #  end_point = np.matmul(scene.transform.transform_end, np.array([*origin,1]))
    lines = [{"color": "blue", "vertices": [origin, end_point2]}]
    return lines


def get_arrowSimple(scene, translation, color='blue'):
    origin = scene.transform.transform_start[:3, 3]
    lines = [{'color': color, 'vertices': [origin, origin + translation]}]
    return lines


def clean_color(color):
    if color == 'red':
        color = [255, 0, 0]
    elif color == 'green':
        color = [0, 255, 0]
    elif color == 'blue':
        color = [0, 0, 255]
    elif color == 'yellow':
        color = [255, 255, 0]
    elif color == 'pink':
        color = [255, 0, 255]
    elif color == 'cyan':
        color = [0, 255, 255]
    elif color == 'white':
        color = [255, 255, 255]
    return color


####### Dataset generation


# https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
# http://planning.cs.uiuc.edu/node198.html
def rand_quat():
    u = np.random.uniform(0, 1, 3)
    h1 = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
    h2 = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
    h3 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
    h4 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])

    return np.quaternion(h1, h2, h3, h4)


def rand_angle():
    return np.random.uniform(-np.pi, np.pi)


def rand_quat_planar():
    angle = np.random.uniform(-np.pi, np.pi)
    h1 = 1
    h2 = 0
    h3 = 0
    h4 = angle

    return np.quaternion(h1, h2, h3, h4)


def angle_diff(a, b):
    d = a - b
    return (d + np.pi) % (np.pi * 2.0) - np.pi


class Mesh:
    def __init__(self, fname):
        self.mesh = trimesh.load_mesh(fname)
        median = np.mean(self.mesh.bounds[:, :], axis=0)
        self.mesh.apply_translation(-median)
        max_length = np.max(np.abs(self.mesh.bounds[:, :]))
        length = 1.0 / (max_length * 2.0)
        self.mesh.apply_scale(length)

    def apply_transform(self, transform):
        self.mesh.apply_transform(transform)
        return self.mesh

    def apply_scale(self, scale):
        self.mesh.apply_scale(scale)
        return self.mesh

    def clone(self):
        return copy.deepcopy(self)


@lru_cache(maxsize=128)
def get_mesh(fname):
    return Mesh(fname)


def show_mesh(mesh_id, cat='car', aligned=True):
    print(mesh_id)
    mesh_fname = f'/globalwork/gross/ModelNet/ModelNet40{"Aligned" if aligned else ""}/{cat}/train/{cat}_{str(mesh_id).zfill(4)}.off'
    if not os.path.isfile(mesh_fname):
        mesh_fname = f'/globalwork/gross/ModelNet/ModelNet40{"Aligned" if aligned else ""}/{cat}/test/{cat}_{str(mesh_id).zfill(4)}.off'
    mesh = get_mesh(mesh_fname)
    mesh.apply_scale(3.)

    def on_button_next(b):
        clear_output()
        show_mesh(mesh_id + 1, cat=cat, aligned=aligned)

    def on_button_prev(b):
        clear_output()
        show_mesh(max(mesh_id - 1, 1), cat=cat, aligned=aligned)

    button_next = ipywidgets.Button(description="Next")
    button_prev = ipywidgets.Button(description="Prev")
    display(ipywidgets.HBox([button_prev, button_next]))
    button_next.on_click(on_button_next)
    button_prev.on_click(on_button_prev)
    scene = trimesh.Scene(mesh.mesh)
    viewer = scene.show(viewer='notebook')
    display(viewer)
    return scene, viewer


class RandomTransform:
    #  def __init__(self):
    #      self.angle = np.random.uniform(-np.pi, np.pi)
    #      self.velocity = np.random.uniform(0, 10)
    #      self.translation = np.array([np.sin(self.angle), np.cos(self.angle), 0]) * self.velocity
    #      self.q = np.quaternion(1,0,0,0)

    #      polar_angle = np.random.uniform(-np.pi, np.pi)
    #      polar_distance = np.random.uniform(4, 20)
    #      self.start_position = np.array([np.sin(polar_angle), np.cos(polar_angle), 0]) * polar_distance

    #      self.transform_start = np.eye(4)
    #      self.transform_start[:3,3] = self.start_position

    #      self.rel_transform = np.eye(4)
    #      self.rel_transform[:3,3] = self.translation

    #      self.transform_end = np.eye(4)
    #      self.transform_end[:3,3] = self.start_position + self.translation

    def __init__(self, polar_dist_range):
        #  self.angle = np.random.uniform(-np.pi, np.pi)
        self.angle = np.random.uniform(-np.pi, np.pi)
        self.velocity = np.random.uniform(0, 1)
        self.translation = np.array([np.sin(self.angle), np.cos(self.angle), 0]) * self.velocity
        #  # self.q = rand_quat()
        #  self.q = rand_quat_planar()
        self.rel_angle = rand_angle() / 2.0
        self.q = quaternion.from_rotation_vector(np.array([0, 0, 1.]) * self.rel_angle)

        polar_angle = np.random.uniform(-np.pi, np.pi)
        polar_distance = np.random.uniform(*polar_dist_range)
        self.start_position = np.array([np.sin(polar_angle), np.cos(polar_angle), 0]) * polar_distance
        #  self.q_start = rand_quat_planar()
        self.start_angle = rand_angle()
        self.q_start = quaternion.from_rotation_vector(np.array([0, 0, 1.]) * self.rel_angle)

        self.end_position = self.start_position + self.translation
        self.end_angle = self.start_angle + self.rel_angle
        #  self.q_end = self.q_start * self.q
        self.q_end = quaternion.from_rotation_vector(np.array([0, 0, 1.]) * self.end_angle)

        #  self.transform_start = np.eye(4)
        #  self.transform_start[:3,:3] = quaternion.as_rotation_matrix(self.q_start)
        #  self.transform_start[:3,3] = self.start_position
        self.transform_start = get_mat_angle(self.start_position, self.start_angle)

        #  self.rel_transform = np.eye(4)
        #  self.rel_transform[:3,:3] = quaternion.as_rotation_matrix(self.q)
        #  self.rel_transform[:3,3] = self.translation
        self.rel_transform = get_mat_angle(self.translation, self.rel_angle)

        #  self.transform_end = np.eye(4)
        #  self.transform_end[:3,3] = self.end_position
        #  self.transform_end[:3,:3] = quaternion.as_rotation_matrix(self.q_end)
        #  self.transform_end = np.matmul(self.rel_transform, self.transform_start)
        self.transform_end = get_mat_angle(self.end_position, self.end_angle)

    def __repr__(self):
        return f'{self.translation} {self.rel_angle}'


def to_array_list(df, length=None, by_id=True):
    """Converts a dataframe to a list of arrays, with one array for every unique index entry.
    Index is assumed to be 0-based contiguous. If there is a missing index entry, an empty
    numpy array is returned for it.
    Elements in the arrays are sorted by their id.
    :param df:
    :param length:
    :return:
    """

    if by_id:
        assert 'id' in df.columns

        # if `id` is the only column, don't sort it (and don't remove it)
        if len(df.columns) == 1:
            by_id = False

    idx = df.index.unique()
    if length is None:
        length = max(idx) + 1

    ll = [np.empty(0) for _ in range(length)]
    for i in idx:
        a = df.loc[i]
        if by_id:
            if isinstance(a, pd.Series):
                a = a[1:]
            else:
                a = a.copy().set_index('id').sort_index()

        ll[i] = a.values.reshape((-1, a.shape[-1]))
    return np.asarray(ll)


# From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/tracking.py
class KittiTrackingLabels(object):
    """Kitt Tracking Label parser. It can limit the maximum number of objects per track,
    filter out objects with class "DontCare", or retain only those objects present
    in a given frame.
    """

    columns = 'id class truncated occluded alpha x1 y1 x2 y2 xd yd zd x y z roty'.split()
    classes = 'Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc DontCare'.split()

    def __init__(self, path_or_df, bbox_with_size=True, remove_dontcare=True, split_on_reappear=True, truncated_threshold=2., occluded_threshold=3.):

        if isinstance(path_or_df, pd.DataFrame):
            self._df = path_or_df
        else:
            if not os.path.exists(path_or_df):
                raise ValueError('File {} doesn\'t exist'.format(path_or_df))

            self._df = pd.read_csv(path_or_df, sep=' ', header=None, names=self.columns, index_col=0, skip_blank_lines=True)

        self.bbox_with_size = bbox_with_size

        if remove_dontcare:
            self._df = self._df[self._df['class'] != 'DontCare']

        for c in self._df.columns:
            self._convert_type(c, np.float32, np.float64)
            self._convert_type(c, np.int32, np.int64)

        if not nest.is_sequence(occluded_threshold):
            occluded_threshold = (0, occluded_threshold)

        if not nest.is_sequence(truncated_threshold):
            truncated_threshold = (0, truncated_threshold)

        self._df = self._df[self._df['occluded'] >= occluded_threshold[0]]
        self._df = self._df[self._df['occluded'] <= occluded_threshold[1]]

        self._df = self._df[self._df['truncated'] >= truncated_threshold[0]]
        self._df = self._df[self._df['truncated'] <= truncated_threshold[1]]

        # make 0-based contiguous ids
        ids = self._df.id.unique()
        offset = max(ids) + 1
        id_map = {id: new_id for id, new_id in zip(ids, np.arange(offset, len(ids) + offset))}
        self._df.replace({'id': id_map}, inplace=True)
        self._df.id -= offset

        self.ids = list(self._df.id.unique())
        self.max_objects = len(self.ids)
        self.index = self._df.index.unique()

        if split_on_reappear:
            added_ids = self._split_on_reappear(self._df, self.presence, self.ids[-1])
            self.ids.extend(added_ids)
            self.max_objects += len(added_ids)

    def _convert_type(self, column, dest_type, only_from_type=None):
        cond = only_from_type is None or self._df[column].dtype == only_from_type
        if cond:
            self._df[column] = self._df[column].astype(dest_type)

    @property
    def bbox(self):
        bbox = self._df[['id', 'y1', 'x1', 'y2', 'x2']].copy()
        if self.bbox_with_size:
            bbox['y2'] -= bbox['y1']
            bbox['x2'] -= bbox['x1']
        """Converts a dataframe to a list of arrays
        :param df:
        :param length:
        :return:
        """

        return to_array_list(bbox)

    @property
    def presence(self):
        return self._presence(self._df, self.index, self.max_objects)

    @property
    def num_objects(self):
        ns = self._df.id.groupby(self._df.index).count()
        absent = list(set(range(len(self))) - set(self.index))
        other = pd.DataFrame([0] * len(absent), absent)
        ns = ns.append(other)
        ns.sort_index(inplace=True)
        return ns.as_matrix().squeeze()

    @property
    def cls(self):
        return to_array_list(self._df[['id', 'class']])

    @property
    def occlusion(self):
        return to_array_list(self._df[['id', 'occluded']])

    @property
    def id(self):
        return to_array_list(self._df['id'])

    def __len__(self):
        return self.index[-1] - self.index[0] + 1

    @classmethod
    def _presence(cls, df, index, n_objects):
        p = np.zeros((index[-1] + 1, n_objects), dtype=bool)
        for i, row in df.iterrows():
            p[i, row.id] = True
        return p

    @classmethod
    def _split_on_reappear(cls, df, p, id_offset):
        """Assign a new identity to an objects that appears after disappearing previously.
        Works on `df` in-place.
        :param df: data frame
        :param p: presence
        :param id_offset: offset added to new ids
        :return:
        """

        next_id = id_offset + 1
        added_ids = []
        nt = p.sum(0)
        start = np.argmax(p, 0)
        end = np.argmax(np.cumsum(p, 0), 0)
        diff = end - start + 1
        is_contiguous = np.equal(nt, diff)
        for id, contiguous in enumerate(is_contiguous):
            if not contiguous:

                to_change = df[df.id == id]
                index = to_change.index
                diff = index[1:] - index[:-1]
                where = np.where(np.greater(diff, 1))[0]
                for w in where:
                    to_change.loc[w + 1:, 'id'] = next_id
                    added_ids.append(next_id)
                    next_id += 1

                df[df.id == id] = to_change

        return added_ids


def load_kitti_velo_scan(filename):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(filename, dtype=np.float32)
    return scan.reshape((-1, 4))


basepath_kitti = '/globalwork/data/KITTI_tracking'


def load_kitti_velo_scan_frame(seq, frame, use_vo=True):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(f'{basepath_kitti}/training/velodyne/{str(seq).zfill(4)}/{str(frame).zfill(6)}.bin', dtype=np.float32)
    scan = scan.reshape((-1, 4))
    if use_vo:
        vo_mat = np.loadtxt(f'{basepath_kitti}/preprocessed/training/visual_odometry/vo_{str(seq).zfill(4)}_{str(frame).zfill(6)}.txt', dtype=np.float32)
        R1 = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
        R2 = np.array([[0., -1., 0], [1, 0, 0], [0, 0, 1]])
        R = np.eye(4)
        R[:3, :3] = np.matmul(R1, R2)
        vo_mat = np.matmul(np.transpose(R), np.matmul(vo_mat, R))
        scan_4 = np.concatenate([scan[:, :3], np.ones((scan.shape[0], 1))], axis=1)
        scan_4_transformed = np.matmul(scan_4, vo_mat.T)
        return scan_4_transformed[:, :3] / scan_4_transformed[:, 3:]
    else:
        return scan[:, :3]


# From fruistum pointnets/prepare_data.py
def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
        (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def extract_pc_in_box2d(pc, pc_velo, box2d, calib, img_width, img_height):
    ''' pc: (N,3), box2d: (xmin,ymin,xmax,ymax) '''
    _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)
    xmin, ymin, xmax, ymax = box2d
    box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & (pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)
    box_fov_inds = box_fov_inds & img_fov_inds
    pc_in_box_fov = pc[box_fov_inds, :]
    return pc_in_box_fov


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    #  print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def extract_color_from_pc(tracklet_pc, calib, R, im):
    im = np.array(im).astype(np.float32)
    colors = np.zeros((tracklet_pc.shape[0], 3))
    pts_3d = np.matmul(tracklet_pc[:, :3], R.T)
    pts_2d = project_to_image(pts_3d, calib.P).astype(np.int32)
    num = 0
    for idx, (x, y) in enumerate(pts_2d):
        if 0 <= x < im.shape[1] and 0 <= y < im.shape[0]:
            num += 1
            colors[idx] = im[y, x]
    return colors


def extract_pointcloud(tracklet, from_bbox2d, calib):
    seq = int(tracklet[0])
    frame = int(tracklet[1])
    pc_velo = np.fromfile(f'{basepath_kitti}/training/velodyne/{str(seq).zfill(4)}/{str(frame).zfill(6)}.bin', dtype=np.float32).reshape((-1, 4))
    R1 = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    R2 = np.array([[0., -1., 0], [1, 0, 0], [0, 0, 1]])
    R = np.matmul(R1, R2)

    vo_mat = np.loadtxt(f'{basepath_kitti}/preprocessed/training/visual_odometry/vo_{str(seq).zfill(4)}_{str(frame).zfill(6)}.txt', dtype=np.float32)
    R4 = np.eye(4)
    R4[:3, :3] = np.matmul(R1, R2)
    vo_mat = np.matmul(np.transpose(R4), np.matmul(vo_mat, R4))

    pc = copy.deepcopy(pc_velo)
    im = Image.open(f'{basepath_kitti}/training/image_02/{str(seq).zfill(4)}/{str(frame).zfill(6)}.png')
    if from_bbox2d:
        boxvec2d = tracklet[13:17]
        tracklet_pc = extract_pc_in_box2d(pc[:, :3], pc_velo[:, :3], boxvec2d, calib, im.size[0], im.size[1])
    else:
        pc[:, :3] = np.matmul(pc[:, :3], R.T)
        boxvec = tracklet[6:13]
        qs = compute_box_3d(boxvec)
        tracklet_pc = extract_pc_in_box3d(pc[:, :3], qs)[0]
        tracklet_pc[:, :3] = np.matmul(tracklet_pc[:, :3], R)
    #  print(tracklet_pc)
    tracklet_pc_color = np.empty((tracklet_pc.shape[0], 6), dtype=np.float32)
    tracklet_pc_color[:, :3] = tracklet_pc
    tracklet_pc_color[:, 3:] = extract_color_from_pc(tracklet_pc, calib, R, im)

    return tracklet_pc_color, vo_mat, np.linalg.inv(vo_mat)


def extract_pointclouds(tracklet1, tracklet2, from_bbox2d, calib):
    return extract_pointcloud(tracklet1, from_bbox2d, calib), extract_pointcloud(tracklet2, from_bbox2d, calib)


def get_transform_components(boxvec):
    R1 = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    R2 = np.array([[0., -1., 0], [1, 0, 0], [0, 0, 1]])
    R = np.matmul(R1, R2)
    position = boxvec[:3]
    angle = boxvec[6]
    position = np.matmul(position, R)
    h, w, l = boxvec[3:6]
    position[2] += h / 2.0
    return position, angle


def get_relative_transform(boxvec1, boxvec2):
    R1 = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    R2 = np.array([[0., -1., 0], [1, 0, 0], [0, 0, 1]])
    R4 = np.eye(4)
    R4[:3, :3] = np.matmul(R1, R2)
    translation = boxvec2[:3] - boxvec1[:3]
    #  z_difference = translation[1]
    #  translation[1] = 0.  # Constrain translation to ground plane (up-axis in KITTI-space is -y)
    angle = boxvec2[6] - boxvec1[6]  # TODO: constrain to -pi, pi
    rotation_center = boxvec1[:3]
    #  mat = get_mat_angle(translation, angle, rotation_center)
    #  mat = np.matmul(np.transpose(R4), np.matmul(mat, R4))

    translation = np.matmul(translation, R4[:3, :3])  # Now transform translation from KITTI to global coordinate system
    z_difference = translation[2]
    translation[2] = 0.
    rotation_center = np.matmul(rotation_center, R4[:3, :3])  # Same for rotation center
    mat = get_mat_angle(translation, angle, rotation_center)
    return mat, translation, angle, rotation_center, z_difference


# From frustum pointnets/kitti_util.py
def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# Adapted from frustum pointnets/kitti_util.py
def compute_box_3d(boxvec):
    ''' Returns:
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(boxvec[6])

    # 3d bounding box dimensions
    h, w, l = boxvec[3:6]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + boxvec[0]
    corners_3d[1, :] = corners_3d[1, :] + boxvec[1]
    corners_3d[2, :] = corners_3d[2, :] + boxvec[2]

    # project the 3d bounding box into the image plane
    return np.transpose(corners_3d)


####################################

vres = 64
vfov = 26.9
hres = 4500
hfov = 360.0

n = vres * hres
ray_origins = np.zeros((n, 3))
ray_directions = np.zeros((n, 3))

cam_offset = np.array([200, -300, 100])
cam_offset = np.array([0, 0, 0])

for vidx in range(vres):
    for hidx in range(hres):
        idx = vidx * hres + hidx
        vangle = -vfov / 2.0 + vfov / (vres - 1) * vidx
        if hfov == 360.0:
            hangle = -hfov / 2.0 + hfov / (hres) * hidx
        else:
            hangle = -hfov / 2.0 + hfov / (hres - 1) * hidx

        xoffset = np.sin(hangle / 180. * np.pi)
        yoffset = np.cos(hangle / 180. * np.pi)
        zoffset = np.tan(vangle / 180. * np.pi)
        #         print(xoffset, yoffset)
        ray_origins[idx, :] = np.array([0, 0, 0]) + cam_offset
        ray_directions[idx, :] = np.array([xoffset, yoffset, zoffset]) * 120.0


class Scene:
    def __init__(self):
        self.additional_meta = dict()
        self.transform = RandomTransform([4, 20])

    def save_pointclouds(self, basepath, scene_idx):
        for idx, pointcloud in enumerate(self.pointclouds):
            np.save(f'{basepath}/pointcloud{idx+1}/{str(scene_idx).zfill(8)}', pointcloud)

    def save_transform(self, basepath, scene_idx):
        np.save(f'{basepath}/transform/{str(scene_idx).zfill(8)}', self.transform.rel_transform)

    def save_meta(self, basepath, scene_idx):
        base_data = {
            'start_position': np_to_str(self.transform.start_position),
            'start_angle': self.transform.start_angle,
            'end_position': np_to_str(self.transform.end_position),
            'end_angle': self.transform.end_angle,
            'translation': np_to_str(self.transform.translation),
            'rel_angle': self.transform.rel_angle,
        }
        data = {**base_data, **self.additional_meta}
        with open(f'{basepath}/meta/{str(scene_idx).zfill(8)}.json', 'w') as outfile:
            json.dump(data, outfile)


class FromKITTIScene(Scene):
    def __init__(self, seq, tracklet1, tracklet2, from_bbox2d=False):
        super().__init__()
        assert tracklet1[0] == tracklet2[0]  # same sequence
        assert tracklet1[2] == tracklet2[2]  # same trackid (object)
        assert tracklet1[3] == tracklet2[3]  # same class
        seq = int(tracklet1[0])
        calib = Calibration(f'{basepath_kitti}/training/calib/{seq:04}.txt')
        (pc1, vo1, vo1_inv), (pc2, vo2, vo2_inv) = extract_pointclouds(tracklet1, tracklet2, from_bbox2d, calib=calib)
        rel_transform, translation, angle, rotation_center, z_difference = get_relative_transform(tracklet1[6:13].astype(float), tracklet2[6:13].astype(float))
        pc2[:, 2] -= z_difference
        self.pointclouds = [pc1, pc2]

        pc1_center, pc1_angle = get_transform_components(tracklet1[6:13].astype(float))
        pc2_center, pc2_angle = get_transform_components(tracklet2[6:13].astype(float))
        self.transform.start_position = pc1_center
        self.transform.start_angle = pc1_angle
        self.transform.end_position = pc2_center
        self.transform.end_angle = pc2_angle
        self.transform.translation = translation
        self.transform.rel_angle = angle

        self.additional_meta = {
            'class': tracklet1[3],
            'truncated': tracklet1[4],
            'occluded': tracklet1[5],
            'seq': tracklet1[0],
            'frames': [tracklet1[1], tracklet2[1]],
            'trackids': [tracklet1[2], tracklet2[2]],
            'vo1': np_to_str(vo1),
            'vo2': np_to_str(vo2),
            'bbox3ds': [np_to_str(tracklet1[6:13].astype(float)), np_to_str(tracklet2[6:13].astype(float))],
            'bbox2ds': [np_to_str(tracklet1[13:17].astype(float)), np_to_str(tracklet2[13:17].astype(float))],
        }


class FromHeldScene(Scene):
    def __init__(self, trackid, frame1, frame2, tracklet1, tracklet2):
        super().__init__()
        pc1, timestamp1 = tracklet1
        pc2, timestamp2 = tracklet2
        self.pointclouds = [pc1, pc2]

        pc1_center, pc1_angle = np.array([0., 0, 0]), 0.
        pc2_center, pc2_angle = np.array([0., 0, 0]), 0.
        self.transform.start_position = pc1_center
        self.transform.start_angle = pc1_angle
        self.transform.end_position = pc2_center
        self.transform.end_angle = pc2_angle
        self.transform.translation = np.array([0., 0, 0])
        self.transform.rel_angle = 0.

        self.additional_meta = {'class': 'Car', 'frames': [frame1, frame2], 'timestamps': [timestamp1, timestamp2], 'trackid': trackid}


class SyntheticScene(Scene):
    def __init__(self, seed, version, second_object_set=False, polar_dist_range=[4, 20], obj_size_range=dict(car=[6, 6], person=[1.6, 2.0]), allow_persons=False, person_prob=0.2):
        super().__init__()
        self.seed = seed
        self.version = version
        self.transform = RandomTransform(polar_dist_range)
        self.cat = 'car'
        if allow_persons and np.random.random() < person_prob:
            self.cat = 'person'
        self.mesh_scale = np.random.uniform(*obj_size_range[self.cat])
        if second_object_set:
            if self.cat == 'car':
                mesh_blacklist = []
                mesh_ids = [i for i in range(54, 104) if i not in mesh_blacklist]
                assert len(mesh_ids) == 50
            elif self.cat == 'person':
                mesh_blacklist = [1, 13, 16, 19, 26, 29, 30, 40, 41, 44, 45, 46, 51, 58, 60, 76, 77, 82, 85, 86, 92, 93, 94, 101, 106, 107, 108]
                mesh_ids = [i for i in range(54, 105) if i not in mesh_blacklist]
                assert len(mesh_ids) == 40
            else:
                assert False
        else:
            if self.cat == 'car':
                mesh_blacklist = [21, 31, 46]
                mesh_ids = [i for i in range(1, 54) if i not in mesh_blacklist]
                assert len(mesh_ids) == 50
            elif self.cat == 'person':
                mesh_blacklist = [1, 13, 16, 19, 26, 29, 30, 40, 41, 44, 45, 46, 51, 58, 60, 76, 77, 82, 85, 86, 92, 93, 94, 101, 106, 107, 108]
                mesh_ids = [i for i in range(1, 54) if i not in mesh_blacklist]
                assert len(mesh_ids) == 40
            else:
                assert False
        self.mesh_id = np.random.choice(mesh_ids)
        self.mesh_fname = f'/globalwork/gross/ModelNet/ModelNet40Aligned/{self.cat}/train/{self.cat}_{str(self.mesh_id).zfill(4)}.off'
        if not os.path.isfile(self.mesh_fname):
            self.mesh_fname = f'/globalwork/gross/ModelNet/ModelNet40Aligned/{self.cat}/test/{self.cat}_{str(self.mesh_id).zfill(4)}.off'

    def __repr__(self):
        return f'T: {self.transform}, mesh_id: {self.mesh_id}'

    @property
    def mesh_start(self):
        return get_mesh(self.mesh_fname).clone().apply_scale(self.mesh_scale).apply_transform(self.transform.transform_start)

    @property
    def mesh_end(self):
        return get_mesh(self.mesh_fname).clone().apply_scale(self.mesh_scale).apply_transform(self.transform.transform_end)

    def generate_pointcloud(self):
        n_parts = 20
        ray_origins_parts = np.array_split(ray_origins, n_parts)
        ray_directions_parts = np.array_split(ray_directions, n_parts)
        both_locations = []
        for mesh in [self.mesh_start, self.mesh_end]:
            locations = np.empty((0, 3))
            for ray_origins_, ray_directions_ in zip(tqdm_notebook(ray_origins_parts), ray_directions_parts):
                index_tri_, index_ray_, locations_ = mesh.ray.intersects_id(ray_origins_, ray_directions_, multiple_hits=False, return_locations=True)
                if type(locations_) == np.ndarray and len(locations_) > 0:
                    locations = np.vstack([locations, locations_])
                elif type(locations_) == list and len(locations_) > 0:
                    print('Error: non-zero length list', locations_)
            both_locations.append(locations)
        self.pointclouds = both_locations

    def generate_pointcloud_embree(self, add_noise=True, sigma=0.05, clip=0.05):
        n_parts = 20
        ray_origins_parts = np.array_split(ray_origins, n_parts)
        ray_directions_parts = np.array_split(ray_directions, n_parts)
        both_locations = []
        for mesh in [self.mesh_start, self.mesh_end]:
            locations = np.empty((0, 3))
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
            for ray_origins_, ray_directions_ in zip(ray_origins_parts, ray_directions_parts):
                index_tri_, index_ray_, locations_ = intersector.intersects_id(ray_origins_, ray_directions_, multiple_hits=False, return_locations=True)
                if type(locations_) == np.ndarray and len(locations_) > 0:
                    locations = np.vstack([locations, locations_])
                elif type(locations_) == list and len(locations_) > 0:
                    print('Error: non-zero length list', locations_)
            if add_noise:
                noise_strength_at_dist = np.maximum(0.005, sigma * np.linalg.norm(mesh.centroid) / 80.)
                noise = np.clip(noise_strength_at_dist * np.random.randn(locations.shape[0], 3), -1 * clip, clip)
                locations = locations + noise
            both_locations.append(locations)
        self.pointclouds = both_locations

    def save_meta(self, basepath, scene_idx):
        self.additional_meta = {
            'version': self.version,
            'seed': self.seed,
            'mesh_id': int(self.mesh_id),
            'mesh_scale': self.mesh_scale,
            'cat': self.cat,
        }
        super().save_meta(basepath, scene_idx)

    def show(self, predicted_transform=None):
        # stack rays into line segments for visualization as Path3D
        #     ray_visualize = trimesh.load_path(np.hstack((ray_origins,
        #                                                  ray_origins + ray_directions*5.0)).reshape(-1, 2, 3))
        if predicted_transform is not None:
            predicted_transform_ray_origins = np.array([self.transform.transform_start[:3, 3]])
            predicted_transform_ray_directions = np.array([predicted_transform[:3]])
            ray_visualize = trimesh.load_path(np.hstack((predicted_transform_ray_origins, predicted_transform_ray_origins + predicted_transform_ray_directions * 1.0)).reshape(-1, 2, 3))
        mesh = self.mesh_start
        mesh2 = self.mesh_end
        objects = [mesh, mesh2]
        if predicted_transform is not None:
            objects.append(ray_visualize)
        scene_ = trimesh.Scene(objects)
        #  print(scene.camera)
        #  scene.camera.transform = np.eye(4)
        #  scene.camera.transform[1,3]=-100
        #  scene.camera.transform = np.eye(4)
        #  scene.camera.transform[2,3]=1
        return scene_.show()


class SyntheticSceneCats(SyntheticScene):
    def __init__(self, seed, version, cats, second_object_set=False, polar_dist_range=[4, 20], obj_size_range=[1., 5.0]):
        super(Scene, self).__init__()
        self.seed = seed
        self.version = version
        self.transform = RandomTransform(polar_dist_range)
        self.cat = np.random.choice(cats)
        self.mesh_scale = np.random.uniform(*obj_size_range)
        mesh_ids = np.arange(20) + 1
        if second_object_set:
            mesh_ids = np.arange(20) + 20 + 1
        self.mesh_id = np.random.choice(mesh_ids)
        self.mesh_fname = f'/globalwork/gross/ModelNet/ModelNet40Aligned/{self.cat}/train/{self.cat}_{str(self.mesh_id).zfill(4)}.off'
        if not os.path.isfile(self.mesh_fname):
            self.mesh_fname = f'/globalwork/gross/ModelNet/ModelNet40Aligned/{self.cat}/test/{self.cat}_{str(self.mesh_id).zfill(4)}.off'


###########


class ICP:
    def _preprocess_point_cloud(pc, voxel_size):
        pcd_down = o3.voxel_down_sample(pc, voxel_size)

        radius_normal = voxel_size * 2
        o3.estimate_normals(pcd_down, o3.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3.compute_fpfh_feature(pcd_down, o3.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def _icp_global_prepare_dataset(source, target, voxel_size):
        source_down, source_fpfh = ICP._preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = ICP._preprocess_point_cloud(target, voxel_size)
        return source_down, target_down, source_fpfh, target_fpfh

    def constrain_transform(transform, constrain_rotation=None, constrain_translation=None):
        #  Adapted from https://github.com/CloudCompare/CloudCompare/blob/master/CC/src/RegistrationTools.cpp: RegistrationTools::FilterTransformation (002e246936f113a2625dd2fe83d19a73d5fce257)
        new_transform = np.eye(4)
        new_transform[:3, 3] = transform[:3, 3]

        #  print(transform)

        if constrain_translation is not None and type(constrain_translation) == str:
            if 'x' not in constrain_translation:
                new_transform[0, 3] = 0
            if 'y' not in constrain_translation:
                new_transform[1, 3] = 0
            if 'z' not in constrain_translation:
                new_transform[2, 3] = 0

        if constrain_rotation is not None:
            if constrain_rotation == 'xy':
                R = transform[:3, :3]
                if R[2, 0] < 1.0:
                    #  theta_rad = -asin(R.getValue(2,0));
                    #  cos_theta = cos(theta_rad);
                    #  phi_rad = atan2(R.getValue(1,0)/cos_theta, R.getValue(0,0)/cos_theta);
                    #  cos_phi	= cos(phi_rad);
                    #  sin_phi	= sin(phi_rad);

                    #  outTrans.R.setValue(0,0,cos_phi);
                    #  outTrans.R.setValue(1,1,cos_phi);
                    #  outTrans.R.setValue(1,0,sin_phi);
                    #  outTrans.R.setValue(0,1,-sin_phi);

                    theta_rad = -np.arcsin(R[2, 0])
                    cos_theta = np.cos(theta_rad)
                    phi_rad = np.arctan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
                    cos_phi = np.cos(phi_rad)
                    sin_phi = np.sin(phi_rad)

                    new_transform[0, 0] = cos_phi
                    new_transform[1, 1] = cos_phi
                    new_transform[1, 0] = sin_phi
                    new_transform[0, 1] = -sin_phi
                else:
                    #  simpler/faster to ignore this (very) specific case!
                    pass
            else:
                assert False
        #  print(new_transform)
        return new_transform


class Points:
    def __init__(self, ps, color=[255, 0, 0], opacity=None):
        if ps.shape[1] == 3:
            self.points = np.concatenate([ps, np.ones((ps.shape[0], 1))], axis=1)
        elif ps.shape[1] == 6:
            self.points = np.concatenate([ps[:, :3], np.ones((ps.shape[0], 1)), ps[:, 3:]], axis=1)
        self.color = clean_color(color)
        self.opacity = opacity

    def get_color_html(self):
        return f'rgb({self.color[0]},{self.color[1]},{self.color[2]})'

    def get_points(self, show_point_color=False):
        return get_points(self.points, self.color, show_point_color)

    def get_centroid(self):
        return np.mean(self.points, axis=0)

    def transform_points(self, mats):
        self.points = transform_points(self.points, mats)


class VisuMesh:
    def __init__(self, mesh, color=[255, 0, 0]):
        self.mesh = mesh
        ps = mesh.mesh.vertices.astype(np.float32)
        self.vertices = np.concatenate([ps, np.ones((ps.shape[0], 1))], axis=1)
        self.faces = mesh.mesh.faces
        self.color = clean_color(color)

    def get_vertices(self):
        return self.vertices[:, :3].tolist()

    def get_faces(self):
        return self.faces.tolist()

    def get_color_html(self):
        return f'rgb({self.color[0]},{self.color[1]},{self.color[2]})'

    def get_color_array(self):
        return np.array([self.color] * self.vertices.shape[0], dtype=np.int32)

    def get_centroid(self):
        return np.mean(self.vertices, axis=0)

    def transform_points(self, mats):
        self.vertices = transform_points(self.vertices, mats)


def get_box3d_lines(qs, color='blue'):
    lines = []
    # Partially from frustumpointnets/kitti_util.py
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        lines.append({'color': color, 'vertices': [qs[i], qs[j]]})

        i, j = k + 4, (k + 1) % 4 + 4
        lines.append({'color': color, 'vertices': [qs[i], qs[j]]})

        i, j = k, k + 4
        lines.append({'color': color, 'vertices': [qs[i], qs[j]]})
    return lines


class VisualizationScene:
    def __init__(self, ps=None, visibility_dict=dict(), background='black', show_origin=True):
        self.points = dict()
        self.meshes = dict()
        self.polylines = []
        if ps is not None:
            self.add_points(ps)
        if show_origin:
            self.add_polyline(origin_lines)
        self.visibility_dict = visibility_dict
        self.background = background

    def add_points(self, name, ps, color=[255, 0, 0], opacity=None):
        self.points[name] = Points(ps, color, opacity)
        if self.points[name].points.shape[0] == 0:
            print(f'Point cloud {name} has no points')

    def add_mesh(self, name, mesh, color):
        self.meshes[name] = VisuMesh(mesh, color)

    def add_polyline(self, line, color='blue'):
        self.polylines.extend(line)

    def add_arrow(self, fro, to, color='blue', relative=False):
        lines = [{'color': color, 'vertices': [fro, fro + to if relative else to]}]
        self.polylines.extend(lines)

    def add_box3d(self, boxvec, color='blue', inKITTICoordinates=True):
        # boxvec in KITTI format: x,y,z,xd,yd,zd,yrot
        assert inKITTICoordinates
        boxvec = copy.deepcopy(boxvec)
        R1 = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
        R2 = np.array([[0., -1., 0], [1, 0, 0], [0, 0, 1]])
        R = np.matmul(R1, R2)
        qs = compute_box_3d(boxvec)  # Still in KITTI coordinates
        qs = np.matmul(qs, R)

        lines = get_bbox_lines(qs, c=color)
        self.polylines.extend(lines)

    def add_boxes3d(self, boxvecs, color='blue', inKITTICoordinates=True):
        for boxvec in boxvecs:
            self.add_box3d(boxvec, color=color, inKITTICoordinates=inKITTICoordinates)

    def _do_icp_p2p(self, pc1, pc2, radius, init, constrain):
        return o3.registration_icp(pc1, pc2, radius, init, o3.TransformationEstimationPointToPoint(with_constraint=constrain, with_scaling=False))

    def _do_icp_goicp(self, pc1, pc2, constrain):
        assert False

    def _do_icp_global(self, pc1, pc2, constrain):
        voxel_size = 0.05
        source_down, target_down, source_fpfh, target_fpfh = ICP._icp_global_prepare_dataset(pc1, pc2, voxel_size)
        distance_threshold = voxel_size * 1.5
        result = o3.registration_ransac_based_on_feature_matching(source_down, target_down, source_fpfh, target_fpfh, distance_threshold, o3.TransformationEstimationPointToPoint(with_constraint=constrain, with_scaling=False), 4, [o3.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3.CorrespondenceCheckerBasedOnDistance(distance_threshold)], o3.RANSACConvergenceCriteria(4000000, 500))
        return result

    def _do_icp_fast_global(self, pc1, pc2, constrain):
        voxel_size = 0.05
        distance_threshold = voxel_size * 0.5
        source_down, target_down, source_fpfh, target_fpfh = ICP._icp_global_prepare_dataset(pc1, pc2, voxel_size)
        result = o3.registration_fast_based_on_feature_matching(source_down, target_down, source_fpfh, target_fpfh, o3.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
        return result

    def do_icp(self, name1, name2, method, init=np.eye(4), radius=0.2, constrain=False):
        pc1 = o3.geometry.PointCloud()
        pc1.points = o3.Vector3dVector(self.points[name1].points[:, :3])
        pc2 = o3.geometry.PointCloud()
        pc2.points = o3.Vector3dVector(self.points[name2].points[:, :3])

        start = time.time()
        sys.__stdout__.write(method + '\n')
        if method == 'p2p':
            reg_result = self._do_icp_p2p(pc1, pc2, radius, init, constrain=constrain)
        elif method == 'global':
            reg_result = self._do_icp_global(pc1, pc2, constrain=constrain)
        elif method == 'goicp':
            reg_result = self._do_icp_goicp(self.points[name1].points[:, :3], self.points[name2].points[:, :3], constrain=constrain)
        elif method == 'fast_global':
            reg_result = self._do_icp_fast_global(pc1, pc2, constrain=constrain)

        print(f'{method} {name1}->{name2} icp registration took {time.time() - start:.3} sec.', reg_result.transformation[:3, 3])

        before = o3.evaluate_registration(pc1, pc2, 0.02, init)
        after = o3.evaluate_registration(pc1, pc2, 0.02, reg_result.transformation)
        return [reg_result, before, after]

    def icp_and_add(self, name1, name2, color, method='p2p', radius=0.2, constrain=True):
        [reg_icp, before, after] = self.do_icp(name1, name2, method=method, radius=radius, constrain=constrain)
        name = f'{name1}_icp_{method}{"_unconstr" if not constrain else ""}'
        self.add_points(name, self.points[name1].points[:, :3], color=color)

        #  if constrain is not None:
        #      if constrain == 'simple':
        #          rotation_mat = reg_icp.transformation[:3,:3]
        #          rot_vec = Rotation.from_dcm(rotation_mat).as_rotvec()
        #          ground_plane_constrained_transform = get_mat_angle(reg_icp.transformation[:3,3], rot_vec[2])
        #          self.points[name].transform_points(ground_plane_constrained_transform)
        #      else:
        #          ground_plane_constrained_transform = ICP.constrain_transform(reg_icp.transformation, constrain_rotation='xy', constrain_translation='xy')
        #      self.points[name].transform_points(ground_plane_constrained_transform)
        #  else:
        #      self.points[name].transform_points(reg_icp.transformation)
        self.points[name].transform_points(reg_icp.transformation)
        return name

    def change_visibility(self, key, visible):
        pss = [ps for ps in self.scene.children if ps.name == key]
        for ps in pss:
            ps.material.visible = visible
        self.visibility_dict[key] = visible

    def show(self, show_point_color=False):
        initial_point_size = 0.03
        point_materials = []
        for idx, (name, points) in enumerate(self.points.items()):
            if idx == 0:
                cloud = PyntCloud(points.get_points(show_point_color=show_point_color))
                #  print(clean_polylines(self.polylines))
                self.scene = cloud.plot(return_scene=True, initial_point_size=initial_point_size, polylines=clean_polylines(self.polylines), background=self.background)
                pss = [ps for ps in self.scene.children if type(ps) == pythreejs.objects.Points_autogen.Points]
                assert len(pss) > 0
                pss[0].name = name
                if points.opacity is not None:
                    pss[0].material.opacity = points.opacity
                    pss[0].material.transparent = True
                point_materials.append(pss[0].material)
            else:
                ppoints = get_points(points.points, points.color, show_point_color=show_point_color)
                ps = pyntcloud.plot.pythreejs_backend.get_pointcloud_pythreejs(ppoints[["x", "y", "z"]].values, ppoints[['red', 'green', 'blue']].values / 255.)
                ps.name = name
                ps.material.size = initial_point_size
                self.scene.children = [ps] + list(self.scene.children)
                if points.opacity is not None:
                    ps.material.opacity = points.opacity
                    ps.material.transparent = True
                #  if name == 'original':
                #      ps.material.opacity = 0.3
                #      ps.material.transparent = True
                point_materials.append(ps.material)

        for idx, (name, mesh) in enumerate(self.meshes.items()):
            # https://render.githubusercontent.com/view/ipynb?commit=645ea6bea758555978f83bd0004ce561fa58d99c&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6a7570797465722d776964676574732f707974687265656a732f363435656136626561373538353535393738663833626430303034636535363166613538643939632f6578616d706c65732f4578616d706c65732e6970796e62&nwo=jupyter-widgets%2Fpythreejs&path=examples%2FExamples.ipynb&repository_id=15400194&repository_type=Repository#Buffer-Geometries
            vertices = mesh.get_vertices()
            faces = mesh.get_faces()
            vertexcolors = ['#ff0000' for _ in range(len(vertices))]
            # Map the vertex colors into the 'color' slot of the faces
            faces = [f + [None, [vertexcolors[i] for i in f], None] for f in faces]
            geometry = pythreejs.Geometry(vertices=vertices, faces=faces, colors=vertexcolors)
            geometry.exec_three_obj_method('computeFaceNormals')
            mesh_obj = pythreejs.Mesh(geometry=geometry, material=pythreejs.MeshBasicMaterial(color='white', side='DoubleSide'), position=[0, 0, 0])
            mesh_obj.name = name
            self.scene.children = [mesh_obj] + list(self.scene.children)

        for key, value in self.visibility_dict.items():
            self.change_visibility(key, value)

        widgets = []
        size = ipywidgets.FloatSlider(value=initial_point_size, min=0.0, max=initial_point_size * 10, step=initial_point_size / 100)
        for point_materials in point_materials:
            ipywidgets.jslink((size, 'value'), (point_materials, 'size'))
        widgets.append(ipywidgets.Label('Point size:'))
        widgets.append(size)
        display(ipywidgets.HBox(children=widgets))


#  if __name__ == '__main__':
#      test()
