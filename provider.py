import numpy as np
from config import configGlobal as cfg
import json
from pointcloud import str_to_np
import logging

logger = logging.getLogger('tp')


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def getDataFiles(list_filename):
    return [int(line.rstrip()) for line in open(list_filename)]


#  def load_h5(h5_filename):
#  f = h5py.File(h5_filename)
#  data = f['data'][:]
#  label = f['label'][:]
#  return (data, label)


def load_from_separate_files(idx, dont_load_pointclouds=False):
    data = json.load(open(f'{cfg.data.basepath}/meta/{str(idx).zfill(8)}.json', 'r'))
    translation, rel_angle = str_to_np(data['translation']), data['rel_angle']
    pc1center, pc2center = str_to_np(data['start_position']), str_to_np(data['end_position'])
    pc1angle, pc2angle = data['start_angle'], data['end_angle']
    if dont_load_pointclouds:
        return translation, rel_angle, pc1center, pc2center, pc1angle, pc2angle

    pc1 = np.load(f'{cfg.data.basepath}/pointcloud1/{str(idx).zfill(8)}.npy')
    pc2 = np.load(f'{cfg.data.basepath}/pointcloud2/{str(idx).zfill(8)}.npy')
    if pc1.shape[0] == 0 or pc2.shape[0] == 0:
        logger.error(f'Empty pointcloud! {idx}')
    pc1 = pc1[np.random.choice(pc1.shape[0], cfg.model.num_points, replace=True), :] if pc1.shape[0] > 0 else np.zeros((cfg.model.num_points, 3), dtype=np.float32)
    pc2 = pc2[np.random.choice(pc2.shape[0], cfg.model.num_points, replace=True), :] if pc2.shape[0] > 0 else np.zeros((cfg.model.num_points, 3), dtype=np.float32)
    #  pc1 = np.repeat(pc1, 10, axis=0)[:cfg.model.num_points,:]  # For testing repeated validation. If this is deterministic, the it leads to the same results
    #  pc2 = np.repeat(pc2, 10, axis=0)[:cfg.model.num_points,:]
    #  transform = np.load(f'{cfg.data.basepath}/transform/{str(idx).zfill(8)}.npy')
    #  q = quaternion.from_rotation_matrix(transform[:3,:3])
    #  p = transform[:3,3]
    #  return pc1, pc2, np.concatenate([p, quaternion.as_float_array(q)])
    return pc1, pc2, translation, rel_angle, pc1center, pc2center, pc1angle, pc2angle


def load_batch(indices, override_batch_size=None, dont_load_pointclouds=False):
    batch_size = cfg.training.batch_size if override_batch_size is None else override_batch_size
    pcs1 = np.empty((batch_size, cfg.model.num_points, cfg.data.num_channels))
    pcs2 = np.empty((batch_size, cfg.model.num_points, cfg.data.num_channels))

    translations = np.empty((batch_size, 3))
    rel_angles = np.empty((batch_size, 1))

    pc1centers = np.empty((batch_size, 3))
    pc2centers = np.empty((batch_size, 3))
    pc1angles = np.empty((batch_size, 1))
    pc2angles = np.empty((batch_size, 1))

    for idx, ex_idx in enumerate(indices):
        if dont_load_pointclouds:
            translation, rel_angle, pc1center, pc2center, pc1angle, pc2angle = load_from_separate_files(ex_idx, dont_load_pointclouds=dont_load_pointclouds)
        else:
            pc1, pc2, translation, rel_angle, pc1center, pc2center, pc1angle, pc2angle = load_from_separate_files(ex_idx, dont_load_pointclouds=dont_load_pointclouds)
            pcs1[idx] = pc1[:, :3]
            pcs2[idx] = pc2[:, :3]

        translations[idx] = translation
        rel_angles[idx] = rel_angle

        pc1centers[idx] = pc1center
        pc2centers[idx] = pc2center
        pc1angles[idx] = pc1angle
        pc2angles[idx] = pc2angle
    return pcs1, pcs2, translations, rel_angles, pc1centers, pc2centers, pc1angles, pc2angles


def loadDataFile(idx):
    #  return load_h5(idx)
    return load_from_separate_files(idx)


#  def load_h5_data_label_seg(h5_filename):
#  f = h5py.File(h5_filename)
#  data = f['data'][:]
#  label = f['label'][:]
#  seg = f['pid'][:]
#  return (data, label, seg)

#  def loadDataFile_with_seg(filename):
#  return load_h5_data_label_seg(filename)
