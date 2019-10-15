import json
import logging
import os
import sys
import time
#  from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
from contextlib import contextmanager

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import evaluation
import open3d as o3
import provider
from pointcloud import ICP, get_mat_angle

logger = logging.getLogger('tp')


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


def load_pountclouds(file_idx, cfg, return_numpy=False):
    ps1 = np.load(f'{cfg.data.basepath}/pointcloud1/{str(file_idx).zfill(8)}.npy')[:, :3]
    ps2 = np.load(f'{cfg.data.basepath}/pointcloud2/{str(file_idx).zfill(8)}.npy')[:, :3]
    pc1_centroid = ps1.mean(axis=0)
    if return_numpy:
        return ps1, ps2, pc1_centroid
    pc1 = o3.geometry.PointCloud()
    pc1.points = o3.Vector3dVector(ps1)
    pc2 = o3.geometry.PointCloud()
    pc2.points = o3.Vector3dVector(ps2)
    return pc1, pc2, pc1_centroid


def get_median_init(pc1, pc2):
    approx_translation = np.median(np.asarray(pc2.points), axis=0) - np.median(np.asarray(pc1.points), axis=0)
    init = np.eye(4)
    init[:3, 3] = approx_translation
    return init


def get_centroid_init(pc1, pc2):
    approx_translation = np.mean(np.asarray(pc2.points), axis=0) - np.mean(np.asarray(pc1.points), axis=0)
    init = np.eye(4)
    init[:3, 3] = approx_translation
    return init


def icp_p2point(file_idx, cfg, radius=0.2, its=30, init=None, with_constraint=None):
    with_constraint = with_constraint if with_constraint is not None else cfg.evaluation.special.icp.with_constraint
    pc1, pc2, pc1_centroid = load_pountclouds(file_idx, cfg)
    if init is None:
        #  init = get_median_init(pc1, pc2)
        init = get_centroid_init(pc1, pc2)
    start = time.time()
    reg_p2p = o3.registration_icp(pc1, pc2, radius, init, o3.TransformationEstimationPointToPoint(with_constraint=with_constraint, with_scaling=False), o3.registration.ICPConvergenceCriteria(max_iteration=its))  # Default: 30
    time_elapsed = time.time() - start
    return reg_p2p.transformation, pc1_centroid, time_elapsed


def icp_p2plane(file_idx, cfg):
    assert False


def icp_o3_gicp(file_idx, cfg, refine=None, refine_radius=0.05, precomputed_results=None):
    pc1, pc2, pc1_centroid = load_pountclouds(file_idx, cfg)
    voxel_size = 0.05
    start = time.time()
    if precomputed_results is None:
        distance_threshold = voxel_size * 1.5
        source_down, target_down, source_fpfh, target_fpfh = ICP._icp_global_prepare_dataset(pc1, pc2, voxel_size)
        reg_res = o3.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            distance_threshold,
            o3.TransformationEstimationPointToPoint(with_constraint=cfg.evaluation.special.icp.with_constraint, with_scaling=False),
            4,  # scaling=False
            [o3.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3.RANSACConvergenceCriteria(4000000, 500))
        transformation = reg_res.transformation
    else:
        precomp_pred_translation, precomp_pred_angle, precomp_pred_center = precomputed_results
        transformation = get_mat_angle(precomp_pred_translation, precomp_pred_angle, precomp_pred_center)

    if refine is None:
        time_elapsed = time.time() - start
        return transformation, pc1_centroid, time_elapsed
    else:
        if refine == 'p2p':
            reg_p2p = o3.registration_icp(pc1, pc2, refine_radius, transformation, o3.TransformationEstimationPointToPoint(with_constraint=cfg.evaluation.special.icp.with_constraint, with_scaling=False))
            #  if file_idx == 8019:
            #  print('->', reg_p2p.transformation)
            time_elapsed = time.time() - start
            return reg_p2p.transformation, pc1_centroid, time_elapsed
        else:
            assert False


def icp_o3_gicp_fast(file_idx, cfg, refine=None, refine_radius=0.05, precomputed_results=None):
    pc1, pc2, pc1_centroid = load_pountclouds(file_idx, cfg)
    voxel_size = 0.05
    distance_threshold = voxel_size * 0.5
    start = time.time()
    if precomputed_results is None:
        source_down, target_down, source_fpfh, target_fpfh = ICP._icp_global_prepare_dataset(pc1, pc2, voxel_size)
        reg_res = o3.registration_fast_based_on_feature_matching(source_down, target_down, source_fpfh, target_fpfh, o3.FastGlobalRegistrationOption(with_constraint=cfg.evaluation.special.icp.with_constraint, maximum_correspondence_distance=distance_threshold))
        transformation = reg_res.transformation
    else:
        precomp_pred_translation, precomp_pred_angle, precomp_pred_center = precomputed_results
        transformation = get_mat_angle(precomp_pred_translation, precomp_pred_angle, precomp_pred_center)

    if refine is None:
        time_elapsed = time.time() - start
        return transformation, pc1_centroid, time_elapsed
    else:
        if refine == 'p2p':
            reg_p2p = o3.registration_icp(pc1, pc2, refine_radius, transformation, o3.TransformationEstimationPointToPoint(with_constraint=cfg.evaluation.special.icp.with_constraint, with_scaling=False))
            time_elapsed = time.time() - start
            return reg_p2p.transformation, pc1_centroid, time_elapsed
        else:
            assert False


def icp_goicp(file_idx, cfg, refine=None, refine_radius=0.05):
    assert False


def evaluate(cfg, use_old_results=False):
    val_idxs = provider.getDataFiles(f'{cfg.data.basepath}/split/val.txt')
    #  val_idxs = val_idxs[:100]

    epoch = 0
    total_time = 0.

    do_refinement = cfg.evaluation.special.icp.has('refine')
    refinement_method = cfg.evaluation.special.icp.refine if do_refinement else None

    if cfg.evaluation.special.icp.variant in ['o3_gicp', 'o3_gicp_fast'] and do_refinement:
        gicp_result_dir = f'{cfg.logging.logdir[:-4]}/val/eval{str(epoch).zfill(6)}'
        assert os.path.isdir(gicp_result_dir), gicp_result_dir
        assert os.path.isfile(f'{gicp_result_dir}/eval_180.json'), f'{gicp_result_dir}/eval_180.json'
        eval_dict = json.load(open(f'{gicp_result_dir}/eval_180.json', 'r'))
        precomp_time = eval_dict['mean_time'] * float(len(val_idxs))
        total_time += precomp_time
        precomp_pred_translations = np.load(f'{gicp_result_dir}/pred_translations.npy')
        precomp_pred_angles = np.load(f'{gicp_result_dir}/pred_angles.npy')
        precomp_pred_centers = np.load(f'{gicp_result_dir}/pred_s1_pc1centers.npy')
        print('Precomputed results loaded')

    pcs1, pcs2, all_gt_translations, all_gt_angles, all_gt_pc1centers, all_gt_pc2centers, all_gt_pc1angles, all_gt_pc2angles = provider.load_batch(val_idxs, override_batch_size=len(val_idxs))
    eval_dir = f'{cfg.logging.logdir}/val/eval{str(epoch).zfill(6)}'
    if use_old_results and os.path.isfile(f'{eval_dir}/pred_translations.npy'):
        all_pred_translations = np.load(f'{eval_dir}/pred_translations.npy')
        all_pred_angles = np.load(f'{eval_dir}/pred_angles.npy')
        all_pred_centers = np.load(f'{eval_dir}/pred_s1_pc1centers.npy')
    else:
        all_pred_translations = np.empty((len(val_idxs), 3), dtype=np.float32)
        all_pred_angles = np.empty((len(val_idxs), 1), dtype=np.float32)
        all_pred_centers = np.empty((len(val_idxs), 3), dtype=np.float32)

        for idx, file_idx in enumerate(tqdm(val_idxs)):
            if cfg.evaluation.special.icp.variant == 'p2point':
                pred_transform, pred_center, time_elapsed = icp_p2point(file_idx, cfg, radius=0.10)
            elif cfg.evaluation.special.icp.variant == 'p2plane':
                pred_transform, pred_center, time_elapsed = icp_p2plane(file_idx, cfg)
            elif cfg.evaluation.special.icp.variant == 'goicp':
                pred_transform, pred_center, time_elapsed = icp_goicp(file_idx, cfg, refine=refinement_method, refine_radius=0.10)
            elif cfg.evaluation.special.icp.variant == 'o3_gicp':
                pred_transform, pred_center, time_elapsed = icp_o3_gicp(file_idx, cfg, refine=refinement_method, refine_radius=0.10, precomputed_results=(precomp_pred_translations[idx], precomp_pred_angles[idx], precomp_pred_centers[idx]) if do_refinement else None)
            elif cfg.evaluation.special.icp.variant == 'o3_gicp_fast':
                pred_transform, pred_center, time_elapsed = icp_o3_gicp_fast(file_idx, cfg, refine=refinement_method, refine_radius=0.10, precomputed_results=(precomp_pred_translations[idx], precomp_pred_angles[idx], precomp_pred_centers[idx]) if do_refinement else None)
            else:
                assert False
            #  all_pred_centers[idx] = pred_center
            #  Important! The output of the ICP functions is around the origin, not around the centroid as used internally
            all_pred_centers[idx] = np.array([0., 0, 0])

            all_pred_translations[idx] = pred_transform[:3, 3]
            rotation_mat = pred_transform[:3, :3]
            rot_vec = Rotation.from_dcm(rotation_mat).as_rotvec()
            all_pred_angles[idx] = rot_vec[2]
            total_time += time_elapsed

        os.makedirs(eval_dir, exist_ok=True)
        np.save(f'{eval_dir}/pred_translations.npy', all_pred_translations)
        np.save(f'{eval_dir}/pred_angles.npy', all_pred_angles)
        np.save(f'{eval_dir}/pred_s1_pc1centers.npy', all_pred_centers)

    for accept_inverted_angle in [False, True]:
        eval_dict = evaluation.evaluate(cfg, val_idxs, all_pred_translations, all_pred_angles, all_gt_translations, all_gt_angles, all_pred_centers, all_gt_pc1centers, eval_dir=eval_dir, accept_inverted_angle=accept_inverted_angle, mean_time=total_time / len(val_idxs))
        logger.info(eval_dict)
