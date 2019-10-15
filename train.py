import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tp_utils'))
import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('Agg')

import argparse
import datetime
import numpy as np
import tensorflow as tf
import provider
import copy
import models.tp8 as MODEL_tp8
from config import load_config, configGlobal, save_config
import logging
from tqdm import tqdm as tqdm_orig
import evaluation
import icp
import time
from pointcloud import get_mat_angle
from scipy.spatial.transform import Rotation
from tensorflow.python import pywrap_tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 1==ignore INFO
np.set_printoptions(precision=4, linewidth=200)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
default_config_path = os.path.join(BASE_DIR, './configs/default.json')

parser = argparse.ArgumentParser()
parser.add_argument('operation', choices=['train', 'eval_only'], help='Operation to run')
parser.add_argument('--config', required=True, default='', help='Config file')
parser.add_argument('--refineICP', action='store_true', help='Whether the results should be refined with ICP')
parser.add_argument('--its', required=False, default=30, help='How many iteration the result should be refined with ICP')
parser.add_argument('--use_old_results', action='store_true', help='Whether the model should actually be loaded and used for inference, or old results should be used for ICP refinement')
parser.add_argument('--refineICPmethod', required=False, default='p2p', choices=['p2p'], help='ICP method for refinement')
parser.add_argument('--eval_epoch', required=False, default='199', help='Epoch to eval in eval_only mode')
FLAGS = parser.parse_args()

load_config(FLAGS.config)
cfg = configGlobal

datestr_now = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(cfg.logging.logdir, exist_ok=True)
configcopyfile = f'{cfg.logging.logdir}/config.json'
if os.path.exists(configcopyfile):
    configcopyfile = f'{configcopyfile[:-5]}_{datestr_now}.json'
save_config(configcopyfile)

if cfg.model.model == 'tp8':
    MODEL = MODEL_tp8
else:
    assert False


class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class MultiLineFormatter(logging.Formatter):
    def format(self, record):
        s = logging.Formatter.format(self, record)
        try:
            header, footer = s.split(record.message)
            s = s.replace('\n', '\n' + ' ' * len(header))
            return s
        except Exception:
            return s


formatter = MultiLineFormatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', '%Y-%m-%d %H:%M:%S')

streamhandler = TqdmLoggingHandler(logging.INFO)
streamhandler.setLevel(logging.INFO)
streamhandler.setFormatter(formatter)

logfile = f'{cfg.logging.logdir}/out.log'
if os.path.exists(logfile):
    logfile = f'{logfile[:-4]}_{datestr_now}.log'
filehandler = logging.FileHandler(logfile)
filehandler.setLevel(logging.DEBUG)
filehandler.setFormatter(formatter)

tf_logger = logging.getLogger('tensorflow')
for handler in tf_logger.handlers:
    handler.close()
    tf_logger.removeFilter(handler)
    tf_logger.removeHandler(handler)
tf_logger.addHandler(streamhandler)
tf_logger.addHandler(filehandler)
tf_logger.setLevel(logging.DEBUG)

logger = logging.getLogger('tp')
logger.addHandler(streamhandler)
logger.addHandler(filehandler)
logger.setLevel(logging.DEBUG)

logger.debug(configGlobal)


class tqdm(tqdm_orig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, leave=False)

    def __iter__(self):
        return super().__iter__()

    def __del__(self):
        self.ascii = True
        self.dynamic_ncols = False
        self.ncols = 100
        logger.info(self.__repr__())
        return super().__del__()


TRAIN_INDICES = provider.getDataFiles(f'{cfg.data.basepath}/split/train.txt')
VAL_INDICES = provider.getDataFiles(f'{cfg.data.basepath}/split/val.txt')


def get_learning_rate(batch):
    num_batches_per_epoch = len(TRAIN_INDICES) // cfg.training.batch_size

    if cfg.training.lr_extension.mode == 'decay':
        lr_decay_step = cfg.training.lr_extension.step
        if cfg.training.lr_extension.per == 'step':
            pass
        elif cfg.training.lr_extension.per == 'epoch':
            lr_decay_step *= cfg.training.batch_size * num_batches_per_epoch
        else:
            assert False

        learning_rate = tf.train.exponential_decay(
            cfg.training.learning_rate,  # Base learning rate.
            batch * cfg.training.batch_size,  # Current index into the dataset.
            lr_decay_step,
            cfg.training.lr_extension.rate,
            staircase=True)
    elif cfg.training.lr_extension.mode == 'clr':
        assert False
        #  learning_rate = clr.cyclic_learning_rate(batch * cfg.training.batch_size, learning_rate=cfg.training.learning_rate, max_lr=10*cfg.training.learning_rate, step_size=len(TRAIN_INDICES)//cfg.training.batch_size*6)
        #  learning_rate = clr.cyclic_learning_rate(batch * cfg.training.batch_size, step_size=len(TRAIN_INDICES)//cfg.training.batch_size*6)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    num_batches_per_epoch = len(TRAIN_INDICES) // cfg.training.batch_size

    assert cfg.training.bn_extension.mode == 'decay'

    bn_decay_step = cfg.training.bn_extension.step
    if cfg.training.bn_extension.per == 'step':
        pass
    elif cfg.training.bn_extension.per == 'epoch':
        bn_decay_step *= cfg.training.batch_size * num_batches_per_epoch
    else:
        assert False

    bn_momentum = tf.train.exponential_decay(cfg.training.bn_extension.init, batch * cfg.training.batch_size, bn_decay_step, cfg.training.bn_extension.rate, staircase=True)
    bn_decay = tf.minimum(cfg.training.bn_extension.clip, 1 - bn_momentum)
    return bn_decay


def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
    varlist = []
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            varlist.append(key)
    return varlist


def train(eval_only=False, eval_epoch=None, eval_only_model_to_load=None, do_timings=False, override_batch_size=None):
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(cfg.gpu_index)):
            pcs1, pcs2, translations, rel_angles, pc1centers, pc2centers, pc1angles, pc2angles = MODEL.placeholder_inputs(cfg.training.batch_size, cfg.model.num_points)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('hyperparameters/bn_decay', bn_decay)

            # Get model and loss
            end_points = MODEL.get_model(pcs1, pcs2, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pcs1, pcs2, translations, rel_angles, pc1centers, pc2centers, pc1angles, pc2angles, end_points)
            tf.summary.scalar('losses/loss', loss)

            #  correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            #  accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(cfg.training.batch_size)
            #  tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('hyperparameters/learning_rate', learning_rate)
            if cfg.training.optimizer.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=cfg.training.optimizer.momentum)
            elif cfg.training.optimizer.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                assert False, "Invalid optimizer"
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=1000)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #  merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(cfg.logging.logdir, 'train'), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(cfg.logging.logdir, 'val'))
        val_writer_180 = tf.summary.FileWriter(os.path.join(cfg.logging.logdir, 'val_180'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pcs1': pcs1, 'pcs2': pcs2, 'translations': translations, 'rel_angles': rel_angles, 'is_training_pl': is_training_pl, 'pred_translations': end_points['pred_translations'], 'pred_remaining_angle_logits': end_points['pred_remaining_angle_logits'], 'pc1centers': pc1centers, 'pc2centers': pc2centers, 'pc1angles': pc1angles, 'pc2angles': pc2angles, 'pred_s1_pc1centers': end_points['pred_s1_pc1centers'], 'pred_s1_pc2centers': end_points['pred_s1_pc2centers'], 'pred_s2_pc1centers': end_points['pred_s2_pc1centers'], 'pred_s2_pc2centers': end_points['pred_s2_pc2centers'], 'pred_pc1angle_logits': end_points['pred_pc1angle_logits'], 'pred_pc2angle_logits': end_points['pred_pc2angle_logits'], 'loss': loss, 'train_op': train_op, 'merged': merged, 'step': batch}

        start_epoch = 0
        if eval_only:
            model_to_load = cfg.logging.logdir
            if eval_only_model_to_load is not None:
                model_to_load = eval_only_model_to_load
            if not FLAGS.use_old_results and not do_timings:
                assert os.path.isfile(f'{model_to_load}/model-{eval_epoch}.index'), f'{model_to_load}/model-{eval_epoch}.index'
                saver.restore(sess, f'{model_to_load}/model-{eval_epoch}')
            start_epoch = int(eval_epoch)

            if eval_only_model_to_load is None:
                num_batches_per_epoch = len(TRAIN_INDICES) // cfg.training.batch_size

                if FLAGS.use_old_results or do_timings:
                    start_epoch = int(eval_epoch)
                else:
                    restored_batch = sess.run(batch)
                    assert restored_batch % num_batches_per_epoch == 0
                    start_epoch = restored_batch // num_batches_per_epoch - 1
                    assert start_epoch == int(eval_epoch)
            logger.info(f'Evaluating at epoch {start_epoch}')
        else:
            if os.path.isfile(os.path.join(cfg.logging.logdir, 'model.ckpt.index')):
                saver.restore(sess, os.path.join(cfg.logging.logdir, 'model.ckpt'))

                num_batches_per_epoch = len(TRAIN_INDICES) // cfg.training.batch_size

                restored_batch = sess.run(batch)
                assert restored_batch % num_batches_per_epoch == 0
                start_epoch = restored_batch // num_batches_per_epoch
                logger.info(f'Continuing training at epoch {start_epoch}')
            elif cfg.training.pretraining.model != '':
                assert os.path.isfile(cfg.training.pretraining.model + '.index')
                variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                variables_to_load = [var for var in variables if var not in [batch]]
                saverPretraining = tf.train.Saver(variables_to_load)
                saverPretraining.restore(sess, cfg.training.pretraining.model)
                #  print(variables)
                #  print(len(variables), len(variables_to_load))
                #  varlist = print_tensors_in_checkpoint_file(file_name=cfg.training.pretraining.model, all_tensors=True, tensor_name=None)
                #  print(varlist)
                #  print(variables_to_load[:8])
                #  print(len(varlist))
                restored_batch = sess.run(batch)
                assert restored_batch == 0
                logger.info(f'Pre-trained weights loaded from {cfg.training.pretraining.model}, starting initial evaluation')
                lr, bn_d = sess.run([learning_rate, bn_decay])
                eval_one_epoch(sess, ops, val_writer, val_writer_180, 'pretr', eval_only=False, do_timings=False)
                logger.info(f'Initial evaluation finished')

        try:
            start = time.time()
            for epoch in range(start_epoch, cfg.training.num_epochs):
                lr, bn_d = sess.run([learning_rate, bn_decay])
                logger.info('**** EPOCH %03d ****    ' % (epoch) + f'lr: {lr:.8f}, bn_decay: {bn_d:.8f}')
                #  sys.stdout.flush()

                if not eval_only:
                    train_one_epoch(sess, ops, train_writer, epoch)
                if eval_only or True or epoch % 10 == 0:
                    if do_timings:
                        for i in range(10):
                            eval_one_epoch(sess, ops, val_writer, val_writer_180, epoch, eval_only=eval_only, do_timings=True, override_batch_size=override_batch_size)
                    else:
                        eval_one_epoch(sess, ops, val_writer, val_writer_180, epoch, eval_only=eval_only, do_timings=False)
                if eval_only:
                    break

                if not eval_only:
                    was_last_epoch = epoch == cfg.training.num_epochs - 1
                    # Save the variables to disk.
                    if epoch % 2 == 0 or was_last_epoch:
                        save_path = saver.save(sess, os.path.join(cfg.logging.logdir, "model.ckpt"))
                        logger.info("Model saved in file: %s" % save_path)

                    if epoch % 5 == 0 or was_last_epoch or cfg.evaluation.save_every_epoch:
                        save_path = saver.save(sess, os.path.join(cfg.logging.logdir, "model"), global_step=epoch)
                        logger.info("Model saved in file: %s" % save_path)

                now = time.time()
                time_elapsed = now - start
                time_elapsed_str = str(datetime.timedelta(seconds=time_elapsed))
                time_remaining = time_elapsed / (epoch + 1) * (cfg.training.num_epochs - epoch - 1)
                time_remaining_str = str(datetime.timedelta(seconds=time_remaining))
                logger.info(f'Finished epoch {epoch}. Time elapsed: {time_elapsed_str}, Time remaining: {time_remaining_str}')
            logger.info('Finished Training')
        except KeyboardInterrupt:
            logger.info('Interrupted')


def train_one_epoch(sess, ops, train_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    batch_size = cfg.training.batch_size

    train_idxs = copy.deepcopy(TRAIN_INDICES)
    np.random.shuffle(train_idxs)
    num_batches = len(train_idxs) // batch_size

    loss_sum = 0

    pbar = tqdm(range(num_batches), desc=f'train', postfix=dict(last_loss_str=''))
    for batch_idx in pbar:
        #  logger.info('----- batch ' + str(batch_idx) + ' -----')
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        pcs1, pcs2, translations, rel_angles, pc1centers, pc2centers, pc1angles, pc2angles = provider.load_batch(train_idxs[start_idx:end_idx])

        # Augment batched point clouds by jittering
        pcs1 = provider.jitter_point_cloud(pcs1)
        pcs2 = provider.jitter_point_cloud(pcs2)
        feed_dict = {
            ops['pcs1']: pcs1,
            ops['pcs2']: pcs2,
            ops['translations']: translations,
            ops['rel_angles']: rel_angles,
            ops['is_training_pl']: is_training,
            ops['pc1centers']: pc1centers,
            ops['pc2centers']: pc2centers,
            ops['pc1angles']: pc1angles,
            ops['pc2angles']: pc2angles,
        }
        summary, step, _, loss_val, pred_translations, pred_remaining_angle_logits = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred_translations'], ops['pred_remaining_angle_logits']], feed_dict=feed_dict)
        #  step_in_epochs = float(epoch) + float(end_idx / len(train_idxs))
        train_writer.add_summary(summary, step)

        #  pred_val = np.argmax(pred_val, 1)
        #  correct = np.sum(pred_val == current_label[start_idx:end_idx])
        #  total_correct += correct
        #  total_seen += cfg.training.batch_size
        loss_sum += loss_val
        pbar.set_postfix(last_loss_str=f'{loss_val:.5f}')
        #  if batch_idx == 0:
        #  logger.info(np.concatenate([pred_val, transforms], axis=1)[:5,:])

    logger.info('train mean loss: %f' % (loss_sum / float(num_batches)))
    #  logger.info('accuracy: %f' % (total_correct / float(total_seen)))
    train_writer.flush()


def eval_one_epoch(sess, ops, val_writer, val_writer_180, epoch, eval_only, do_timings, override_batch_size=None):
    """ ops: dict mapping from string to tf ops """

    is_training = False
    batch_size = cfg.training.batch_size if override_batch_size is None else override_batch_size

    val_idxs = VAL_INDICES
    num_batches = int(np.ceil(len(val_idxs) / batch_size))
    num_full_batches = int(np.floor(len(val_idxs) / batch_size))

    loss_sum = 0
    global_step = sess.run([ops['step']])[0]
    #  step_in_epochs = epoch + 1
    eval_dir = f'{cfg.logging.logdir}/val/eval{str(epoch).zfill(6)}'
    base_eval_dir = eval_dir
    if FLAGS.refineICP:
        eval_dir = f'{eval_dir}/refined_{FLAGS.refineICPmethod}{"_"+FLAGS.its if FLAGS.its != 30 else ""}'

    if os.path.isdir(eval_dir):
        os.rename(eval_dir, f'{eval_dir}_backup_{int(time.time())}')

    os.makedirs(eval_dir, exist_ok=True)

    all_pred_translations = np.empty((len(val_idxs), 3), dtype=np.float32)
    all_pred_angles = np.empty((len(val_idxs), 1), dtype=np.float32)

    all_pred_s1_pc1centers = np.empty((len(val_idxs), 3), dtype=np.float32)
    all_pred_s1_pc2centers = np.empty((len(val_idxs), 3), dtype=np.float32)
    all_pred_s2_pc1centers = np.empty((len(val_idxs), 3), dtype=np.float32)
    all_pred_s2_pc2centers = np.empty((len(val_idxs), 3), dtype=np.float32)
    all_pred_s2_pc1angles = np.empty((len(val_idxs), 1), dtype=np.float32)
    all_pred_s2_pc2angles = np.empty((len(val_idxs), 1), dtype=np.float32)

    all_gt_translations = np.empty((len(val_idxs), 3), dtype=np.float32)
    all_gt_angles = np.empty((len(val_idxs), 1), dtype=np.float32)
    all_gt_pc1centers = np.empty((len(val_idxs), 3), dtype=np.float32)

    if FLAGS.use_old_results:
        all_pred_translations = np.load(f'{base_eval_dir}/pred_translations.npy')
        all_pred_angles = np.load(f'{base_eval_dir}/pred_angles.npy')
        all_pred_s2_pc1centers = np.load(f'{base_eval_dir}/pred_s2_pc1centers.npy')

    cumulated_times = 0.
    for batch_idx in tqdm(range(num_batches), desc='val'):
        #  logger.info('----- batch ' + str(batch_idx) + ' -----')
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(val_idxs))

        pcs1, pcs2, translations, rel_angles, pc1centers, pc2centers, pc1angles, pc2angles = provider.load_batch(val_idxs[start_idx:end_idx], override_batch_size=override_batch_size)

        feed_dict = {
            ops['pcs1']: pcs1,
            ops['pcs2']: pcs2,
            ops['translations']: translations,
            ops['rel_angles']: rel_angles,
            ops['is_training_pl']: is_training,
            ops['pc1centers']: pc1centers,
            ops['pc2centers']: pc2centers,
            ops['pc1angles']: pc1angles,
            ops['pc2angles']: pc2angles,
        }
        start = time.time()
        summary, loss_val, pred_translations, pred_pc1angle_logits, pred_pc2angle_logits, pred_remaining_angle_logits, pred_s1_pc1centers, pred_s1_pc2centers, pred_s2_pc1centers, pred_s2_pc2centers = sess.run([ops['merged'], ops['loss'], ops['pred_translations'], ops['pred_pc1angle_logits'], ops['pred_pc2angle_logits'], ops['pred_remaining_angle_logits'], ops['pred_s1_pc1centers'], ops['pred_s1_pc2centers'], ops['pred_s2_pc1centers'], ops['pred_s2_pc2centers']], feed_dict=feed_dict)
        cumulated_times += time.time() - start
        #  val_writer.add_summary(summary, step)
        actual_batch_size = end_idx - start_idx
        pred_translations = pred_translations[:actual_batch_size]
        pred_angles_pc1 = MODEL.classLogits2angle(pred_pc1angle_logits[:actual_batch_size])
        pred_angles_pc2 = MODEL.classLogits2angle(pred_pc2angle_logits[:actual_batch_size])
        pred_angles_remaining = MODEL.classLogits2angle(pred_remaining_angle_logits[:actual_batch_size])
        pred_angles = pred_angles_pc2 - pred_angles_pc1 + pred_angles_remaining

        if actual_batch_size == batch_size:  # last batch is not counted
            loss_sum += loss_val

        for idx in range(actual_batch_size):
            global_idx = start_idx + idx
            if eval_only and FLAGS.refineICP:
                if FLAGS.use_old_results:
                    init = get_mat_angle(all_pred_translations[global_idx], all_pred_angles[global_idx], rotation_center=all_pred_s2_pc1centers[global_idx])
                else:
                    init = get_mat_angle(pred_translations[idx], pred_angles[idx], rotation_center=pred_s2_pc1centers[idx])
                # Careful: Pass full point cloud, not subsampled one
                refined_pred_transform, refined_pred_center, time_elapsed = icp.icp_p2point(val_idxs[global_idx], cfg, with_constraint=True, radius=0.1, init=init, its=int(FLAGS.its))
                cumulated_times += time_elapsed
                #  if refined_pred_transform[2, 2] == -1:
                #  refined_pred_transform = init
                #  Overwrite predicted translation and angle in place, then update all_pred_... later
                pred_translations[idx] = refined_pred_transform[:3, 3]
                rotation_mat = refined_pred_transform[:3, :3]
                rot_vec = Rotation.from_dcm(rotation_mat).as_euler('xyz')
                #  if global_idx == 47:
                #  print(global_idx, '   ', evaluation.eval_angle(rot_vec[2], all_pred_angles[global_idx], True)[0])
                #  print(refined_pred_transform)
                #  print(init)
                #  print(rot_vec)
                pred_angles[idx] = rot_vec[2]
                #  The transformation ICP outputs is in world space, thus with a rotation around 0,0,0. Store this so that a comparable translation and rotation can be computed later
                pred_s2_pc1centers[idx] = [0., 0, 0]

            all_pred_translations[global_idx] = pred_translations[idx]
            all_pred_angles[global_idx] = pred_angles[idx]

            all_pred_s1_pc1centers[global_idx] = pred_s1_pc1centers[idx]
            all_pred_s1_pc2centers[global_idx] = pred_s1_pc2centers[idx]
            all_pred_s2_pc1centers[global_idx] = pred_s2_pc1centers[idx]
            all_pred_s2_pc2centers[global_idx] = pred_s2_pc2centers[idx]

            all_pred_s2_pc1angles[global_idx] = pred_angles_pc1[idx]
            all_pred_s2_pc2angles[global_idx] = pred_angles_pc2[idx]

            all_gt_translations[global_idx] = translations[idx]
            all_gt_angles[global_idx] = rel_angles[idx]
            all_gt_pc1centers[global_idx] = pc1centers[idx]

    mean_per_transform_loss = loss_sum / num_full_batches if num_full_batches > 0 else 0.
    mean_execution_time = cumulated_times / float(len(val_idxs))

    if do_timings:
        print(f'Timing bs={override_batch_size}: {mean_execution_time}')
    elif cfg.evaluation.has('special') and cfg.evaluation.special.mode == 'held':
        #  print(all_pred_translations)
        _, eval_dict = evaluation.evaluate_held(cfg, val_idxs, all_pred_translations, all_pred_angles, all_gt_translations, all_gt_angles, eval_dir=eval_dir, mean_time=mean_execution_time)
    else:
        for accept_inverted_angle, _val_writer in zip([False, True], [val_writer, val_writer_180]):
            eval_dict = evaluation.evaluate(cfg, val_idxs, all_pred_translations, all_pred_angles, all_gt_translations, all_gt_angles, all_pred_s2_pc1centers, all_gt_pc1centers, eval_dir=eval_dir, accept_inverted_angle=accept_inverted_angle, mean_time=mean_execution_time)
            corr_levels_translation_str = ' '.join([f'{a*100.0:.2f}%' for a in eval_dict.corr_levels_translation])
            corr_levels_angles_str = ' '.join([f'{a*100.0:.2f}%' for a in eval_dict.corr_levels_angles])
            corr_levels_str = ' '.join([f'{a*100.0:.2f}%' for a in eval_dict.corr_levels])
            logger.info(f'Mean translation distance: {eval_dict.mean_dist_translation}, Mean angle distance: {eval_dict.mean_dist_angle}, Levels: {corr_levels_str}, Translation levels: {corr_levels_translation_str}, Angle levels: {corr_levels_angles_str}, Fitness: {eval_dict.reg_eval.fitness*100.0:.2f}%, Inlier RMSE: {eval_dict.reg_eval.inlier_rmse*100.0:.2f}%, Mean ex. time: {mean_execution_time:.5f}')

            if not eval_only:
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='losses/loss', simple_value=mean_per_transform_loss)]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/t_a_mean_dist', simple_value=eval_dict.mean_dist_translation)]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/t_b_1cm', simple_value=eval_dict.corr_levels_translation[0])]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/t_c_10cm', simple_value=eval_dict.corr_levels_translation[1])]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/t_d_1m', simple_value=eval_dict.corr_levels_translation[2])]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/a_a_mean_dist', simple_value=eval_dict.mean_dist_angle)]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/a_b_1d', simple_value=eval_dict.corr_levels_angles[0])]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/a_c_5d', simple_value=eval_dict.corr_levels_angles[1])]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/a_d_10d', simple_value=eval_dict.corr_levels_angles[2])]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/o_b_1cm', simple_value=eval_dict.corr_levels[0])]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/o_c_10cm', simple_value=eval_dict.corr_levels[1])]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/o_d_1m', simple_value=eval_dict.corr_levels[2])]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/fitness', simple_value=eval_dict.reg_eval.fitness)]), global_step=global_step)
                _val_writer.add_summary(summary=tf.Summary(value=[tf.summary.Summary.Value(tag='accuracy/inlier_rmse', simple_value=eval_dict.reg_eval.inlier_rmse)]), global_step=global_step)
                _val_writer.flush()

    np.save(f'{eval_dir}/pred_translations.npy', all_pred_translations)
    np.save(f'{eval_dir}/pred_angles.npy', all_pred_angles)

    np.save(f'{eval_dir}/pred_s1_pc2centers.npy', all_pred_s1_pc2centers)
    if True or not eval_only:
        np.save(f'{eval_dir}/pred_s1_pc1centers.npy', all_pred_s1_pc1centers)
        np.save(f'{eval_dir}/pred_s2_pc1centers.npy', all_pred_s2_pc1centers)
        np.save(f'{eval_dir}/pred_s2_pc2centers.npy', all_pred_s2_pc2centers)
        np.save(f'{eval_dir}/pred_s2_pc1angles.npy', all_pred_s2_pc1angles)
        np.save(f'{eval_dir}/pred_s2_pc2angles.npy', all_pred_s2_pc2angles)

    logger.info('val mean loss: %f' % (mean_per_transform_loss))


if __name__ == "__main__":
    if cfg.evaluation.has('special'):
        if cfg.evaluation.special.mode == 'icp':
            print(FLAGS.config)
            icp.evaluate(cfg, FLAGS.use_old_results)
        elif cfg.evaluation.special.mode == 'held':
            train(eval_only=True, eval_epoch=FLAGS.eval_epoch, eval_only_model_to_load=cfg.evaluation.special.held.model)
        elif cfg.evaluation.special.mode == 'timings':
            #  for bs in reversed([1, 2, 4, 8, 16, 32, 64, 128, 256]):
            for bs in [32]:
                cfg.training.batch_size = bs
                train(eval_only=True, eval_epoch=FLAGS.eval_epoch, do_timings=True, override_batch_size=bs)
        else:
            assert False
    else:
        if FLAGS.operation == 'train':
            train()
        if FLAGS.operation == 'eval_only':
            train(eval_only=True, eval_epoch=FLAGS.eval_epoch)
