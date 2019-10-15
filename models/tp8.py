import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'utils'))
import tf_util
import tf_util_dgcnn
from config import configGlobal as cfg


def placeholder_inputs(batch_size, num_point):
    pcs1 = tf.placeholder(tf.float32, shape=(batch_size, num_point, cfg.data.num_channels))
    pcs2 = tf.placeholder(tf.float32, shape=(batch_size, num_point, cfg.data.num_channels))
    translations = tf.placeholder(tf.float32, shape=(batch_size, 3))
    rel_angles = tf.placeholder(tf.float32, shape=(batch_size, 1))

    pc1_centers = tf.placeholder(tf.float32, shape=(batch_size, 3))
    pc2_centers = tf.placeholder(tf.float32, shape=(batch_size, 3))
    pc1_angles = tf.placeholder(tf.float32, shape=(batch_size, 1))
    pc2_angles = tf.placeholder(tf.float32, shape=(batch_size, 1))
    return pcs1, pcs2, translations, rel_angles, pc1_centers, pc2_centers, pc1_angles, pc2_angles


def tf_get_rotation_matrix_z(a):
    return tf.reshape(tf.stack([tf.cos(a), -tf.sin(a), 0, tf.sin(a), tf.cos(a), 0, 0, 0, 1]), [3, 3])


def _get_dgcnn(pcs, layer_sizes, scope_name, is_training, bn_decay):
    assert len(layer_sizes) > 0
    num_point = pcs.shape[1]
    k = 20
    with tf.variable_scope(scope_name):
        adj_matrix = tf_util_dgcnn.pairwise_distance(pcs)
        nn_idx = tf_util_dgcnn.knn(adj_matrix, k=k)
        edge_feature = tf_util_dgcnn.get_edge_feature(pcs, nn_idx=nn_idx, k=k)

        net = tf_util_dgcnn.conv2d(edge_feature, layer_sizes[0], [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
        for idx, layer_size in enumerate(layer_sizes[1:-1]):
            net = tf_util_dgcnn.conv2d(net, layer_size, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training, scope=f'conv{idx+2}', bn_decay=bn_decay)
        net = tf.reduce_max(net, axis=-2, keepdims=True)
        net = tf_util_dgcnn.conv2d(net, layer_sizes[-1], [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training, scope=f'conv{len(layer_sizes)}', bn_decay=bn_decay)

        net = tf_util_dgcnn.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')
        return net


def _get_pointnet(pcs_extended, layer_sizes, scope_name, is_training, bn_decay):
    assert len(layer_sizes) > 0
    num_point = pcs_extended.shape[1]
    num_channel = pcs_extended.shape[2]
    with tf.variable_scope(scope_name):
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(pcs_extended, layer_sizes[0], [1, num_channel], padding='VALID', stride=[1, 1], bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
        for idx, layer_size in enumerate(layer_sizes[1:]):
            net = tf_util.conv2d(net, layer_size, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training, scope=f'conv{idx+2}', bn_decay=bn_decay)
        net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')
        return net


def get_backbone(net, embedding_layer_sizes, scope_name, is_training, bn_decay, backbone):
    if backbone == 'pointnet':
        return _get_pointnet(net, embedding_layer_sizes, 'embedding', is_training, bn_decay)
    elif backbone == 'dgcnn':
        return _get_dgcnn(net, embedding_layer_sizes, 'embedding', is_training, bn_decay)
    else:
        assert False


def get_backbone_with_options(net, options, scope_name, is_training, bn_decay, backbone):
    return get_backbone(net, options, scope_name, is_training, bn_decay, backbone)


def get_mlp(net, layer_sizes, scope_name, is_training, bn_decay, dropout=None):
    assert len(layer_sizes) > 0
    with tf.variable_scope(scope_name):
        for idx, layer_size in enumerate(layer_sizes[:-1]):
            net = tf_util.fully_connected(net, layer_size, bn=True, is_training=is_training, scope=f'fc{idx+1}', bn_decay=bn_decay)
        if dropout is not None:
            net = tf_util.dropout(net, keep_prob=dropout, is_training=is_training, scope='dp1')
        return tf_util.fully_connected(net, layer_sizes[-1], activation_fn=None, scope=f'fc{len(layer_sizes)}')


def get_mlp_with_options(net, options, scope_name, is_training, bn_decay):
    return get_mlp(net, options[0], scope_name, is_training, bn_decay, options[1])


def get_transformer_net(pcs, embedding_layer_sizes, mlp_layer_sizes, scope_name, is_training, bn_decay, backbone, dropout=None):
    batch_size = pcs.shape[0]
    with tf.variable_scope(scope_name):
        net = get_backbone(pcs, embedding_layer_sizes, 'embedding', is_training, bn_decay, backbone)
        net = tf.reshape(net, [batch_size, -1])
        return get_mlp(net, mlp_layer_sizes, 'mlp', is_training, bn_decay, dropout)


def get_transformer_net_from_options(pcs, options, scope_name, is_training, bn_decay, backbone, with_angles):
    return get_transformer_net(pcs, options[0], options[1][0] + [3 + (cfg.model.angles.num_bins * 2 if with_angles else 0)], scope_name, is_training, bn_decay, backbone, options[1][1])


def get_embedding_net(pcs, is_training, end_points, bn_decay=None):
    num_point = pcs.get_shape()[1].value

    center_mean = tf.reduce_mean(pcs, axis=1)
    pcs_extended = tf.expand_dims(pcs, -1)
    pcs_mean_centered = pcs_extended - tf.tile(tf.expand_dims(tf.expand_dims(center_mean, -1), 1), [1, num_point, 1, 1])

    s1_pred_center = get_transformer_net_from_options(pcs_mean_centered, cfg.model.options.s1transformer, 'transformer1', is_training, bn_decay, cfg.model.backbone, with_angles=False)
    s1_pred_center = s1_pred_center + center_mean  # Make it an absolute center prediction

    # S1 normalization
    # pcs_extended is e.g. [64, 1024, 3, 1], pred_center_s1 is [64, 3]
    pcs_centered_s1 = pcs_extended - tf.tile(tf.expand_dims(tf.expand_dims(s1_pred_center, -1), 1), [1, num_point, 1, 1])

    s2_output = get_transformer_net_from_options(pcs_centered_s1, cfg.model.options.s2transformer, 'transformer2', is_training, bn_decay, cfg.model.backbone, with_angles=True)

    s2_pred_center = s2_output[:, :3] + s1_pred_center
    s2_pred_angle_logits = s2_output[:, 3:]

    # S2 normalization
    # pcs_extended is e.g. [64, 1024, 3, 1], pred_center_s1 is [64, 3]
    pcs_centered_s2 = pcs_extended - tf.tile(tf.expand_dims(tf.expand_dims(s2_pred_center, -1), 1), [1, num_point, 1, 1])
    s2_pred_angles = tf_get_angles(s2_pred_angle_logits)

    rotation_mats = tf.stack(tf.map_fn(lambda angle: tf_get_rotation_matrix_z(-angle), s2_pred_angles))

    pcs_normalized_s2 = tf.matmul(pcs_centered_s2[:, :, :, 0], rotation_mats)
    pcs_normalized_s2 = tf.expand_dims(pcs_normalized_s2, -1)

    embedding_output = get_backbone_with_options(pcs_normalized_s2, cfg.model.options.embedding, 'final_embedding', is_training, bn_decay, backbone=cfg.model.backbone)

    return embedding_output, center_mean, s1_pred_center, s2_pred_center, s2_pred_angle_logits


def get_model(pcs1, pcs2, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = pcs1.get_shape()[0].value
    end_points = {}

    with tf.variable_scope("siamese"):
        embedding_output1, center_mean1, s1_pred_center1, s2_pred_center1, s2_pred_angle_logits1 = get_embedding_net(pcs1, is_training, end_points, bn_decay)
    with tf.variable_scope("siamese", reuse=tf.AUTO_REUSE):
        embedding_output2, center_mean2, s1_pred_center2, s2_pred_center2, s2_pred_angle_logits2 = get_embedding_net(pcs2, is_training, end_points, bn_decay)
    embedding_output_combined = tf.concat([embedding_output1, embedding_output2], axis=3)

    end_points['pred_s1_pc1centers'] = s1_pred_center1
    end_points['pred_s1_pc2centers'] = s1_pred_center2
    end_points['pred_s2_pc1centers'] = s2_pred_center1
    end_points['pred_s2_pc2centers'] = s2_pred_center2
    end_points['pred_pc1angle_logits'] = s2_pred_angle_logits1
    end_points['pred_pc2angle_logits'] = s2_pred_angle_logits2

    net = tf.reshape(embedding_output_combined, [batch_size, -1])
    net = get_mlp(net, [*cfg.model.options.remaining_transform_prediction[0], 3 + cfg.model.angles.num_bins * 2], '', is_training, bn_decay, dropout=cfg.model.options.remaining_transform_prediction[1])
    end_points['pred_translations'] = net[:, :3] + (s2_pred_center2 - s2_pred_center1)
    end_points['pred_remaining_angle_logits'] = net[:, 3:]

    return end_points


def tf_get_angle_difference(a1, a2):
    pi = tf.constant(np.pi)
    r = tf.mod(a2 - a1, 2.0 * pi)
    return tf.where(r > pi, r - pi * 2.0, r)


# For axis and angle: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/
def tf_quaternion_to_angle(q):
    return tf.acos(q[3]) * 2.0


#  From frustum pointnets
def huber_loss(error, delta, name=None):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses, name=name)


def tf_angle2class(angle):
    ''' Convert continuous angle to discrete class and residual.
        num_class: int scalar, number of classes N

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    twopi = tf.constant(2.0 * np.pi)
    angle = tf.mod(angle, twopi)
    angle_per_class = twopi / tf.to_float(cfg.model.angles.num_bins)
    shifted_angle = tf.mod(angle + angle_per_class / 2.0, twopi)
    class_id = tf.to_int32(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (tf.to_float(class_id) * angle_per_class + angle_per_class / 2.0)
    return class_id[:, 0], residual_angle


def tf_class2angle(pred_cls, residual, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    tf_pi = tf.constant(np.pi, dtype=tf.float32)
    angle_per_class = 2.0 * tf_pi / tf.to_float(cfg.model.angles.num_bins)
    angle_center = tf.to_float(pred_cls) * angle_per_class
    angle = angle_center + residual
    if to_label_format:
        angle = tf.mod(angle + tf_pi, 2.0 * tf_pi) - tf_pi  # Transfer to range -pi,pi
    return angle


def tf_class2angle2(pred_cls, residuals, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    residual = tf.gather_nd(residuals, tf.transpose(tf.stack([tf.range(pred_cls.shape[0], dtype=tf.int64), pred_cls])))
    tf_pi = tf.constant(np.pi, dtype=tf.float32)
    angle_per_class = 2.0 * tf_pi / tf.to_float(cfg.model.angles.num_bins)
    angle_center = tf.to_float(pred_cls) * angle_per_class
    angle = angle_center + residual
    if to_label_format:
        angle = tf.mod(angle + tf_pi, 2.0 * tf_pi) - tf_pi  # Transfer to range -pi,pi
    return angle


def class2angle(pred_cls, residual, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(cfg.model.angles.num_bins)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def classLogits2angle(logits, to_label_format=True):
    class_logits, residuals = logits[:, :cfg.model.angles.num_bins], logits[:, cfg.model.angles.num_bins:]
    classes = np.argmax(class_logits, axis=1)
    return np.array([class2angle(_class, _residuals[_class]) for _class, _residuals in zip(classes, residuals)])


def tf_classLogits2angle(logits, to_label_format=True):
    class_logits, residuals = logits[:, :cfg.model.angles.num_bins], logits[:, cfg.model.angles.num_bins:]
    classes = tf.argmax(class_logits, axis=1)
    return tf_class2angle2(classes, residuals)


def tf_get_target_angle_distribution(target_angle):
    target_angle = target_angle[0]
    nbins = cfg.model.angles.num_bins
    angle_per_bin = 360. / nbins
    sigma_in_degree = cfg.training.loss.options.soft_angle_classes_sigma_in_degree
    dist = tf.distributions.Normal(loc=[target_angle - 360., target_angle, target_angle + 360.], scale=sigma_in_degree)  # Create three distributions (non-overlapping, as sigma_in_degree should be small), which are later stitched together so that a target value around 360. degree also raises the probability for angles around 0.
    angles = tf.multiply(tf.cast(tf.range(nbins + 1), tf.float32), tf.to_float(angle_per_bin))
    dist_values = dist.cdf(tf.tile(tf.expand_dims(angles, 1), [1, 3]))  # Here the probabilities at the angle bins are evaluated, with cdf, so that in the end probabilities add up to 1.0
    dist_values = tf.manip.roll(dist_values, -1, axis=0) - dist_values  # Subtract each value from the one on its left -> cumulative probabilities to probabilities
    dist_values = tf.reduce_sum(dist_values, axis=1)  # Combine the three shifted probability distributions
    return dist_values[:-1]  # Last entry was artificially added, and is not considered


def _tf_get_angle_loss(logits, target_angles):
    angle_class_logits = logits[:, :cfg.model.angles.num_bins]
    angle_residuals_normalized = logits[:, cfg.model.angles.num_bins:]

    target_angle_classes, target_angle_residuals = tf_angle2class(target_angles)

    if cfg.training.loss.options.soft_angle_classes:
        angle_class_distributions = tf.map_fn(lambda target_angle: tf_get_target_angle_distribution(target_angle), target_angles)
        angle_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=angle_class_logits, labels=angle_class_distributions))
    else:
        angle_class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=angle_class_logits, labels=target_angle_classes))
    angle_class_onehot = tf.one_hot(target_angle_classes, depth=cfg.model.angles.num_bins, on_value=1, off_value=0, axis=-1)  # BxNUM_HEADING_BIN
    angle_residual_normalized_label = target_angle_residuals / (np.pi / cfg.model.angles.num_bins)
    angle_residual_normalized_loss = huber_loss(tf.reduce_sum(angle_residuals_normalized * tf.to_float(angle_class_onehot), axis=1) - angle_residual_normalized_label, delta=1.0, name='angle_residual_normalized_loss')

    return tf.stack([angle_class_loss + 20.0 * angle_residual_normalized_loss, angle_class_loss, angle_residual_normalized_loss])


def tf_get_angle_losses(logits, target_angles, accept_inverted_angle=False):
    if accept_inverted_angle:
        angle_losses = _tf_get_angle_loss(logits, target_angles)
        angle_losses_180 = _tf_get_angle_loss(logits, target_angles + np.pi)
        angle_losses = tf.cond(angle_losses[0] > angle_losses_180[0], lambda: angle_losses, lambda: angle_losses_180)
    else:
        angle_losses = _tf_get_angle_loss(logits, target_angles)
    return angle_losses[0], angle_losses[1], angle_losses[2]


def tf_get_angles(logits):
    angle_class_logits = logits[:, :cfg.model.angles.num_bins]
    angle_classes = tf.argmax(angle_class_logits, axis=1, output_type=tf.int32)
    angle_residuals_normalized = logits[:, cfg.model.angles.num_bins:]
    angle_residuals = angle_residuals_normalized * (tf.constant(np.pi) / cfg.model.angles.num_bins)
    angle_residual_indices = tf.transpose(tf.stack([tf.range(logits.shape[0]), angle_classes]))
    angle_perclass_residuals = tf.gather_nd(angle_residuals, angle_residual_indices)
    return tf_class2angle(angle_classes, angle_perclass_residuals)


def _get_loss_separate(pcs1, pcs2, translations, rel_angles, pc1_centers, pc2_centers, pc1_angles, pc2_angles, end_points):
    batch_size = translations.get_shape()[0].value

    angle_factor = cfg.model.options.angle_factor
    early_stage_factor = cfg.model.options.early_stage_factor

    pc1_angle_classes, pc1_angle_residuals = tf_angle2class(pc1_angles)

    pc1_stage1_transl_loss = huber_loss(end_points['pred_s1_pc1centers'] - pc1_centers, delta=1.0)
    pc2_stage1_transl_loss = huber_loss(end_points['pred_s1_pc2centers'] - pc2_centers, delta=1.0)
    stage1_translation_losses = (pc1_stage1_transl_loss + pc2_stage1_transl_loss) / 2.0

    pc1_stage2_transl_loss = huber_loss(end_points['pred_s2_pc1centers'] - pc1_centers, delta=1.0)
    pc2_stage2_transl_loss = huber_loss(end_points['pred_s2_pc2centers'] - pc2_centers, delta=1.0)
    pc1_stage2_angle_loss, pc1_stage2_angle_class_loss, pc1_stage2_angle_residual_loss = tf_get_angle_losses(end_points['pred_pc1angle_logits'], pc1_angles, cfg.model.angles.accept_inverted_angle)
    pc2_stage2_angle_loss, pc2_stage2_angle_class_loss, pc2_stage2_angle_residual_loss = tf_get_angle_losses(end_points['pred_pc2angle_logits'], pc2_angles, cfg.model.angles.accept_inverted_angle)
    stage2_translation_losses = (pc1_stage2_transl_loss + pc2_stage2_transl_loss) / 2.0
    stage2_angle_losses = (pc1_stage2_angle_loss + pc2_stage2_angle_loss) / 2.0

    stage3_translation_loss = huber_loss(end_points['pred_translations'] - translations, delta=2.0)

    pc1_pred_angles = tf_get_angles(end_points['pred_pc1angle_logits'])
    pc2_pred_angles = tf_get_angles(end_points['pred_pc2angle_logits'])
    remaining_diff_target_angle = (pc2_angles - pc1_angles) - (pc2_pred_angles - pc1_pred_angles)
    stage3_angle_loss, stage3_angle_class_loss, stage3_angle_residual_loss = tf_get_angle_losses(end_points['pred_remaining_angle_logits'], remaining_diff_target_angle, cfg.model.angles.accept_inverted_angle)

    #  loss = early_stage_factor * (stage1_translation_losses) + stage3_losses
    loss_translation = early_stage_factor * (stage1_translation_losses + stage2_translation_losses) + stage3_translation_loss
    loss_angle = early_stage_factor * (stage2_angle_losses) + stage3_angle_loss
    loss = loss_translation + angle_factor * loss_angle
    per_transform_loss = loss / batch_size

    tf.summary.scalar('losses/translation', loss_translation)
    tf.summary.scalar('losses/angle', loss_angle)

    tf.summary.scalar('losses_stages/stage1_pc1_transl_loss', pc1_stage1_transl_loss)
    tf.summary.scalar('losses_stages/stage1_pc2_transl_loss', pc2_stage1_transl_loss)
    tf.summary.scalar('losses_stages/stage2_pc1_transl_loss', pc1_stage2_transl_loss)
    tf.summary.scalar('losses_stages/stage2_pc2_transl_loss', pc2_stage2_transl_loss)
    tf.summary.scalar('losses_stages/stage3_transl_loss', stage3_translation_loss)

    tf.summary.scalar('losses_stages/stage2_pc1_angle_loss', pc1_stage2_angle_loss)
    tf.summary.scalar('losses_stages/stage2_pc1_angle_class_loss', pc1_stage2_angle_class_loss)
    tf.summary.scalar('losses_stages/stage2_pc1_angle_residual_loss', pc1_stage2_angle_residual_loss)
    tf.summary.scalar('losses_stages/stage2_pc2_angle_loss', pc2_stage2_angle_loss)
    tf.summary.scalar('losses_stages/stage2_pc2_angle_class_loss', pc2_stage2_angle_class_loss)
    tf.summary.scalar('losses_stages/stage2_pc2_angle_residual_loss', pc2_stage2_angle_residual_loss)
    tf.summary.scalar('losses_stages/stage3_angle_loss', stage3_angle_loss)
    tf.summary.scalar('losses_stages/stage3_angle_class_loss', stage3_angle_class_loss)
    tf.summary.scalar('losses_stages/stage3_angle_residual_loss', stage3_angle_residual_loss)
    return per_transform_loss


def tf_translate_pcs(pcs, translation):
    return tf.tile(tf.expand_dims(translation, 1), [1, pcs.shape[1], 1])


def tf_transform_pcs(pcs, translations=None, angles=None, rotation_centers=None):
    if rotation_centers is not None:
        pcs = tf_translate_pcs(pcs, -rotation_centers)
    if angles is not None:
        rotation_mats = tf.map_fn(lambda angle: tf_get_rotation_matrix_z(angle), angles)
        pcs = tf.matmul(pcs, rotation_mats)
    if translations is not None:
        pcs = tf_translate_pcs(pcs, -translations)
    if rotation_centers is not None:
        pcs = tf_translate_pcs(pcs, rotation_centers)
    return pcs


def _get_loss_p2p(pcs1, pcs2, translations, rel_angles, pc1_centers, pc2_centers, pc1_angles, pc2_angles, end_points):
    batch_size = translations.get_shape()[0].value
    pred_translations = end_points['pred_translations']
    pred_s2_pc1centers = end_points['pred_s2_pc1centers']
    pred_angles_pc1 = tf_classLogits2angle(end_points['pred_pc1angle_logits'])
    pred_angles_pc2 = tf_classLogits2angle(end_points['pred_pc2angle_logits'])
    pred_angles_remaining = tf_classLogits2angle(end_points['pred_remaining_angle_logits'])
    pred_angles = pred_angles_pc2 - pred_angles_pc1 + pred_angles_remaining

    pcs1_transformed = tf_transform_pcs(pcs1, pred_translations, pred_angles, pred_s2_pc1centers)
    pcs1_transformed_gt = tf_transform_pcs(pcs1, translations, rel_angles[:, 0], pc1_centers)

    point_distances = tf.norm(pcs1_transformed - pcs1_transformed_gt, axis=1)
    loss = tf.reduce_mean(tf.square(point_distances))
    if cfg.model.angles.accept_inverted_angle:
        pcs1_transformed_180 = tf_transform_pcs(pcs1, pred_translations, pred_angles, pred_s2_pc1centers)
        pcs1_transformed_gt_180 = tf_transform_pcs(pcs1, translations, rel_angles[:, 0], pc1_centers)

        point_distances_180 = tf.norm(pcs1_transformed_180 - pcs1_transformed_gt_180, axis=1)
        loss_180 = tf.reduce_mean(tf.square(point_distances_180))
        loss = tf.minimum(loss, loss_180)
    per_transform_loss = loss / batch_size
    #  per_point_loss = per_transform_loss / pcs1.shape[1]

    return per_transform_loss


def get_loss(*args):
    if cfg.training.loss.loss == 'separate':
        return _get_loss_separate(*args)
    elif cfg.training.loss.loss == 'p2p':
        return _get_loss_p2p(*args)
    else:
        assert False


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
