import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))

ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate

import tf_util
# from pointnet_util import *
import extra_loss
import pan_util


def placeholder_inputs(batch_size, num_point):
	pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
	labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
	return pointclouds_pl, labels_pl


def get_model(point_input, labels_pl, is_training, bn_decay=None, coarse_flag=0, 
	mmd_flag=0, pfs_flag=False, fully_concate=False):
	""" Seg, input is BxNx9, output BxNx13 """
	batch_size = point_input.get_shape()[0].value
	num_point1 = point_input.get_shape()[1].value
	num_point2 = int(np.floor(num_point1 / 4.0))
	num_point3 = int(np.floor(num_point2 / 4.0))
	# num_point4 = int(np.floor(num_point3 / 4.0))

	end_points = {}

	k = 10
	pk = 10

	de1_1 = 1
	de1_2 = 2
	de2_1 = 1
	de2_2 = 2
	de3_1 = 1
	de3_2 = 2

	r1_11 = 0
	r1_12 = 0.1
	r1_21 = 0
	r1_22 = 0.2

	r2_11 = 0
	r2_12 = 0.4
	r2_21 = 0
	r2_22 = 0.8

	r3_11 = 0
	r3_12 = 1.6
	r3_21 = 0
	r3_22 = 3.2

	p1_1 = 0
	p1_2 = 0.4

	p2_1 = 0
	p2_2 = 1.6

	# activation_fn = tf.math.softplus 
	activation_fn = tf.nn.relu

##################################################################################################
	# Hierarchy 1

	point_cloud1 = tf.slice(point_input, [0, 0, 0], [-1, -1, 3])

	hie_matrix1 = tf.math.maximum(tf.sqrt(tf_util.pairwise_distance(point_cloud1)), 1e-20)
	# dist_matrix1_1 = hie_matrix1
	# dist_matrix1_2 = hie_matrix1
	dist_matrix1_1 = None
	dist_matrix1_2 = None


	adj_matrix = tf_util.pairwise_distance(point_cloud1)
	net1_1 = pan_util.point_atrous_conv(point_input, adj_matrix, dist_matrix1_1, k, de1_1, r1_11, r1_12, 64, 
		scope='page1_1', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)


	adj_matrix = tf_util.pairwise_distance(net1_1)
	net1_2 = pan_util.point_atrous_conv(net1_1, adj_matrix, dist_matrix1_2, k, de1_2, r1_21, r1_22, 64,
		scope='page1_2', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)
	

	net = tf.squeeze(net1_2)


##################################################################################################
	# Hierarchy 2

	dist_threshold1 = False
	net, p1_idx, _, point_cloud2 = pan_util.edge_preserve_sampling(net, point_cloud1, num_point2, hie_matrix1, 
		dist_threshold1, pk, 1, p1_1, p1_2, pfs_flag, atrous_flag=False)
		
	# point_cloud2 = gather_point(point_cloud1, p1_idx)
	hie_matrix2 = tf.math.maximum(tf.sqrt(tf_util.pairwise_distance(point_cloud2)), 1e-20)
	# dist_matrix2_1 = hie_matrix2
	# dist_matrix2_2 = hie_matrix2
	dist_matrix2_1 = None
	dist_matrix2_2 = None


	adj_matrix = tf_util.pairwise_distance(net)
	net2_1 = pan_util.point_atrous_conv(net, adj_matrix, dist_matrix2_1, k, de2_1, r2_11, r2_12, 128,
		scope='page2_1', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)


	adj_matrix = tf_util.pairwise_distance(net2_1)
	net2_2 = pan_util.point_atrous_conv(net2_1, adj_matrix, dist_matrix2_2, k, de2_2, r2_21, r2_22, 128,
		scope='page2_2', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)


	net = tf.squeeze(net2_2)


##################################################################################################
	# Hierarchy 3
	
	dist_threshold2 = False
	net, p2_idx, _, point_cloud3 = pan_util.edge_preserve_sampling(net, point_cloud2, num_point3, hie_matrix2, 
		dist_threshold2, pk, 1, p2_1, p2_2, pfs_flag, atrous_flag=False)
		
	# point_cloud3 = gather_point(point_cloud2, p2_idx)
	hie_matrix3 = tf.math.maximum(tf.sqrt(tf_util.pairwise_distance(point_cloud3)), 1e-20)	
	# dist_matrix3_1 = hie_matrix3
	# dist_matrix3_2 = hie_matrix3
	dist_matrix3_1 = None
	dist_matrix3_2 = None


	adj_matrix = tf_util.pairwise_distance(net)
	net3_1 = pan_util.point_atrous_conv(net, adj_matrix, dist_matrix3_1, k, de3_1, r3_11, r3_12, 256,
		scope='page3_1', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)


	adj_matrix = tf_util.pairwise_distance(net3_1)
	net3_2 = pan_util.point_atrous_conv(net3_1, adj_matrix, dist_matrix3_2, k, de3_2, r3_21, r3_22, 256,
		scope='page3_2', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)


##################################################################################################	
	# Embeded Features

	net = tf_util.conv2d(net3_2, 1024, [1,1],
					   padding='VALID', stride=[1,1], activation_fn=activation_fn,
					   bn=True, is_training=is_training,
					   scope='encoder', bn_decay=bn_decay)

	net = tf.reduce_max(net, axis=1, keepdims=True)
	net = tf.reshape(net, [batch_size, -1])
	net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, activation_fn=None,
								scope='rg1', bn_decay=bn_decay)
	
	if mmd_flag > 0:
		end_points['embedding'] = net
	else:
		end_points['embedding'] = None

	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='rgdp1')
	
	global_feature_size = 1024
	net = tf_util.fully_connected(net, global_feature_size, bn=True, is_training=is_training, activation_fn=activation_fn,
								scope='rg2', bn_decay=bn_decay)
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='rgdp2')
	net = tf.reshape(net, [batch_size, 1, 1, global_feature_size])

	net = tf.tile(net, [1, num_point3, 1, 1])

	net = tf.concat([net, net3_2], axis=-1)

	net = tf_util.conv2d(net, 512, [1,1],
					   padding='VALID', stride=[1,1], activation_fn=activation_fn,
					   bn=True, is_training=is_training,
					   scope='decoder', bn_decay=bn_decay)

	if coarse_flag > 0:
		coarse_net = tf.squeeze(net)
		coarse_net = tf_util.conv1d(coarse_net, 128, 1, padding='VALID', bn=True, activation_fn=activation_fn,
			is_training=is_training, scope='coarse_fc1', bn_decay=bn_decay)
		coarse_net = tf_util.dropout(coarse_net, keep_prob=0.5, is_training=is_training, scope='cdp1')
		coarse_net = tf_util.conv1d(coarse_net, 50, 1, padding='VALID', activation_fn=None, scope='coarse_fc2')

		coarse_labels_pl = tf_util.gather_labels(labels_pl, p1_idx)
		coarse_labels_pl = tf_util.gather_labels(coarse_labels_pl, p2_idx)

		end_points['coarse_pred'] = coarse_net
		end_points['coarse_label'] = coarse_labels_pl

	else:
		end_points['coarse_pred'] = None
		end_points['coarse_label'] = None


##################################################################################################
	# Hierarchy 3

	adj_matrix = tf_util.pairwise_distance(net)
	net3_2 = pan_util.point_atrous_conv(net, adj_matrix, dist_matrix3_2, k, de3_2, r3_21, r3_22, 256,
		scope='pagd3_2', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)

	net = tf.concat([net3_2, net3_1], axis=-1)
	if fully_concate:
		net3_2 = tf.squeeze(net3_2)

	adj_matrix = tf_util.pairwise_distance(net)
	net3_1 = pan_util.point_atrous_conv(net, adj_matrix, dist_matrix3_1, k, de3_1, r3_11, r3_12, 256,
		scope='pagd3_1', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)

	net3_1 = tf.squeeze(net3_1)


##################################################################################################
	# Hierarchy 2

	idx, weight = pan_util.three_nn_upsampling(point_cloud2, point_cloud3)
	net3_1 = three_interpolate(net3_1, idx, weight)
	if fully_concate:
		net3_2 = three_interpolate(net3_2, idx, weight)	

	net = tf.concat([tf.expand_dims(net3_1, 2), net2_2], axis=-1)
	
	adj_matrix = tf_util.pairwise_distance(net)
	net2_2 = pan_util.point_atrous_conv(net, adj_matrix, dist_matrix2_2, k, de2_2, r2_21, r2_22, 128,
		scope='pagd2_2', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)

	net = tf.concat([net2_2, net2_1], axis=-1)
	if fully_concate:
		net2_2 = tf.squeeze(net2_2)

	adj_matrix = tf_util.pairwise_distance(net)
	net2_1 = pan_util.point_atrous_conv(net, adj_matrix, dist_matrix2_1, k, de2_1, r2_11, r2_12, 128,
		scope='pagd2_1', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)

	net2_1 = tf.squeeze(net2_1)


##################################################################################################
	# Hierarchy 1
	
	idx, weight = pan_util.three_nn_upsampling(point_cloud1, point_cloud2)
	net2_1 = three_interpolate(net2_1, idx, weight)
	net3_1 = three_interpolate(net3_1, idx, weight)
	if fully_concate:
		net2_2 = three_interpolate(net2_2, idx, weight)
		net3_2 = three_interpolate(net3_2, idx, weight)

	net = tf.concat([tf.expand_dims(net2_1, 2), net1_2], axis=-1)

	adj_matrix = tf_util.pairwise_distance(net)
	net1_2 = pan_util.point_atrous_conv(net, adj_matrix, dist_matrix1_2, k, de1_2, r1_21, r1_22, 64,
		scope='pagd1_2', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)

	
	net = tf.concat([net1_2, net1_1], axis=-1)

	adj_matrix = tf_util.pairwise_distance(net)
	net1_1 = pan_util.point_atrous_conv(net, adj_matrix, dist_matrix1_1, k, de1_1, r1_11, r1_12, 64,
		scope='pagd1_1', bn=True, bn_decay=bn_decay, is_training=is_training, activation_fn=activation_fn)


##################################################################################################
	# Final Prediction

	if fully_concate:
		net = tf.concat([net1_1, net1_2, tf.expand_dims(net2_1, 2), tf.expand_dims(net2_2, 2), tf.expand_dims(net3_1, 2), tf.expand_dims(net3_2, 2)], axis=-1)
	else:
		net = tf.concat([net1_1, tf.expand_dims(net2_1, 2), tf.expand_dims(net3_1, 2)], axis=-1)

	net = tf.squeeze(net)

	net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, activation_fn=activation_fn,
		is_training=is_training, scope='fc1', bn_decay=bn_decay)
	
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
	net = tf_util.conv1d(net, 13, 1, padding='VALID', activation_fn=None, scope='fc2')
	
	return net, end_points


##***********************************************************************************************


def get_loss(pred, label, end_points, coarse_flag, mmd_flag):
	""" pred: BxNxC,
		label: BxN, """
	seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
	seg_loss = tf.reduce_mean(seg_loss)
	tf.summary.scalar('seg loss', seg_loss)

	loss = seg_loss

	if coarse_flag > 0:
		coarse_seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['coarse_pred'], labels=end_points['coarse_label'])
		coarse_seg_loss = tf.reduce_mean(coarse_seg_loss)
		coarse_seg_loss = coarse_seg_loss * coarse_flag
		tf.summary.scalar('coarse seg loss', coarse_seg_loss)

		loss = loss + coarse_seg_loss

	if mmd_flag > 0:
		batch_size = end_points['embedding'].get_shape()[0].value
		feature_size = end_points['embedding'].get_shape()[1].value

		true_samples = tf.random_normal(tf.stack([batch_size, feature_size]))
		mmd_loss = extra_loss.compute_mmd(end_points['embedding'], true_samples)
		mmd_loss = mmd_loss * mmd_flag
		tf.summary.scalar('mmd loss', mmd_loss)

		loss = loss + mmd_loss

	tf.add_to_collection('losses', loss)
	return loss



if __name__=='__main__':
	with tf.Graph().as_default():
		inputs = tf.zeros((32,2048,6))
		cls_labels = tf.zeros((32),dtype=tf.int32)
		output, ep = get_model(inputs, cls_labels, tf.constant(True))
		print(output)
