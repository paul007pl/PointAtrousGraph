""" PAN Layers

Author: Liang PAN
Date: Aug. 2019

"""


import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point, principal_feature_sample
from tf_grouping import query_ball_point, group_point, knn_point, select_top_k
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util


def point_atrous_conv(feature_input, adj_input, dist_input, knn, atrous, 
	radius_min, radius_max, num_output_channels, scope, kernel_size=[1, 1],
	stride=[1, 1], padding='VALID', use_xavier=True, stddev=1e-3, weight_decay=0.0,
	activation_fn=tf.nn.relu, bn=False, bn_decay=None, 
	is_training=None, is_dist=False):
	'''
	Input:
		feature_input: (batch_size, npoints, 1, num_features)
		adj_input: (batch_size, num_points, num_points)
		dist_input: (batch_size, num_points, num_points)
		knn: int32
		atrous: int32
		radius_min: float32
		radius_max: float32
		num_output_channels: int32
		kernel_size: a list of 2 ints
		scope: string
		stride: a list of 2 ints
		padding: 'SAME' or 'VALID'
		use_xavier: bool, use xavier_initializer if true
		stddev: float, stddev for truncated_normal init
		weight_decay: float
		activation_fn: function
		bn: bool, whether to use batch norm
		bn_decay: float or float tensor variable in [0,1]
		is_training: bool Tensor variable
	Returns:
		net: (batch_size, num_points, 1, num_output_channels)
	'''
	feature_shape = feature_input.get_shape()
	# batch_size = feature_shape[0]
	num_points = int(feature_shape[1])

	edge_input = feature_input

	if num_points > 1:
		k = int(min(knn, num_points/atrous))
		if k > 1:
			nn_idx = tf_util.get_atrous_knn(adj_input, k, atrous, dist_input, radius_min, radius_max)
			edge_input = tf_util.get_edge_feature(feature_input, nn_idx, k)
		# else:
		# 	edge_input = feature_input

	net = tf_util.conv2d(edge_input, num_output_channels, kernel_size=kernel_size,
		padding=padding, stride=stride, use_xavier=use_xavier, stddev=stddev, 
		weight_decay=weight_decay, activation_fn=activation_fn,
		bn=bn, is_training=is_training,
		scope=scope, bn_decay=bn_decay)
	net = tf.reduce_max(net, axis=-2, keepdims=True)

	return net


def edge_preserve_sampling(feature_input, point_input, num_samples, adj_input, dist_threshold, 
	knn, atrous, radius_min, radius_max, PFS_flag=False, atrous_flag=True):
	'''
	Input:
		feature_input: (batch_size, num_points, num_features)
		point_input: (batch_size, num_points, 3)
		num_samples: int32
		adj_input: (batch_size, num_points, num_points)
		dist_threshold: bool
		knn: int32
		atrous: int32
		radius_min: float32
		radius_max: float32
		PFS_flag: bool
	Returns:
		net: (batch_size, num_samples, 1, 2 * num_features)
		p_idx: (batch_size, num_samples)
		pn_idx: (batch_size, num_samples, knn)
		point_output: (batch_size, num_samples, 3)
	'''
	feature_shape = feature_input.get_shape()
	batch_size = feature_shape[0]
	num_points = int(feature_shape[1])

	if PFS_flag == False:
		p_idx = farthest_point_sample(num_samples, point_input)
	else:
		p_idx = tf_util.gather_principal_feature(feature_input, num_samples)

	point_output = gather_point(point_input, p_idx)

	padj_matrix = tf.squeeze(group_point(adj_input, tf.expand_dims(p_idx, axis=-1)))
	if batch_size == 1:
		padj_matrix = tf.expand_dims(padj_matrix, axis=0)

	if num_points == 1:
		padj_matrix = tf.expand_dims(padj_matrix, axis=1)

	if dist_threshold:
		pdist_matrix = padj_matrix
	else:
		pdist_matrix = None

	
	if atrous_flag:
		pk = int(min(knn, num_points/atrous))
		pn_idx = tf_util.get_atrous_knn(padj_matrix, pk, atrous, pdist_matrix, radius_min, radius_max)
	else:
		pk = int(min(knn, num_points))
		_, pn_idx = knn_point(pk, point_input, point_output)

	neighbor_feature = group_point(feature_input, pn_idx)
	neighbor_feature = tf.reduce_max(neighbor_feature, axis=-2, keepdims=True)
	center_feature = group_point(feature_input, tf.expand_dims(p_idx, 2))
	net = tf.concat([center_feature, neighbor_feature], axis=-1)

	return net, p_idx, pn_idx, point_output


def three_nn_upsampling(target_points, source_points):
	'''
	Input:
		target_points: (batch_size, num_tpoints, 3)
		source_points: (batch_size, num_spoints, 3)
	Returns:
		idx: (batch_size, num_tpoints, 3)
		weight: (batch_size, num_tpoints, 3)
	'''

	dist, idx = three_nn(target_points, source_points)
	dist = tf.maximum(dist, 1e-10)
	norm = tf.reduce_sum((1.0/dist), axis=2, keepdims=True)
	norm = tf.tile(norm, [1, 1, 3])
	weight = (1.0 / dist) / norm

	return idx, weight

