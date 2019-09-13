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
import extra_loss
import pan_util


def placeholder_inputs(batch_size, num_point, normal_flag=False):
	if normal_flag:
		pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
	else:
		pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
	labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
	return pointclouds_pl, labels_pl

 
def get_model(point_input, is_training, pfs_flag=False, bn_decay=None):
	""" Classification PointNet, input is BxNxC, output Bx40 """
	batch_size = point_input.get_shape()[0].value
	num_point1 = point_input.get_shape()[1].value
	num_point2 = int(np.floor(num_point1 / 4.0))
	num_point3 = int(np.floor(num_point2 / 4.0))
	# num_point4 = num_point3 / 4
	num_features = point_input.get_shape()[2].value

	end_points = {}
	k = 10
	pk = 10

	de1_1 = 1
	de1_2 = 2
	de2_1 = 1
	de2_2 = 2
	de3_1 = 1
	de3_2 = 2

	if num_features > 3:
		point_cloud1 = tf.slice(point_input, [0, 0, 0], [-1, -1, 3])
	else:
		point_cloud1 = point_input

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
	p1_2 = 0.8

	p2_1 = 0
	p2_2 = 1.6

	# activation_fn = tf.math.softplus 
	activation_fn = tf.nn.relu

##################################################################################################
	# Hierarchy 1

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
	
	
	net1_1 = tf.squeeze(net1_1)
	net1_2 = tf.squeeze(net1_2)


##################################################################################################
	# Hierarchy 2

	dist_threshold1 = False
	net, p1_idx, pn_idx, point_cloud2 = pan_util.edge_preserve_sampling(net1_2, point_cloud1, num_point2, hie_matrix1, 
		dist_threshold1, pk, 1, p1_1, p1_2, PFS_flag=pfs_flag, atrous_flag=False)
	
	net1_1 = tf.squeeze(tf.reduce_max(group_point(net1_1, pn_idx), axis=-2, keepdims=True))
	net1_2 = tf.squeeze(net)

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


	net2_1 = tf.squeeze(net2_1)
	net2_2 = tf.squeeze(net2_2)


##################################################################################################
	# Hierarchy 3

	dist_threshold2 = False
	net, p2_idx, pn_idx, point_cloud3 = pan_util.edge_preserve_sampling(net2_2, point_cloud2, num_point3, hie_matrix2, 
		dist_threshold2, pk, 1, p2_1, p2_2, PFS_flag=pfs_flag, atrous_flag=False)
	
	net1_1 = tf.reduce_max(group_point(net1_1, pn_idx), axis=-2, keepdims=True)
	net1_2 = tf.reduce_max(group_point(net1_2, pn_idx), axis=-2, keepdims=True)
	net2_1 = tf.reduce_max(group_point(net2_1, pn_idx), axis=-2, keepdims=True)
	net2_2 = net

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

	net = tf.concat([net1_1, net1_2, net2_1, net2_2, net3_1, net3_2], axis=-1)
	# net = tf.concat([net1_2, net2_2, net3_2], axis=-1)
	net = tf_util.conv2d(net, 1024, [1, 1], 
					   padding='VALID', stride=[1,1], activation_fn=activation_fn,
					   bn=True, is_training=is_training,
					   scope='agg', bn_decay=bn_decay)

	net = tf.reduce_max(net, axis=1, keepdims=True) 

	# MLP on global point cloud vector
	net = tf.reshape(net, [batch_size, -1]) 

	end_points['embedding'] = net

	net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
								scope='fc1', bn_decay=bn_decay)
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
	net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
								scope='fc2', bn_decay=bn_decay)
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
	net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

	return net, end_points



def get_loss(pred, label, end_points, mmd_flag):
	""" pred: B*NUM_CLASSES,
	  label: B, """
	labels = tf.one_hot(indices=label, depth=40)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
	classify_loss = tf.reduce_mean(loss)
	tf.summary.scalar('cls loss', classify_loss)

	loss = classify_loss

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
	batch_size = 2
	num_pt = 124
	pos_dim = 3

	input_feed = np.random.rand(batch_size, num_pt, pos_dim)
	label_feed = np.random.rand(batch_size)
	label_feed[label_feed>=0.5] = 1
	label_feed[label_feed<0.5] = 0
	label_feed = label_feed.astype(np.int32)

	# # np.save('./debug/input_feed.npy', input_feed)
	# input_feed = np.load('./debug/input_feed.npy')
	# print input_feed

	with tf.Graph().as_default():
		input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
		pos, ftr = get_model(input_pl, tf.constant(True))
		# loss = get_loss(logits, label_pl, None)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		feed_dict = {input_pl: input_feed, label_pl: label_feed}
		res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
		print (res1.shape)
		print (res1)

		print (res2.shape)
		print (res2)












