import numpy as np
import tensorflow as tf


def compute_kernel(x, y):
	x_size = tf.shape(x)[0]
	y_size = tf.shape(y)[0]
	dim = tf.shape(x)[1]
	tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
	tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
	return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def matrix_mean_wo_diagonal(matrix, num_row, num_col=None):
	if num_col is None:
		mu = (tf.reduce_sum(matrix) - tf.reduce_sum(tf.matrix_diag_part(matrix))) / (num_row * (num_row - 1.0))
	else:
		mu = (tf.reduce_sum(matrix) - tf.reduce_sum(tf.matrix_diag_part(matrix))) \
			 / (num_row * num_col - tf.minimum(num_col, num_row))
	return mu


def get_squared_dist_ref(x, y):
	x_expand = tf.expand_dims(x, axis=2)  # m-by-d-by-1
	x_permute = tf.transpose(x_expand, perm=(2, 1, 0))  # 1-by-d-by-m

	y_expand = tf.expand_dims(y, axis=2)  # m-by-d-by-1
	y_permute = tf.transpose(y_expand, perm=(2, 1, 0))
	
	dxx = x_expand - x_permute  # m-by-d-by-m, the first page is ai - a1
	dist_xx = tf.reduce_sum(tf.multiply(dxx, dxx), axis=1)  # m-by-m, the first column is (ai-a1)^2
	
	dxy = x_expand - y_permute  # m-by-d-by-m, the first page is ai - b1
	dist_xy = tf.reduce_sum(tf.multiply(dxy, dxy), axis=1)  # m-by-m, the first column is (ai-b1)^2
	
	dyy = y_expand - y_permute  # m-by-d-by-m, the first page is ai - b1
	dist_yy = tf.reduce_sum(tf.multiply(dyy, dyy), axis=1)  # m-by-m, the first column is (ai-b1)^2

	return dist_xx, dist_xy, dist_yy


def compute_mmd(x, y, sigma_sqr=1.0, wo_diag=False, normalize=False):
	if normalize:
		x = tf.math.l2_normalize(x, axis=-1)
		y = tf.math.l2_normalize(y, axis=-1)

	if wo_diag:
		dist_xx, dist_xy, dist_yy = get_squared_dist_ref(x, y)
		k_xx = tf.exp(-dist_xx / (2.0 * sigma_sqr**2))
		k_yy = tf.exp(-dist_yy / (2.0 * sigma_sqr**2))
		k_xy = tf.exp(-dist_xy / (2.0 * sigma_sqr**2))

		batch_size = x.get_shape()[0].value
		m = tf.constant(batch_size, tf.float32)
		e_kxx = matrix_mean_wo_diagonal(k_xx, m)
		e_kxy = matrix_mean_wo_diagonal(k_xy, m)
		e_kyy = matrix_mean_wo_diagonal(k_yy, m)
		mmd = e_kxx + e_kyy - 2.0 * e_kxy
	else:
		k_xx = compute_kernel(x, x)
		k_yy = compute_kernel(y, y)
		k_xy = compute_kernel(x, y)
		mmd = tf.reduce_mean(k_xx) + tf.reduce_mean(k_yy) - 2 * tf.reduce_mean(k_xy)

	mmd = tf.where(mmd > 0, mmd, 0, name='value')
	return mmd