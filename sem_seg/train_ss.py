import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import indoor3d_util
# from model import *

parser = argparse.ArgumentParser()
# parser.add_argument('--num_gpu', type=int, default=2, help='the number of GPUs to use [default: 2]')
parser.add_argument('--gpu', type=int, default=2, help='GPU to use [default: GPU 2]')
parser.add_argument('--model', default='pan3_sseg', help='Model name [default: pan3_sseg]')
parser.add_argument('--log', default='log_sseg', help='Log dir [default: log_sseg]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=101, help='Epoch to run [default: 101]')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training for each GPU [default: 12]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
parser.add_argument('--test_room_data_filelist', default='./meta/area6_data_label.txt', 
						help='TXT filename, filelist, each line is a test room data label file.')
parser.add_argument('--coarse_loss', type=float, default=1, help='Whether to use coarse loss')
parser.add_argument('--mmd_loss', type=float, default=200, help='Whether to use mmd loss')
parser.add_argument('--pfs', action='store_true', help='Whether to use pfs')
parser.add_argument('--fuct', action='store_true', help='Whether to use fully concated features')
FLAGS = parser.parse_args()

# TOWER_NAME = 'tower'

GPU_INDEX = FLAGS.gpu
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

COARSE_FLAG = FLAGS.coarse_loss
MMD_FLAG = FLAGS.mmd_loss
PFS_FLAG = FLAGS.pfs
FUCT_FLAG = FLAGS.fuct

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX)

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

LOG_DIR = FLAGS.log
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
# os.system('cp model.py %s' % (LOG_DIR)) 
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_ss.py %s' % (LOG_DIR)) 
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 13

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

ALL_FILES = provider.getDataFiles('indoor3d_sem_seg_hdf5_data/all_files.txt') 
room_filelist = [line.rstrip() for line in open('indoor3d_sem_seg_hdf5_data/room_filelist.txt')] 
print (len(room_filelist))

# Load ALL data
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
	data_batch, label_batch = provider.loadDataFile(h5_filename)
	data_batch_list.append(data_batch)
	label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
print(data_batches.shape)
print(label_batches.shape)

test_area = 'Area_'+str(FLAGS.test_area)
train_idxs = []
test_idxs = []
for i,room_name in enumerate(room_filelist):
	if test_area in room_name:
		test_idxs.append(i)
	else:
		train_idxs.append(i)

train_data = data_batches[train_idxs,...]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs,...]
test_label = label_batches[test_idxs]
print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)

TEST_ROOM_PATH_LIST = [os.path.join(ROOT_DIR,line.rstrip()) for line in open(FLAGS.test_room_data_filelist)]

BEST_MEAN_IOU = 0
BEST_ALL_ACC = 0
BEST_CLS_ACC = 0

def log_string(out_str):
	LOG_FOUT.write(out_str+'\n')
	LOG_FOUT.flush()
	print(out_str)


def get_learning_rate(batch):
	learning_rate = tf.train.exponential_decay(
						BASE_LEARNING_RATE,  # Base learning rate.
						batch * BATCH_SIZE,  # Current index into the dataset.
						DECAY_STEP,          # Decay step.
						DECAY_RATE,          # Decay rate.
						staircase=True)
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
	return learning_rate        

def get_bn_decay(batch):
	bn_momentum = tf.train.exponential_decay(
						BN_INIT_DECAY,
						batch*BATCH_SIZE,
						BN_DECAY_DECAY_STEP,
						BN_DECAY_DECAY_RATE,
						staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay

def average_gradients(tower_grads):
	"""Calculate average gradient for each shared variable across all towers.

	Note that this function provides a synchronization point across all towers.

	Args:
		tower_grads: List of lists of (gradient, variable) tuples. The outer list
		is over individual gradients. The inner list is over the gradient
		calculation for each tower.
	Returns:
		 List of pairs of (gradient, variable) where the gradient has been 
		 averaged across all towers.
	"""
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			expanded_g = tf.expand_dims(g, 0)
			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		grad = tf.concat(grads, 0)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads

def train():
	# with tf.Graph().as_default(), tf.device('/cpu:0'):
	with tf.Graph().as_default():
		with tf.device('/gpu:'+str(GPU_INDEX)):
			batch = tf.Variable(0, trainable=False)
			
			bn_decay = get_bn_decay(batch)
			tf.summary.scalar('bn_decay', bn_decay)

			learning_rate = get_learning_rate(batch)
			tf.summary.scalar('learning_rate', learning_rate)
			
			trainer = tf.train.AdamOptimizer(learning_rate)
			
			pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
			is_training_pl = tf.placeholder(tf.bool, shape=())
			
			pred, end_points = MODEL.get_model(pointclouds_pl, labels_pl, is_training_pl, bn_decay, 
				COARSE_FLAG, MMD_FLAG, PFS_FLAG, FUCT_FLAG)
			loss = MODEL.get_loss(pred, labels_pl, end_points, COARSE_FLAG, MMD_FLAG)
			tf.summary.scalar('loss', loss)

			# correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_phs[-1]))
			correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
			accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
			tf.summary.scalar('accuracy', accuracy)
			
			train_op = trainer.minimize(loss, global_step=batch)
			saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
		
		# Create a session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		sess = tf.Session(config=config)

		# Add summary writers
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
									sess.graph)
		test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

		# Init variables for two GPUs
		init = tf.group(tf.global_variables_initializer(),
						 tf.local_variables_initializer())
		sess.run(init)

		ops = {'pointclouds_pl': pointclouds_pl,
				 'labels_pl': labels_pl,
				 'is_training_pl': is_training_pl,
				 'pred': pred,
				 'loss': loss,
				 'train_op': train_op,
				 'merged': merged,
				 'step': batch}

		for epoch in range(MAX_EPOCH):
			log_string('**** EPOCH %03d ****' % (epoch))
			sys.stdout.flush()
			 
			train_one_epoch(sess, ops, train_writer)

			# if epoch >= 10:
			best_all_acc_flag, best_cls_acc_flag, best_mean_iou_flag = eval_one_epoch(sess, ops, test_writer)
		
			if best_all_acc_flag == True:
				save_path = saver.save(sess, os.path.join(LOG_DIR,'best_all_acc_model'+'.ckpt'))
				log_string("Model saved in file: %s" % save_path)

			if best_cls_acc_flag == True:
				save_path = saver.save(sess, os.path.join(LOG_DIR,'best_cls_acc_model'+'.ckpt'))
				log_string("Model saved in file: %s" % save_path)

			if best_mean_iou_flag == True:
				save_path = saver.save(sess, os.path.join(LOG_DIR,'best_mean_iou_model'+'.ckpt'))
				log_string("Model saved in file: %s" % save_path)

			# Save the variables to disk.
			if epoch >= 10 and epoch % 5 == 0:
				save_path = saver.save(sess, os.path.join(LOG_DIR,'epoch_' + str(epoch)+'.ckpt'))
				log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
	""" ops: dict mapping from string to tf ops """
	is_training = True
	
	log_string('----')
	current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label) 
	
	file_size = current_data.shape[0]
	# num_batches = file_size // (FLAGS.num_gpu * BATCH_SIZE) 
	num_batches = file_size // (BATCH_SIZE) 
	
	total_correct = 0
	total_seen = 0
	loss_sum = 0
	
	for batch_idx in range(num_batches):
		if batch_idx % 100 == 0:
			print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
		start_idx_0 = batch_idx * BATCH_SIZE
		end_idx_0 = (batch_idx+1) * BATCH_SIZE
		
		
		feed_dict = {ops['pointclouds_pl']: current_data[start_idx_0:end_idx_0, :, :],
								 ops['labels_pl']: current_label[start_idx_0:end_idx_0],
								 ops['is_training_pl']: is_training}
		summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
										 feed_dict=feed_dict)
		train_writer.add_summary(summary, step)
		pred_val = np.argmax(pred_val, 2)
		correct = np.sum(pred_val == current_label[start_idx_0:end_idx_0])
		total_correct += correct
		total_seen += (BATCH_SIZE*NUM_POINT)
		loss_sum += loss_val
	
	log_string('mean loss: %f' % (loss_sum / float(num_batches)))
	log_string('accuracy: %f' % (total_correct / float(total_seen)))

def eval_one_epoch(sess, ops, test_writer):
	global BEST_MEAN_IOU
	global BEST_ALL_ACC
	global BEST_CLS_ACC

	log_string('evaluation')

	is_training = False
	gt_classes = [0 for _ in range(13)]
	positive_classes = [0 for _ in range(13)]
	true_positive_classes = [0 for _ in range(13)]

	for room_path in TEST_ROOM_PATH_LIST:
		current_data, current_label = indoor3d_util.room2blocks_wrapper_normalized(room_path, NUM_POINT)
		current_data = current_data[:,0:NUM_POINT,:]
		current_label = np.squeeze(current_label)

		data_label = np.load(room_path)
		data = data_label[:,0:6]
		max_room_x = max(data[:,0])
		max_room_y = max(data[:,1])
		max_room_z = max(data[:,2])

		file_size = current_data.shape[0]
		num_batches = file_size // BATCH_SIZE
		# print(file_size)

		for batch_idx in range(num_batches):
			start_idx = batch_idx * BATCH_SIZE
			end_idx = min((batch_idx+1) * BATCH_SIZE, file_size)
			cur_batch_size = end_idx - start_idx
			
			feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
						 ops['labels_pl']: current_label[start_idx:end_idx],
						 ops['is_training_pl']: is_training}
			loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
											feed_dict=feed_dict)

			pred_label = np.argmax(pred_val, 2)

			for i in range(start_idx, end_idx):
				for j in range(NUM_POINT):
					gt_l = int(current_label[i, j])
					pred_l = int(pred_label[i-start_idx, j])
					gt_classes[gt_l] += 1
					positive_classes[pred_l] += 1
					true_positive_classes[gt_l] += int(gt_l==pred_l)

	current_all_acc = (sum(true_positive_classes)/float(sum(positive_classes)))
	log_string('overall accuracy: %f' % current_all_acc)

	class_list = []
	for i in range(13):
		acc_class = true_positive_classes[i]/float(gt_classes[i])
		class_list.append(acc_class)

	current_cls_acc = (sum(class_list)/13.0)
	log_string('avg class accuracy: %f' % current_cls_acc)
	log_string ('IoU: ')
	iou_list = []
	for i in range(13):
		iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
		log_string('%f' % iou)
		iou_list.append(iou)

	current_mean_iou = (sum(iou_list)/13.0)
	log_string('avg IoU %f' % current_mean_iou)

	best_all_acc_flag, best_cls_acc_flag, best_mean_iou_flag = False, False, False
	if current_all_acc > BEST_ALL_ACC:
		BEST_ALL_ACC = current_all_acc
		best_all_acc_flag = True
	if current_cls_acc > BEST_CLS_ACC:
		BEST_CLS_ACC = current_cls_acc
		best_cls_acc_flag = True
	if current_mean_iou > BEST_MEAN_IOU:
		BEST_MEAN_IOU = current_mean_iou
		best_mean_iou_flag = True

	log_string('best_all_acc: %f' % BEST_ALL_ACC)
	log_string('best_cls_acc: %f' % BEST_CLS_ACC)
	log_string('best_mean_iou: %f' % BEST_MEAN_IOU)

	return best_all_acc_flag, best_cls_acc_flag, best_mean_iou_flag


if __name__ == "__main__":
	train()
	LOG_FOUT.close()
