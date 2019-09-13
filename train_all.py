'''
	Single-GPU training.
	Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import modelnet_dataset
import modelnet_h5_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pan3_cls', help='Model name [default: pan3_cls]')
parser.add_argument('--log', default='log_cls', help='Log dir [default: log_cls]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--rotate', action='store_true', help='Whether to use rotation augmentation')
parser.add_argument('--mmd_loss', type=float, default=0, help='Whether to use mmd loss')
parser.add_argument('--pfs', action='store_true', help='Whether to use pfs')
parser.add_argument('--dpts', action='store_true', help='Whether to drop points')
parser.add_argument('--eval', action='store_true', help='Whether to evaluate during training')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BEST_ACC = 0
BEST_CLS_ACC = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

NORMAL_FLAG = ~FLAGS.no_normal
ROTATE_FLAG = FLAGS.rotate
MMD_FLAG = FLAGS.mmd_loss
PFS_FLAG = FLAGS.pfs
DPTS_FLAG = FLAGS.dpts
EVAL_FLAG = FLAGS.eval

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX)

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_all.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 40

# Shapenet official train/test split
# if NORMAL_FLAG:
assert(NUM_POINT<=10000)
DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=NORMAL_FLAG, 
	batch_size=BATCH_SIZE, rotate=ROTATE_FLAG)
TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=NORMAL_FLAG, 
	batch_size=BATCH_SIZE)
# else:
# 	assert(NUM_POINT<=2048)
# 	TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), 
# 		batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True, rotate=ROTATE_FLAG)
# 	TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), 
# 		batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

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
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
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

def train():
	with tf.Graph().as_default():
		with tf.device('/gpu:'+str(GPU_INDEX)):
			pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NORMAL_FLAG)
			is_training_pl = tf.placeholder(tf.bool, shape=())
			
			# Note the global_step=batch parameter to minimize. 
			# That tells the optimizer to helpfully increment the 'batch' parameter
			# for you every time it trains.
			batch = tf.get_variable('batch', [],
				initializer=tf.constant_initializer(0), trainable=False)
			bn_decay = get_bn_decay(batch)
			tf.summary.scalar('bn_decay', bn_decay)

			# Get model and loss 
			pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, 
				pfs_flag=PFS_FLAG, bn_decay=bn_decay)
			loss = MODEL.get_loss(pred, labels_pl, end_points, MMD_FLAG)
			tf.summary.scalar('loss', loss)

			correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
			accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
			tf.summary.scalar('accuracy', accuracy)

			print ("--- Get training operator")
			# Get training operator
			learning_rate = get_learning_rate(batch)
			tf.summary.scalar('learning_rate', learning_rate)
			if OPTIMIZER == 'momentum':
				optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
			elif OPTIMIZER == 'adam':
				optimizer = tf.train.AdamOptimizer(learning_rate)
			train_op = optimizer.minimize(loss, global_step=batch)
			
			# Add ops to save and restore all the variables.
			saver = tf.train.Saver()
		
		# Create a session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		# config.gpu_options.allow_growth = False
		# config.gpu_options.per_process_gpu_memory_fraction = 0.12 # 0.2
		config.allow_soft_placement = True
		config.log_device_placement = False
		sess = tf.Session(config=config)

		# Add summary writers
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
		test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

		# Init variables
		init = tf.global_variables_initializer()
		sess.run(init)

		ops = {'pointclouds_pl': pointclouds_pl,
			   'labels_pl': labels_pl,
			   'is_training_pl': is_training_pl,
			   'pred': pred,
			   'loss': loss,
			   'train_op': train_op,
			   'merged': merged,
			   'step': batch,
			   'end_points': end_points}

		best_acc = -1
		for epoch in range(MAX_EPOCH):
			log_string('**** EPOCH %03d ****' % (epoch))
			sys.stdout.flush()
			 
			train_one_epoch(sess, ops, train_writer)

			if EVAL_FLAG:
				best_acc_flag, best_cls_acc_flag = eval_one_epoch(sess, ops, test_writer)

				if best_acc_flag:
					save_path = saver.save(sess, os.path.join(LOG_DIR, "best_acc_model.ckpt"))
					log_string("Model saved in file: %s" % save_path)

				if best_cls_acc_flag:
					save_path = saver.save(sess, os.path.join(LOG_DIR, "best_cls_acc_model.ckpt"))
					log_string("Model saved in file: %s" % save_path)

			# Save the variables to disk.
			if epoch % 10 == 0:
				save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
				log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
	""" ops: dict mapping from string to tf ops """
	is_training = True
	
	log_string(str(datetime.now()))

	# Make sure batch data is of same size
	cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
	cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

	total_correct = 0
	total_seen = 0
	loss_sum = 0
	batch_idx = 0
	while TRAIN_DATASET.has_next_batch():
		batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)
		if DPTS_FLAG:
			batch_data = provider.random_point_dropout(batch_data)
		bsize = batch_data.shape[0]
		cur_batch_data[0:bsize,...] = batch_data
		cur_batch_label[0:bsize] = batch_label

		feed_dict = {ops['pointclouds_pl']: cur_batch_data,
					 ops['labels_pl']: cur_batch_label,
					 ops['is_training_pl']: is_training,}
		summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
			ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
		train_writer.add_summary(summary, step)
		pred_val = np.argmax(pred_val, 1)
		correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
		total_correct += correct
		total_seen += bsize
		loss_sum += loss_val
		if (batch_idx+1)%50 == 0:
			log_string(' ---- batch: %03d ----' % (batch_idx+1))
			log_string('mean loss: %f' % (loss_sum / 50))
			log_string('accuracy: %f' % (total_correct / float(total_seen)))
			total_correct = 0
			total_seen = 0
			loss_sum = 0
		batch_idx += 1

	TRAIN_DATASET.reset()
		
def eval_one_epoch(sess, ops, test_writer):
	""" ops: dict mapping from string to tf ops """
	global EPOCH_CNT
	global BEST_ACC
	global BEST_CLS_ACC

	is_training = False

	# Make sure batch data is of same size
	cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
	cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

	total_correct = 0
	total_seen = 0
	loss_sum = 0
	batch_idx = 0
	shape_ious = []
	total_seen_class = [0 for _ in range(NUM_CLASSES)]
	total_correct_class = [0 for _ in range(NUM_CLASSES)]
	
	log_string(str(datetime.now()))
	log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

	while TEST_DATASET.has_next_batch():
		batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
		bsize = batch_data.shape[0]
		# print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
		# for the last batch in the epoch, the bsize:end are from last batch
		cur_batch_data[0:bsize,...] = batch_data
		cur_batch_label[0:bsize] = batch_label

		if ROTATE_FLAG:
			batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
			for vote_idx in range(12):
				# Shuffle point order to achieve different farthest samplings
				shuffled_indices = np.arange(NUM_POINT)
				np.random.shuffle(shuffled_indices)
				if NORMAL_FLAG:
					rotated_data = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:, shuffled_indices, :],
						vote_idx/float(12) * np.pi * 2)
					rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
				else:
					rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
	                                                  vote_idx/float(12) * np.pi * 2)
					rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)

				jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
				
				jittered_data = provider.jitter_point_cloud(jittered_data)
				rotated_data[:,:,0:3] = jittered_data
				# else:
					# rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
						# vote_idx/float(12) * np.pi * 2)
				feed_dict = {ops['pointclouds_pl']: rotated_data,
							 ops['labels_pl']: cur_batch_label,
							 ops['is_training_pl']: is_training}
				loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
				batch_pred_sum += pred_val
			pred_val = np.argmax(batch_pred_sum, 1)

		else:
			feed_dict = {ops['pointclouds_pl']: cur_batch_data,
					 ops['labels_pl']: cur_batch_label,
					 ops['is_training_pl']: is_training}
			summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
				ops['loss'], ops['pred']], feed_dict=feed_dict)
			test_writer.add_summary(summary, step)
			pred_val = np.argmax(pred_val, 1)

		correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
		total_correct += correct
		total_seen += bsize
		loss_sum += loss_val
		batch_idx += 1
		for i in range(bsize):
			l = batch_label[i]
			total_seen_class[l] += 1
			total_correct_class[l] += (pred_val[i] == l)
	
	current_acc = total_correct / float(total_seen)
	current_cls_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))

	log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
	log_string('eval accuracy: %f'% (current_acc))
	log_string('eval avg class acc: %f' % (current_cls_acc))

	best_acc_flag, best_cls_acc_flag = False, False
	if current_acc > BEST_ACC:
		BEST_ACC = current_acc
		best_acc_flag = True
	if current_cls_acc > BEST_CLS_ACC:
		BEST_CLS_ACC = current_cls_acc
		best_cls_acc_flag = True

	log_string('eval best accuracy: %f'% (BEST_ACC))
	log_string('eval best avg class acc: %f'% (BEST_CLS_ACC))

	EPOCH_CNT += 1

	TEST_DATASET.reset()
	return (best_acc_flag, best_cls_acc_flag)


if __name__ == "__main__":
	log_string('pid: %s'%(str(os.getpid())))
	train()
	LOG_FOUT.close()
