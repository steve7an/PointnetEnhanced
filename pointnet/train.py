import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import time, datetime
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
#BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
BASE_DIR = ''
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--run_mode', default='normal', help='Run mode [default: normal]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu

#MAX_NUM_POINT = 2048
HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

#LOG_DIR = FLAGS.log_dir
#if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_DIR = createLogDir(FLAGS.log_dir)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def createLogDir(LOG_DIR):
    '''Always create a new logdir to prevent overwriting'''
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    else:
        print('Log dir already exists! creating a new one..............')
        n = 0
        while True:
            n+=1
            new_log_dir = LOG_DIR+'/'+str(n)
            if not os.path.exists(new_log_dir):
                os.makedirs(new_log_dir)
                print('New log dir:'+new_log_dir)
                break
        FLAGS.log_dir = new_log_dir
        LOG_DIR = new_log_dir
    return LOG_DIR



def get_running_time(start_time):
    ''' Given a start time, calculate the diff in seconds and return total running time '''
    diff_seconds = time.time() - start_time
    return str(datetime.timedelta(seconds=diff_seconds))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch, gparams):
    BASE_LEARNING_RATE = gparams['BASE_LEARNING_RATE']
    BATCH_SIZE = gparams['BATCH_SIZE']
    DECAY_STEP = gparams['DECAY_STEP']
    DECAY_RATE = gparams['DECAY_RATE']

    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch, gparams):
    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_CLIP = 0.99
    BATCH_SIZE = gparams['BATCH_SIZE']
    BN_DECAY_DECAY_STEP = gparams['BN_DECAY_DECAY_STEP']

    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train(gparams):
    MODEL = gparams['MODEL']
    OPTIMIZER = gparams['OPTIMIZER']
    BATCH_SIZE = gparams['BATCH_SIZE']
    NUM_POINT = gparams['NUM_POINT']
    MOMENTUM =gparams['MOMENTUM']
    MAX_EPOCH =gparams['MAX_EPOCH']

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print("is_training_placeholder:{}".format(is_training_pl))
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch, gparams)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch, gparams)
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
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        #Use the last mean loss for hyperparameter optimization
        mean_loss = accuracy = avg_class_accuracy = 0
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            start_time = time.time() 
            train_one_epoch(sess, ops, train_writer, gparams)
            mean_loss,accuracy, avg_class_accuracy = eval_one_epoch(sess, ops, test_writer, gparams)
            print("--- Total running time for EPOCH %03d : %s ---" % (epoch, get_running_time(start_time)))

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

        return {
            'loss': mean_loss,
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            'Evaluate other params': {'accuracy': accuracy, 'avg_class_accuracy': avg_class_accuracy},
            # -- attachments are handled differently
            'attachments':
            {'time_module': pickle.dumps(time.time)}
        }


def train_one_epoch(sess, ops, train_writer,gparams):
    NUM_POINT = gparams['NUM_POINT']
    BATCH_SIZE = gparams['BATCH_SIZE']

    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
       
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer, gparams):
    NUM_POINT = gparams['NUM_POINT']
    BATCH_SIZE =gparams['BATCH_SIZE']
    NUM_CLASSES = 40

    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
    
    mean_loss = loss_sum / float(total_seen)
    accuracy = total_correct / float(total_seen)
    avg_class_accuracy = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
    log_string('eval mean loss: %f' % (mean_loss))
    log_string('eval accuracy: %f'% accuracy)
    log_string('eval avg class acc: %f' % avg_class_accuracy)
         
    return mean_loss, accuracy, avg_class_accuracy


def hyperOptMain():
    '''Run the training using hyper optimization'''
    #('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
    #('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
    #('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
    #('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    #('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    #('--optimizer', default='adam', help='adam or momentum [default: adam]')
    #('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]
    space = {
        'num_points': hp.choice('num_points',[256,512,1024,2048]),
        'batch_size': hp.choice('batch_size',[2,4,8,16,32]),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.001),
        'momentum': hp.uniform('momentum',0.1, 0.9),
        'optimizer': hp.choice('optimizer',['adam','momentum']),
        'decay_step': hp.uniform('decay_step',10000, 200000),
        'decay_rate': hp.uniform('decay_rate',0.1, 0.7)
    }
    max_evals = 3
    trials = Trials()
    best = fmin(main,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials)

    print (best)


def main(params=[]):
    gparams = dict()
    if params:
        print ("Starting hyperopt")
        gparams['BATCH_SIZE'] = params['batch_size']
        gparams['NUM_POINT'] = params['num_points']
        gparams['BASE_LEARNING_RATE'] = params['learning_rate']
        gparams['MOMENTUM'] = params['momentum']
        gparams['OPTIMIZER'] = params['optimizer']
        gparams['DECAY_STEP']  = params['decay_step']
        gparams['MAX_EPOCH'] = 1
        gparams['DECAY_RATE'] = params['decay_rate']
    else:
        gparams['BATCH_SIZE'] = FLAGS.batch_size        
        gparams['NUM_POINT'] = FLAGS.num_point
        gparams['BASE_LEARNING_RATE'] = FLAGS.learning_rate
        gparams['MOMENTUM'] = FLAGS.momentum
        gparams['OPTIMIZER'] = FLAGS.optimizer
        gparams['DECAY_STEP'] = FLAGS.decay_step
        gparams['MAX_EPOCH'] = FLAGS.max_epoch
        gparams['DECAY_RATE'] = FLAGS.decay_rate

    gparams['BN_DECAY_DECAY_STEP'] = float(gparams['DECAY_STEP'])

   # print("15:Batch size:{} and Num point:{}".format(gparams['BATCH_SIZE'],gparams['NUM_POINT']))

    
    gparams['MODEL'] = importlib.import_module(FLAGS.model) # import network module
    gparams['MODEL_FILE'] = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')

    os.system('cp "%s" "%s"' % (gparams['MODEL_FILE'], LOG_DIR)) # bkp of model def
    os.system('cp train.py "%s"' % (LOG_DIR)) # bkp of train procedure


    start_time = time.time()
    result = train(gparams)
    print("--- Total running time for MAIN : %s ---" % (get_running_time(start_time)))
    return result



if __name__ == "__main__":
    print ("Starting the run with the mode:{}".format(FLAGS.run_mode))
    if FLAGS.run_mode == "normal":
        main()
    else:
        hyperOptMain()
    
    LOG_FOUT.close()
