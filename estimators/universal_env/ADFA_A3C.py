'''
Type anomaly detection file
'''

import tensorflow as tf

import threading
import multiprocessing
import os
import shutil
import itertools

from network_classification import NetworkClassificationEnv

from estimators import ValueEstimator, PolicyEstimator
from policy_monitor import PolicyMonitor
from worker import Worker


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)


tf.logging.set_verbosity(tf.logging.INFO)



tf.flags.DEFINE_string("model_dir", "tmp/a3c", "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_integer("t_max", 1, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 120, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", True, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", None, "Number of threads to run. If not set we run [num_cpu_cores] threads.")



FLAGS = tf.flags.FLAGS


# Initialization of the enviroment
def make_env(**kwargs):
    train_path = '../../datasets/ADFA/UNSW_NB15_training-set.csv'
    test_path = '../../datasets/ADFA/UNSW_NB15_testing-set.csv'

    formated_train_path = "../../datasets/formated/formated_train_ADFA.data"
    formated_test_path = "../../datasets/formated/formated_test_ADFA.data"

    train_test = kwargs.get('train_test','train')

    ########################################################################
    # Aguments needed by enviroment:
    
    # Map from attack to type
    attack_map = {'Normal': 'Normal',
                  'Generic': 'Generic',
                  'Exploits': 'Exploits',
                  'Fuzzers':'Fuzzers',
                  'DoS':'DoS',
                  'Reconnaissance':'Reconnaissance',
                  'Analysis':'Analysis',
                  'Backdoor':'Backdoor',
                  'Shellcode':'Shellcode',
                  'Worms':'Worms'
                }
    
    column_names = ['id_drop','dur','proto','service_logarithm','state_logarithm',
                    'spkts_logarithm','dpkts_logarithm','sbytes_logarithm',
                    'dbytes_logarithm','rate_logarithm','sttl_logarithm',
                    'dttl_logarithm','sload_logarithm','dload_logarithm',
                    'sloss_logarithm','dloss_logarithm','sinpkt_logarithm',
                    'dinpkt_logarithm','sjit_logarithm','djit_logarithm',
                    'swin_logarithm','stcpb_logarithm','dtcpb_logarithm',
                    'dwin_logarithm','tcprtt_logarithm','synack','ackdat','smean',
                    'dmean','trans_depth','response_body_len','ct_srv_src',
                    'ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm',
                    'ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login',
                    'ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst',
                    'is_sm_ips_ports','labels','labels_drop']
    
    ##########################################################################
    
    batch_size = 1
    fails_episode = 100 # number of fails in a episode
    
    # Initialization of the enviroment
    env = NetworkClassificationEnv(
            train_test,
            attack_map,
            column_names=column_names,
            train_path=train_path,test_path=test_path,
            formated_train_path = formated_train_path,
            formated_test_path = formated_test_path,
            batch_size = batch_size,
            fails_episode = fails_episode
            )
    return env

env_ = make_env()
VALID_ACTIONS = list(range(env_.action_space.n))


# Set the number of workers
NUM_WORKERS = multiprocessing.cpu_count()
if FLAGS.parallelism:
  NUM_WORKERS = FLAGS.parallelism


MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Optionally empty model directory
if FLAGS.reset:
  shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))

with tf.device("/cpu:0"):
    
    # Keeps track of the number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)
    
    # Global policy and value nets
    with tf.variable_scope("global") as vs:
        policy_net = PolicyEstimator(num_outputs=len(VALID_ACTIONS),
                                     observation_space=env_.observation_len)
        value_net = ValueEstimator(observation_space=env_.observation_len,
                                   reuse=True)
        
    # Global step iterator
    global_counter = itertools.count()


    # Create worker graphs
    workers = []
    for worker_id in range(NUM_WORKERS):
        # We only write summaries in one of the workers because they're
        # pretty much identical and writing them on all workers
        # would be a waste of space
        worker_summary_writer = None
        if worker_id == 0:
            worker_summary_writer = summary_writer
            
        worker = Worker(name="worker_{}".format(worker_id),
                      env=make_env(),
                      policy_net=policy_net,
                      value_net=value_net,
                      global_counter=global_counter,
                      discount_factor = 0.001,
                      summary_writer=worker_summary_writer,
                      max_global_steps=FLAGS.max_global_steps)
        workers.append(worker)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=10)

    # Used to occasionally save videos for our policy net
    # and write episode rewards to Tensorboard
    pe = PolicyMonitor(
            env=make_env(train_test='test'),
            policy_net=policy_net,
            summary_writer=summary_writer,
            saver=saver)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    

    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        
        
        # Start worker threads
    worker_threads = []
    for worker in workers:
        worker_fn = lambda worker=worker: worker.run(sess, coord, FLAGS.t_max)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)

    # Start a thread for policy eval task
    monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
    monitor_thread.start()

    # Wait for all workers to finish
    coord.join(worker_threads)
    
    
    




    
    
    
