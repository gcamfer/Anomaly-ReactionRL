'''
Type anomaly detection file
'''

import tensorflow as tf

import threading
import multiprocessing
import os
import shutil
import itertools

from my_enviroment import my_env

from estimators import ValueEstimator, PolicyEstimator
from policy_monitor import PolicyMonitor
from worker import Worker


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)






tf.flags.DEFINE_string("model_dir", "/RL/TFM/AnomalyDetectionRL/A3C/tmp/a3c", "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_integer("t_max", 5, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", 10000, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 300, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", True, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", None, "Number of threads to run. If not set we run [num_cpu_cores] threads.")



FLAGS = tf.flags.FLAGS


# Initialization of the enviroment
def make_env():
    kdd_train = '../../datasets/NSL/KDDTrain+.txt'
    kdd_test = '../../datasets/NSL/KDDTest+.txt'
    
    formated_train_path = "../../datasets/formated/formated_train_type.data"
    formated_test_path = "../../datasets/formated/formated_test_type.data"
    batch_size = 1
    iterations_episode = 100
    
    env = my_env('train',train_path=kdd_train,test_path=kdd_test,
                formated_train_path = formated_train_path,
                formated_test_path = formated_test_path,
                batch_size=batch_size,
                iterations_episode=iterations_episode)
    return env

env_ = make_env()
VALID_ACTIONS = list(range(env_.action_space))


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
                                     observation_space=env_.observation_space)
        value_net = ValueEstimator(observation_space=env_.observation_space,
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
                      discount_factor = 0.99,
                      summary_writer=worker_summary_writer,
                      max_global_steps=FLAGS.max_global_steps)
        workers.append(worker)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=10)

    # Used to occasionally save videos for our policy net
    # and write episode rewards to Tensorboard
    pe = PolicyMonitor(
            env=make_env(),
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
    
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    kdd_train = '../../datasets/NSL/KDDTrain+.txt'
    kdd_test = '../../datasets/NSL/KDDTest+.txt'
    
    formated_train_path = "../../datasets/formated/formated_train_type.data"
    formated_test_path = "../../datasets/formated/formated_test_type.data"
    
    #TEST
    batch_size = 1
    env = my_env('test',formated_test_path = formated_test_path,batch_size=batch_size) 
    total_reward = 0    
    epochs = int(env.data_shape[0]/env.batch_size/1)

    true_labels = np.zeros(len(env.attack_types),dtype=int)
    estimated_labels = np.zeros(len(env.attack_types),dtype=int)
    estimated_correct_labels = np.zeros(len(env.attack_types),dtype=int)
    
    for e in range(epochs):
        #states , labels = env.get_sequential_batch(test_path,batch_size = env.batch_size)
        states , labels = env.get_batch(batch_size)
        
        # TODO: fix performance in this loop
        action_probs = worker._policy_net_predict(states, sess)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        
        reward = np.zeros(env.batch_size)
        
        true_labels += np.sum(labels).values

        estimated_labels[action] +=1              
        if action == np.argmax(labels.values):
            reward = 1
            estimated_correct_labels[action] += 1
        
        
        total_reward += reward
        print("\rEpoch {}/{} | Tot Rew -- > {}".format(e,epochs,total_reward), end="")
        
    Accuracy = estimated_correct_labels / true_labels
    Mismatch = estimated_labels - true_labels

    print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {}%'.format(total_reward,
          int(epochs*env.batch_size),float(100*total_reward/(epochs*env.batch_size))))
    outputs_df = pd.DataFrame(index = env.attack_types,columns = ["Estimated","Correct","Total","Acuracy"])
    for indx,att in enumerate(env.attack_types):
       outputs_df.iloc[indx].Estimated = estimated_labels[indx]
       outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
       outputs_df.iloc[indx].Total = true_labels[indx]
       outputs_df.iloc[indx].Acuracy = Accuracy[indx]*100
       outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])


    
print(outputs_df)

#%%



ind = np.arange(1,len(env.attack_types)+1)
fig, ax = plt.subplots()
width = 0.35
p1 = plt.bar(ind, estimated_correct_labels,width,color='g')
p2 = plt.bar(ind, 
             (np.abs(estimated_correct_labels-true_labels)\
              +np.abs(estimated_labels-estimated_correct_labels)),width,
             bottom=estimated_correct_labels,color='r')

    
ax.set_xticks(ind)
ax.set_xticklabels(env.attack_types,rotation='vertical')
#ax.set_yscale('log')

#ax.set_ylim([0, 100])
ax.set_title('Test set scores')
plt.legend((p1[0], p2[0]), ('Correct estimated', 'Incorrect estimated'))
plt.tight_layout()
#plt.show()
plt.savefig('../results/A3C_test_type.eps', format='eps', dpi=1000)




    
    
    
