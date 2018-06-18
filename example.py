import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from keras import losses, metrics
import tensorflow as tf
from clusterone import get_data_path, get_logs_path
from random import randint
import os


bits = 32
# Parameters
train_batch_size = 50
train_set_size = 30000
val_batch_size = 50
val_set_size = 1000
epochs = 50
print_rate = 5

# Turn off Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def get_data(n):
	"""
	[get_data] returns [n] samples of input and output data for
		a 64 bit bit-wise XOR function.
	"""
	x_data = []
	y_data = []
	for i in range(n+val_set_size):
		temp_x = []
		temp_y = []
		for j in range(2 * bits):
			temp_x.append(randint(0, 1))
		for k in range(bits):
			x1 = temp_x[k]
			x2 = temp_x[k+bits]
			temp_y.append(x1 ^ x2)
		x_data.append(temp_x)
		y_data.append(temp_y)
		
	x_train = x_data[:n]
	y_train = y_data[:n]
	x_val = x_data[n:]
	y_val = y_data[n:]
	
	return x_train, y_train, x_val, y_val

####################################
# Begin distributed code

PATH_TO_LOCAL_LOGS = os.path.expanduser(r'C:\Users\Ryan Meredith\Documents\github\distributed_tensorflow\logs')
ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser(r'C:\Users\Ryan Meredith\Documents\github\distributed_tensorflow\data')

flags = tf.app.flags

# Get the environment parameters for distributed TensorFlow
try:
    job_name = os.environ['JOB_NAME']
    task_index = os.environ['TASK_INDEX']
    ps_hosts = os.environ['PS_HOSTS']
    worker_hosts = os.environ['WORKER_HOSTS']
except: # we are not on TensorPort, assuming local, single node
	job_name = None    
	task_index = 0
	ps_hosts = None
	worker_hosts = None
	
# Flags for configuring the distributed task
flags.DEFINE_string("job_name", job_name,
                    "job name: worker or ps")
flags.DEFINE_integer("task_index", task_index,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the chief worker task that performs the variable "
                     "initialization and checkpoint handling")
flags.DEFINE_string("ps_hosts", ps_hosts,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", worker_hosts,
                    "Comma-separated list of hostname:port pairs")

# Training related flags
flags.DEFINE_string("data_dir",
                    get_data_path(
                        dataset_name = "", #all mounted repo
                        local_root = ROOT_PATH_TO_LOCAL_DATA,
                        local_repo = "",
                        path = ""
                        ),
                    "Path to store logs and checkpoints. It is recommended"
                    "to use get_logs_path() to define your logs directory."
                    "so that you can switch from local to clusterone without"
                    "changing your code."
                    "If you set your logs directory manually make sure"
                    "to use /logs/ when running on ClusterOne cloud.")
flags.DEFINE_string("log_dir",
                     get_logs_path(root=PATH_TO_LOCAL_LOGS),
                    "Path to dataset. It is recommended to use get_data_path()"
                    "to define your data directory.so that you can switch "
                    "from local to clusterone without changing your code."
                    "If you set the data directory manually makue sure to use"
                    "/data/ as root path when running on ClusterOne cloud.")


FLAGS = flags.FLAGS
		
# This function defines the master, ClusterSpecs and device setters
def device_and_target():
    # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
    # Don't set a device.
    if FLAGS.job_name is None:
        print("Running single-machine training")
        return (None, "")

    # Otherwise we're running distributed TensorFlow.
    print("Running distributed training")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
        raise ValueError("Must specify an explicit `ps_hosts`")
    if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
        raise ValueError("Must specify an explicit `worker_hosts`")

    cluster_spec = tf.train.ClusterSpec({
            "ps": FLAGS.ps_hosts.split(","),
            "worker": FLAGS.worker_hosts.split(","),
    })
    server = tf.train.Server(
            cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()

    worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    return (
            tf.train.replica_device_setter(
                    worker_device=worker_device,
                    cluster=cluster_spec),
            server.target,
    )

device, target = device_and_target()        

# Defining graph
with tf.device(device):
	#TODO define your graph here
	# Defining network
	input_ = tf.placeholder(tf.float32, shape = (None, 2 * bits))
	x = Dense(128, activation='relu')(input_)
	x = Dropout(0.3)(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.3)(x)
	preds = Dense(bits, activation='sigmoid')(x)
	labels = tf.placeholder(tf.float32, shape=(None, bits))
	
	pred_temp = tf.equal(tf.round(preds), tf.round(labels))
	accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
	
	loss = tf.reduce_mean(losses.mean_squared_error(labels, preds))

	# Defining training optimizer
	optimizer = tf.train.AdamOptimizer()
	global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
	train_step = optimizer.minimize(loss,global_step=global_step)
	
	# Initialize all variables
	init = tf.global_variables_initializer()

# Retrieving training data
x_train, y_train, x_val, y_val = get_data(train_set_size)

#Defining the number of training steps
hooks=[tf.train.StopAtStepHook(last_step=epochs * (train_set_size/train_batch_size))]

with tf.train.MonitoredTrainingSession(master=target,
	is_chief=(FLAGS.task_index == 0),
	checkpoint_dir=FLAGS.log_dir,
	hooks = hooks) as sess:

	sess.run(init)

	epoch = 0
	while not sess.should_stop():
		epoch += 1
	
	
	
		#for epoch in range(epochs):
		avg_loss = 0
		total_batch = int(len(x_train)/train_batch_size)
		for i in range(total_batch):
			index1 = i * train_batch_size
			index2 = (i+1) * train_batch_size
			batch_x, batch_y = x_train[index1:index2], y_train[index1:index2]
			_, c = sess.run([train_step, loss], feed_dict = {input_: batch_x, labels: batch_y, K.learning_phase(): 1})
			
			avg_loss += c / total_batch

		if (epoch + 1) % print_rate == 0:
			# pred_temp = tf.equal(tf.round(preds), tf.round(labels))
			# accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
			val_acc = accuracy.eval({input_: x_val, labels: y_val, K.learning_phase(): 0},session = sess)
			print( "Epoch:", (epoch+1), "Loss =", "{:.5f}".format(avg_loss), "  Validation Accuracy:", val_acc)

	print ("\nTraining complete!")


	# find predictions on val set
	# pred_temp = tf.equal(tf.round(preds), tf.round(labels))
	# accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
	# print( "Final Validation Accuracy:", accuracy.eval({input_: x_val, labels: y_val, K.learning_phase(): 0}, session = sess))