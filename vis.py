import sys
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

with tf.Session() as sess:
    # Command line takes one arg for path to model.  Default is current directory.
    if len(sys.argv) == 1:
        filepath = '.'  # if no arg default to current directory
    else:
        filepath = str(sys.argv[1])
    filename = filepath + "/saved_model.pb"
    
    with gfile.FastGFile(filename, 'rb') as f:
        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
 
LOGDIR='.'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
print("\nNext steps:")
print("1) Run tensorboard --logdir=.")
print("2) Open localhost:6006 in browser")