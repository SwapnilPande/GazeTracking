import tensorflow as tf
from tensorboard import main as tb
import json

with open('ml_param.json') as f:
	paramJSON = json.load(f)
	pathLogging = paramJSON['pathLogging']
tbPath = pathLogging + '/tensorboard'
tf.flags.FLAGS.logdir = tbPath
tb.main()