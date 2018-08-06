import tensorflow as tf
from tensorboard import main as tb
import json

with open('ml_param.json') as f:
	paramJSON = json.load(f)
	pathLogging = paramJSON['dataPaths']['pathLogging']
tbPath = pathLogging + '/tensorboard'
print("Tensorboard path: " + tbPath)
tf.flags.FLAGS.logdir = tbPath
tb.main()