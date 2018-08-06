import argparse #Argument parsing
#Retrieve command line options
parser = argparse.ArgumentParser()
parser.add_argument('tensorboard_path', help = 'Path to tensorboard directory')
args = parser.parse_args()

import tensorflow as tf
from tensorboard import main as tb
import json

print("Tensorboard path: " + args.tensorboard_path)
tf.flags.FLAGS.logdir =  args.tensorboard_path
tb.main()