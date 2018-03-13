# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import os


import numpy as np
import scipy.misc

#from trainer.stylize import stylize
#from trainer.model import model

#import trainer.model as model
try:
  import trainer.draw_and_classify as model
except ImportError:
  import draw_and_classify as model




import math
from argparse import ArgumentParser
import io


import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.examples.tutorials import mnist

from data_utils import get_CIFAR10_data



ITERATIONS = 10000
BATCH_SIZE = 100
A_SIZE = 28
B_SIZE = 28
LEARNING_RATE = 1e-3 
CHECKPOINT_ITERATIONS = 1000 # when to save a checkpoint model...

READ_N=5
WRITE_N=5
GLIMPSES = 10
Z_SIZE = 10
C_HID_SIZE = 100

LOGDIR = ''
MODEL_NAME = 'draw'


def build_parser():
    # builds the command line options....
    parser = ArgumentParser()
    # this is the 'content image' dir for the arg line
    parser.add_argument('--job-dir',
            dest='dir', help='Job Directory',
            metavar='JOBDIR', required=False)
            
            
    parser.add_argument('--batch-size',type=int,
            dest='batch_size', help='Data chunks while training',
            metavar='BATCH_SIZE',default = BATCH_SIZE)            
            
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)            
    
    parser.add_argument('--checkpoint-itts', type=int,
            dest='checkpoint_iterations', help='model checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS',default = CHECKPOINT_ITERATIONS)
            
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)             
    
    
    parser.add_argument('--model-name',
            dest='model_name', help='Name of your model',
            metavar='MODEL_NAME', default = MODEL_NAME)   


    parser.add_argument('--A', type=int,
            dest='A', help='width of input images',
            metavar='A_SIZE',default = A_SIZE)

    parser.add_argument('--B', type=int,
            dest='B', help='height of input images',
            metavar='B_SIZE',default = B_SIZE)
            
    parser.add_argument('--read-n', type=int,
            dest='read_n', help='read patches',
            metavar='READ_N',default = READ_N)

    parser.add_argument('--write-n', type=int,
            dest='write_n', help='write patches',
            metavar='WRITE_N',default = READ_N)
            
    parser.add_argument('--T', type=int,
            dest='T', help='glimpses at the input image',
            metavar='GLIMPSES',default = GLIMPSES)
            
    parser.add_argument('--z-size', type=int,
            dest='z_size', help='dimension of latent variable',
            metavar='Z_SIZE',default = Z_SIZE)     

    parser.add_argument('--class-hidden-size', type=int,
            dest='class_hidden_size', help='Size of the hidden layer in the classifier',
            metavar='C_HID_SIZE',default = C_HID_SIZE)               

            
    parser.add_argument('--logdir', 
            dest='logdir', help='Directory to place Tensor Board logs',
            metavar='LOGDIR', default = LOGDIR)   
            
    return parser            

def main():
  parser = build_parser()
  options = parser.parse_args()    
  

  #print('Load CIFAR10_data')
  #data = get_CIFAR10_data()
  #print('Load success!')
  
  # setup params for training
  ckpt_file=os.path.join(options.dir,options.model_name)
  log_dir=os.path.join(options.logdir,'logs')
  print(ckpt_file)
  print(log_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)  
  
  
  # build model...
  print('Building Model...')
  draw_model = model.DRAW(  read_n = options.read_n,
                            write_n = options.write_n,
                            T = options.T,
                            z_size = options.z_size,
                            h_pred_size = options.class_hidden_size,
                            model_name = options.model_name)
  # get data
  print('Importing Data...')
  data_directory = os.path.join(options.dir, "mnist")
  if not os.path.exists(data_directory):
    os.makedirs(data_directory)
    
  print("data directory:",data_directory)  
  train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data  
  test_data =  mnist.input_data.read_data_sets(data_directory, one_hot=True).test
  val_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).validation

  #print('***********************************')#
  #print(train_data.num_examples)
  #print(test_data.num_examples)
  #print(val_data.num_examples)
  
  #train model
  print('Train model...')
  draw_model.train(train_data,test_data,
      learning_rate = options.learning_rate,
      iterations = options.iterations,
      batch_size = options.batch_size,
      ckpt_file = ckpt_file,
      log_dir = log_dir,
      checkpoint_iterations = options.checkpoint_iterations)  

  # classify test set...
  
  print('Get validation accuracy...')
  # get accuracy for test set...
  #ckpt_file = 'C:\TensorFlow\draw-cloud\models\draw_r5w5z10t10_3_inference'
  print(ckpt_file)
  acc = draw_model.calc_accuracy(val_data,model_dir = ckpt_file)
  print('returned acc:',acc)
  
  

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    