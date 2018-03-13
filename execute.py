# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import os


import numpy as np
import scipy.misc

#from trainer.stylize import stylize
#from trainer.model import model
import model


import math
from argparse import ArgumentParser
import io


import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.examples.tutorials import mnist




ITERATIONS = 10000
BATCH_SIZE = 100
A_SIZE = 28
B_SIZE = 28
LEARNING_RATE = 1e-3 
CHECKPOINT_ITERATIONS = 1000 # when to save a checkpoint model...

MODEL_NAME = 'draw'

READ_N=5
WRITE_N=5
GLIMPSES = 10
Z_SIZE = 10

def build_parser():
    # builds the command line options....
    parser = ArgumentParser()
    # this is the 'content image' dir for the arg line
    parser.add_argument('--job-dir',
            dest='dir', help='Job Directory',
            metavar='JOBDIR', required=True)


    parser.add_argument('--model-name',
            dest='model_name', help='Name of your model',
            metavar='MODEL_NAME', default = MODEL_NAME)             
            
    parser.add_argument('--batch-size',type=int,
            dest='batch_size', help='Data chunks while training',
            metavar='BATCH_SIZE',default = BATCH_SIZE)            
            
            
       
   


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
            
            
   
    return parser            

def main():
  parser = build_parser()
  options = parser.parse_args()    
  
  dir = 'C:\TensorFlow\draw-cloud\models'

  ckpt_file=os.path.join(options.dir,options.model_name)
  ckpt_file = '%s/%s' % (options.dir,options.model_name)
  print(ckpt_file)  
  
  print('r =',options.read_n)
  print('w =',options.read_n)
  print('T =',options.T)
  print('z =',options.z_size)
  
  
  # build model...
  draw_model = model.DRAW( read_n = options.read_n,
                            write_n = options.write_n,
                            T = options.T,
                            z_size = options.z_size,
                            execute = True)
                            
  cost = draw_model.loss()
  #train_op = draw_model.optimize(options.learning_rate)
  
  # Grab data....
  data_dir = ''
  data_directory = os.path.join(data_dir, "mnist")
  if not os.path.exists(data_directory):
    os.makedirs(data_directory)

  data = mnist.input_data.read_data_sets(data_directory, one_hot=True).test # binarized (0-1) mnist data



  # Define the tensors to 'fetch' during a training step.

  # start a session...
  sess=tf.InteractiveSession()
  
  # GRAB MODEL
  saver = tf.train.Saver() # saves variables learned during training
  saver.restore(sess, ckpt_file)
  
  xdata,_=data.next_batch(options.batch_size) # xtrain is (batch_size x img_size)
  feed_dict={draw_model.x:xdata}
  
  
  # define the outputs I want...
  fetch = [draw_model.cs,draw_model.rs,draw_model.ws,draw_model.rsf,draw_model.wsf]

  # Compute outputs
  print('*****EXECUTE********')
  [canvases,read_c,write_c,read_field,write_field] = sess.run(fetch,feed_dict) # generate some examples
  #canvases=np.array(canvases) # T x batch x img_size
  print('*****DONE********')
  # save them...
  out_dir='C:\TensorFlow\draw-cloud\out'
  out_file = os.path.join(out_dir, '%s%s' % (options.model_name,'.npz'))
  print(out_file)
  np.savez(out_file,c = canvases, r= read_c,w = write_c,rf = read_field, wf=write_field, input = xdata)

  print("Outputs saved in file: %s" % out_file)
  sess.close()  
  print('finished program')

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    