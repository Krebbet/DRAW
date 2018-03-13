# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import os


import numpy as np
import scipy.misc

#from trainer.stylize import stylize
#from trainer.model import model

#import trainer.model as model
import trainer.draw_and_classify as model


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

READ_N=5
WRITE_N=5
GLIMPSES = 10
Z_SIZE = 10


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

            
    return parser            

def main():
  parser = build_parser()
  options = parser.parse_args()    
  
  #dir = 'gs://d-draw'
  dir = ''
  
  # setup params for training
  ckpt_file=os.path.join(options.dir,options.model_name)
  log_dir=os.path.join(dir,'logs')
  print(ckpt_file)
  print(log_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)  
  
  
  # build model...
  draw_model = model.DRAW(  read_n = options.read_n,
                            write_n = options.write_n,
                            T = options.T,
                            z_size = options.z_size)
  

  # implements cost function
  cost = draw_model.loss()
  
  # creates training opp...
  train_op = draw_model.optimize(options.learning_rate)
  
  # Grab data....
  data_directory = os.path.join(options.dir, "mnist")
  if not os.path.exists(data_directory):
    os.makedirs(data_directory)
    
  print("data directory",data_directory)  

  train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data

  xtrain,ytrain =train_data.next_batch(options.batch_size) # xtrain is (batch_size x img_size)



  # Define the tensors to 'fetch' during a training step.
  fetches=[]
  fetches.extend([train_op])

  # start a session...
  sess=tf.InteractiveSession()
  

  saver = tf.train.Saver() # saves variables learned during training




  # initialize all the variables....
  tf.global_variables_initializer().run()
  #saver.restore(sess, "C:/TensorFlow/draw-master/data/drawmodel_attn.ckpt") # to restore from model, uncomment this line
  #saver.restore(sess, "data/drawmodel_attn.ckpt") # to restore from model, uncomment this line
  #saver.restore(sess, "drawmodel1.ckpt") # to restore from model, uncomment this line
  # create write and merge them together
  print('Creating writer at logs...')
  writer = tf.summary.FileWriter(os.path.join(log_dir , options.model_name), sess.graph) # for 1.0
  writer.add_graph(sess.graph)
  merged = tf.summary.merge_all()

  
  
  # train!!!
  print('begin training')
  for i in range(options.iterations):
  
    xtrain,ytrain =train_data.next_batch(options.batch_size) # xtrain is (batch_size x img_size)
    feed_dict={draw_model.x:xtrain,draw_model.y:ytrain}
    #[summary,results,y_pred,Lc1,Lc2]=sess.run([merged,fetches,draw_model.y_pred,draw_model.Lc1,draw_model.Lc2],feed_dict)
    [summary,results]=sess.run([merged,fetches],feed_dict)
    #print('y_pred shape:',y_pred.shape)
    #print('y_pred 1:',y_pred[1,:])
    #print('ytrain 1:',ytrain[1,:])
    #print('Lc1',Lc1)
    #print('Lc2',Lc2)
    writer.add_summary(summary, i)
    if i%100==0:
      print("iter=%d" % (i))
    if (i%options.checkpoint_iterations == 0):  
      saver.save(sess,ckpt_file, global_step=i)  

  ## TRAINING FINISHED ##
  print('training complete!')
  saver.save(sess,ckpt_file)  
  
  
  writer.close()
  sess.close()  
  print('Done training')


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    