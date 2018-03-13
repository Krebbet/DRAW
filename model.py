
""""
Create a draw model!
"""

import tensorflow as tf
#from tensorflow.examples.tutorials import mnist
import numpy as np
import os


try:
  import trainer.layers as lyr
except ImportError:
  import layers as lyr
  


 # I need to remove batch size in the constructor to allow for more generalization...
class DRAW(object):
  """
   This creates a draw model architecture to be trained on Tensor flow
   A = input image x size
   B = input image Y size
   enc_size = Encoder LSTM size
   dec_sixe = Decoder LSTM size
   read_n = Read Filter patch size
   write_n = write filter patch size
   
  """
  
  def __init__(self, A = 28, B= 28,
              enc_size = 256,dec_size = 256,
              read_n = 5,write_n = 5,z_size=10,T=10,batch_size =100,
              read_attn = True, write_attn = True,execute = False):

              
              
    # save the model variables.....          
    self.A = A
    self.B = B
    self.enc_size = enc_size
    self.dec_size = dec_size
    self.read_n = read_n 
    self.write_n = write_n
    self.z_size = z_size
    self.T = T
    self.batch_size = batch_size
    self.read_attn = read_attn
    self.write_attn = write_attn
    
    self.img_size = A*B
    self.DO_SHARE = None
    self.execute = execute
    
    
    self.build()
    

    
  def build(self):
  

    ## BUILD MODEL ## 

    # this is our image placeholder input
    self.x = tf.placeholder(tf.float32,shape=(self.batch_size,self.img_size),name ='X') # input (batch_size * img_size)
    
    
    with tf.name_scope('Layers'):
      # we can build some layers we will repeat latter
      self.lstm_enc = tf.contrib.rnn.LSTMCell(self.enc_size, state_is_tuple=True) # encoder Op
      self.lstm_dec = tf.contrib.rnn.LSTMCell(self.dec_size, state_is_tuple=True) # decoder Op
      
      # create our red and write layers
      read = lyr.read_attn if self.read_attn == True else lyr.read_no_attn
      write = lyr.write_attn if self.write_attn == True else lyr.write_no_attn

      ## STATE VARIABLES ## 
      # Initiliaize states before we cascade through the glimpes
      
      T = self.T # just for ease
      cs=[0]*T # sequence of canvases
      mus,logsigmas,sigmas=[0]*T,[0]*T,[0]*T # gaussian params generated by SampleQ. We will need these for computing loss.
      
      if self.execute == True:
        rs = [0]*T
        ws = [0]*T
        wsf = [0]*T # sequence of writes
        rsf = [0]*T        
         
         
      # initial states
      h_dec_prev=tf.zeros((self.batch_size,self.dec_size))
      enc_state=self.lstm_enc.zero_state(self.batch_size, tf.float32)
      dec_state=self.lstm_dec.zero_state(self.batch_size, tf.float32)

    ## DRAW MODEL ## 

    # construct the unrolled computational graph
    for t in range(T):
        with tf.name_scope('Glimpse'):
          with tf.name_scope('canvas'):
            c_prev = tf.zeros((self.batch_size,self.img_size)) if t==0 else cs[t-1]
          with tf.name_scope('Read'):
            x_hat = self.x-tf.sigmoid(c_prev) # error image
            r = read(self,self.x,x_hat,h_dec_prev)
          with tf.name_scope('Z_Space'):  
            h_enc,enc_state = lyr.encode(self,enc_state,tf.concat([r,h_dec_prev], 1))
            z,mus[t],logsigmas[t],sigmas[t]= lyr.sampleQ(self,h_enc)
          with tf.name_scope('Write'):
            h_dec,dec_state = lyr.decode(self,dec_state,z)
            w = write(self,h_dec)
            cs[t]=c_prev + w# store results
            h_dec_prev=h_dec
            
            #output intermediate tensors for analysis in execute mode
            if self.execute == True:
              rs[t] = r
              ws[t] = w
            
        
        # after creating all the unique layers we reuse the weights     
        self.DO_SHARE=True # from now on, share variables
        
        
        # calculate and output Read and Write filters if in execute!
        if self.execute == True:
          Fx,Fy,gamma=  lyr.attn_window(self,"read",h_dec_prev)

          # sum the fn patches
          Fx = tf.reduce_sum(Fx, axis = 1)
          Fy = tf.reduce_sum(Fy, axis = 1)
          # turn into a map
          Fx = tf.reshape(Fx,[100,28,1])
          Fy = tf.reshape(Fy,[100,1,28])
          rsf[t] =tf.matmul(Fx,Fy)
          rsf[t] = tf.reshape(rsf[t],[self.batch_size,self.B*self.A])*tf.reshape(gamma,[-1,1])

          Fx,Fy,gamma= lyr.attn_window(self,"write",h_dec)
          # sum the fn patches
          Fx = tf.reduce_sum(Fx, axis = 1)
          Fy = tf.reduce_sum(Fy, axis = 1)
          # turn into a map
          Fx = tf.reshape(Fx,[100,28,1])
          Fy = tf.reshape(Fy,[100,1,28])
          wsf[t] = tf.matmul(Fx,Fy)
          wsf[t] = tf.reshape(wsf[t],[self.batch_size,self.B*self.A])#*tf.reshape(gamma,[-1,1])
        
        
        
        # cache needed operations
        self.cs = cs
        self.mus = mus
        self.logsigmas =logsigmas
        self.sigmas = sigmas
        
        if self.execute == True:
          self.rs = rs
          self.ws = ws
          self.wsf = wsf
          self.rsf = rsf
        
        
    return

  
  def loss(self):
  
    # create local prototypes for ease....
    cs = self.cs
    mus = self.mus
    logsigmas = self.logsigmas
    sigmas = self.sigmas
    
    T = self.T # just for ease
    with tf.name_scope('Train'):
      with tf.name_scope('Loss'):    
        with tf.name_scope('Lx'):
          # ****** Lx - IMAGE RECONSTRUCTION LOSS ****************    
          # reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
          x_recons=tf.nn.sigmoid(cs[-1])

          # after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
          Lx=tf.reduce_sum(lyr.binary_crossentropy(self.x,x_recons),1) # reconstruction term
          Lx=tf.reduce_mean(Lx)
        
        with tf.name_scope('Ly'):
          # ****** Lz - LATENT VARIABLE LOSS ****************
          kl_terms=[0]*T
          for t in range(T):
              mu2=tf.square(mus[t])
              sigma2=tf.square(sigmas[t])
              logsigma=logsigmas[t]
              kl_terms[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma,1)-.5 # each kl term is (1xminibatch)
          KL=tf.add_n(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T
          Lz=tf.reduce_mean(KL) # average over minibatches
        
        cost=Lx+Lz

      # Track Loss components!
      tf.summary.scalar("cost", cost)
      tf.summary.scalar("Lx", Lx)
      tf.summary.scalar("Lz", Lz)
      
      
    self.cost = cost
      
      
      
  def optimize(self,learning_rate):    
    ## OPTIMIZER ## 
    with tf.name_scope('Opt'):
      optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
      grads=optimizer.compute_gradients(self.cost)
      with tf.name_scope('Clip_Grads'):
        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
        train_op=optimizer.apply_gradients(grads)
        
    return train_op    
    

  
  
  
  
  #def inference(self,x):

  