
""""
This code contains all the functions that
build the various layers of the model.
"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os

eps = 1e-8


def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b



def filterbank(gx, gy, sigma2,delta, A,B,N):
    # create an N sized array [0,1...N-1] and cast it as a tensor vector
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])

    # mu_x,mu_y define the mean at each gaus filter origin
    # shape = (1,N)
    # Remember each of these guas filters will be applied on the 
    # entire row/col. SO the means must be defined at each origin 
    # point and then we apply it to all of A/B
    
    # remember delta is the step size along A/B 
    # It determines our patch size!
    mu_x = gx + (grid_i - N/2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N/2 - 0.5) * delta # eq 20

    # cast A and B as 1d vectors to apply our filters to...(for X,Y)
    # shape = (1,1,A)
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])

    # reshape our position quas position vectors for application
    # shape = (1,N,1)
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    
    # same for our variance (1,1,1)
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    
    # apply a gaus distribution along x and y...
    Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2))
    Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2)) # batch x N x B
    # the result is a shape of (1,N,A/B)
    # Where the elements of the filter are Fijk = exp(- (a_k -mu_j)^2 / (2*sig_i))
    # This gives us 5 seperate filters to apply to each position of a row A or B.
    # We will apply this to each row for Fx and each Col for Fy. The result
    # is a smooth filter over x.
        
        
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    
    return Fx,Fy


def attn_window(model,scope,h_dec):
    
    # grab local casts
    A,B = model.A,model.B
    if scope == 'read' :
      N = model.read_n
    else :
      N = model.write_n


    with tf.variable_scope(scope,reuse=model.DO_SHARE):
        params=linear(h_dec,5)
    # gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    # We have bounded our learned parameters by their domains
    # (gx,gy exist [0,1] and are scaled to [0,A/B]
    # sigma and delta are learned as the log forms so 
    # we have sigma,delta > 0
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(params,5,1)
    
    gx=(A+1)/2*(gx_+1)
    gy=(B+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N
    return filterbank(gx,gy,sigma2,delta,A,B,N)+(tf.exp(log_gamma),)

## READ ## 
def read_no_attn(model,x,x_hat,h_dec_prev):
    return tf.concat([x,x_hat], 1)

# Q: is this applied to a batch or a single image???
def read_attn(model,x,x_hat,h_dec_prev):

    # grab local casts
    A,B = model.A,model.B
    N = model.read_n

    # grab the filter banks and the intensity scalar gamma
    Fx,Fy,gamma=attn_window(model,"read",h_dec_prev)
    
    # take in an image and apply the filters to get an NxN Patch
    def filter_img(img,Fx,Fy,gamma,N):
        # grab the transpose
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        # reshape flattened image so we can do the filtering in 2d space.
        img=tf.reshape(img,[-1,B,A])
        # Fy x Fx^T
        glimpse=tf.matmul(Fy,tf.matmul(img,Fxt))
        # flatten back to a batch of 1d reps.
        glimpse=tf.reshape(glimpse,[-1,N*N])
        # scale by gamma and return (is the reshape nec...)
        return glimpse*tf.reshape(gamma,[-1,1])
        
    # apply the filter to each image and concatenate    
    x=filter_img(x,Fx,Fy,gamma,N) # batch x (read_n*read_n)
    x_hat=filter_img(x_hat,Fx,Fy,gamma,N)
    return tf.concat([x,x_hat], 1) # concat along feature axis

#read = read_attn if FLAGS.read_attn else read_no_attn

## ENCODE ## 
def encode(model,state,input):
    """
    run LSTM
    state = previous encoder state
    input = cat(read,h_dec_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("encoder",reuse=model.DO_SHARE):
        return model.lstm_enc(input,state)

## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##

def sampleQ(model,h_enc):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size)
    """
    with tf.name_scope("Q"):
      with tf.variable_scope("noise",reuse=model.DO_SHARE):
        e=tf.random_normal((model.batch_size,model.z_size), mean=0, stddev=1,name ='Qnoise') 
      with tf.variable_scope("mu",reuse=model.DO_SHARE):
        mu=linear(h_enc,model.z_size)
      with tf.variable_scope("sigma",reuse=model.DO_SHARE):
        logsigma=linear(h_enc,model.z_size)
        sigma=tf.exp(logsigma)

    return (mu + sigma*e, mu, logsigma, sigma)

## DECODER ## 
def decode(model,state,input):
    with tf.variable_scope("decoder",reuse=model.DO_SHARE):
        return model.lstm_dec(input, state)

## WRITER ## 
def write_no_attn(model,h_dec):
    with tf.variable_scope("write",reuse=model.DO_SHARE):
        return linear(h_dec,model.img_size)

def write_attn(model,h_dec):

    N = model.write_n
    A,B = model.A,model.B
    batch_size = model.batch_size
    
    with tf.variable_scope("writeW",reuse=model.DO_SHARE):
        w=linear(h_dec,N*N) # batch x (write_n*write_n)
    
    # here has to be a better way to do this...
    # Why do we need to create an entire output W window when
    # we are only wrting to the filter window...
    w=tf.reshape(w,[batch_size,N,N])
    Fx,Fy,gamma=attn_window(model,"write",h_dec)
    Fyt=tf.transpose(Fy,perm=[0,2,1])
    wr=tf.matmul(Fyt,tf.matmul(w,Fx))
    wr=tf.reshape(wr,[batch_size,B*A])
    #gamma=tf.tile(gamma,[1,B*A])
    
    # why the 1/gamma?
    return wr*tf.reshape(1.0/gamma,[-1,1])
    
    
    
    
    
    
    
    
    
    
    
    
####################
# Loss function
    
def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

    