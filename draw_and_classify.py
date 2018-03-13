
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
   
   h_pred_size = size of hidden layer in classification step after
      the attention model has done its job and produced a z vector.
   
  """
  
  def __init__(self, A = 28, B= 28,
              enc_size = 256,dec_size = 256,
              read_n = 5,write_n = 5,z_size=10,T=10,#batch_size =100,
              class_size = 10, h_pred_size = 100,
              read_attn = True, write_attn = True,execute = False,
              model_name = 'draw_default'):

              
              
    # save the model variables.....          
    self.A = A
    self.B = B
    self.enc_size = enc_size
    self.dec_size = dec_size
    self.read_n = read_n 
    self.write_n = write_n
    self.z_size = z_size
    self.T = T
    #self.batch_size = batch_size
    self.read_attn = read_attn
    self.write_attn = write_attn
    
    self.class_size = class_size 
    self.h_pred_size = h_pred_size
    
    self.img_size = A*B
    self.DO_SHARE = None
    self.execute = execute
    
    self.model_name = model_name
    
    self.build()
    self.loss()
    

    
  def build(self):
  

    ## BUILD MODEL ## 

    # this is our image placeholder input
    self.x = tf.placeholder(tf.float32,shape=[None,self.img_size],name ='X') # input (batch_size * img_size)
    self.batch_size = tf.shape(self.x)[0] 
    
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
      zs =  [0]*T # Latent variable output.
      
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
            c_prev = tf.zeros([self.batch_size,self.img_size]) if t==0 else cs[t-1]
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
            # I am storing this to analyze it at some point...
            zs[t] = z
            
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
        

    # use the outputs to classify the image as 1 of the 10 outputs...
    self.DO_SHARE=False
    with tf.name_scope('Classify'):
      with tf.variable_scope("h_pred",reuse=self.DO_SHARE):
        h_pred = lyr.linear(z,self.h_pred_size)
      with tf.variable_scope("y_pred",reuse=self.DO_SHARE):
        y_pred = lyr.linear(h_pred,self.class_size)
        
      self.scores = tf.nn.softmax(y_pred)

    self.DO_SHARE=True

    
    # cache needed operations
    self.cs = cs
    self.mus = mus
    self.logsigmas =logsigmas
    self.sigmas = sigmas
    self.zs = zs
    
    
    
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
        
        # We add in a classification error to turn the net into a classifier.
        with tf.name_scope('Lc'):
          # input the true answer placeholder...
          self.y = tf.placeholder(tf.float32,shape=(None,self.class_size),name ='Y') # input (batch_size * classifier size)
          
          #Lc=tf.reduce_sum(lyr.binary_crossentropy(self.y,self.y_pred),1) # reconstruction term
          #self.Lc1 = scores
          Lc = -tf.reduce_sum(self.y * tf.log(self.scores))
          
          # get predictions
          self.correct_prediction = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.y, 1))

          # %% And now we can look at the mean of our network's correct guesses
          self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
          

          
          #Lc=tf.reduce_mean(Lc)
          #self.Lc2 = Lc

        
        
        
        cost=Lx+Lz+Lc

      # Track Loss components!
      tf.summary.scalar("cost", cost)
      tf.summary.scalar("Lx", Lx)
      tf.summary.scalar("Lz", Lz)
      tf.summary.scalar("Lc", Lc)
      self.acc_sum = tf.summary.scalar("accuracy", self.accuracy)
      
    self.cost = cost
    return  
      
      
  def optimize(self,learning_rate):    
    ## OPTIMIZER ## 
    with tf.name_scope('Opt'):
      optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
      grads=optimizer.compute_gradients(self.cost)
      with tf.name_scope('Clip_Grads'):
        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
        self.train_op=optimizer.apply_gradients(grads)
        
    return     
    

  def test_acc(self,data,sess,writer,batch_size = 100):

    # grab next batch
    n_itt = data.num_examples // batch_size
    acc= []
    for i in range(n_itt):
    
      x,y =data.next_batch(batch_size, shuffle=False) # xtrain is (batch_size x img_size)
      feed_dict={self.x:x,self.y:y}      
     

      # evaluate batch and grab acc for it...
      [acc_sum]=sess.run([self.acc_sum],feed_dict)
      writer.add_summary(acc_sum, i)
      #acc.append(accuracy)
  
  
    # calc. total acc...
    #acc_final = np.mean(acc)
    return   
  
  #def draw(self,x,dir):
  def calc_accuracy(self,data,model_dir,batch_size = 100):
  
    # create data feed....
   
  
    with tf.Session() as sess:
      # initialize model....
      saver = tf.train.Saver() # saves variables learned during training
      saver.restore(sess, model_dir)      
      
      # grab next batch
      n_itt = data.num_examples // batch_size
      acc= []
      for i in range(n_itt):
      
        x,y =data.next_batch(batch_size, shuffle=False) # xtrain is (batch_size x img_size)
        feed_dict={self.x:x,self.y:y}      
       

        # evaluate batch and grab acc for it...
        [accuracy]=sess.run([self.accuracy],feed_dict)
        acc.append(accuracy)
    
    
      # calc. total acc...
      acc_final = np.mean(acc)
      return acc_final
    
  
  
  # should implement model proper for this...
  def inference(self,data,model_dir):
  
    with tf.Session() as sess:
    
      x,y =data.next_batch(100, shuffle=False) # xtrain is (batch_size x img_size)
      feed_dict={self.x:x,self.y:y}      
        
      # do evaluation on the inputs
      [scores]=sess.run([self.scores],feed_dict)
      
    #get predictions from scores  
    predictions = np.argmax(scores, 1)  
    print(predictions)
    return predictions
    
    

  
  
  def train(self,train_data,test_data,
        learning_rate = 1e-3,
        iterations = 10000,
        batch_size = 100, 
        ckpt_file = 'draw_default',
        log_dir = 'logs',
        checkpoint_iterations = 1000,
        test_acc_itr = 200):
  
  
  
    # get training op.. set learning rate...
    self.optimize(learning_rate)
    
    # set the training feed.
    fetches=[]
    fetches.extend([self.train_op])
    print('batch size =',batch_size)
    
    with tf.Session() as sess:

      print('initialize variables...')
      tf.global_variables_initializer().run()
  

      print('Creating writer at logs...')
      train_dir = os.path.join(log_dir, 'train')
      test_dir = os.path.join(log_dir, 'test')
      
      writer = tf.summary.FileWriter(os.path.join(train_dir , self.model_name), sess.graph) # for 1.0
      test_writer = tf.summary.FileWriter(os.path.join(test_dir , self.model_name), sess.graph) # for 1.0
      merged = tf.summary.merge_all()
      
      
      # setup a saver object to save model...
      saver = tf.train.Saver()
  
      print('Begin Training')
      for i in range(iterations):
        # Grab next batch of data
        xtrain,ytrain =train_data.next_batch(batch_size) # xtrain is (batch_size x img_size)
        
        feed_dict={self.x:xtrain,self.y:ytrain}      
        
        # do training on batch, return the summary and any results...
        [summary,results]=sess.run([merged,fetches],feed_dict)
        
        # check if we should test the test set acc.
        if i%test_acc_itr == 0:
          self.test_acc(test_data,sess,test_writer)
          print('test acc at',i)
        
        # write summary to writer
        writer.add_summary(summary, i)
        
        
        # do any print outs
        if i%500==0:
          print("iter=%d" % (i))
        
        #save model at check points
        if (i%checkpoint_iterations == 0):  
          saver.save(sess,ckpt_file, global_step=i)  
    
  
      # saves variables learned during training
      ## TRAINING FINISHED ##
      print('training complete!')
      saver.save(sess,ckpt_file)  
      
      test_writer.close()
      writer.close()
      sess.close()  



    print('Done training')
    return
  
  
  