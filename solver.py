import tensorflow as tf
import trainer.model as ml


















class Solver(object):


  def __init__(self, model, data, **kwargs):
  # we need a writer,
  # a writer object 
  # 
    self.model = model
    self.data = data
    
    
    
  
  
  def train(self):




# Check out the form of this data... see how it is used...

# What the fuck is this for?
fetches=[]
fetches.extend([train_op])


# start a session...
sess=tf.InteractiveSession()

saver = tf.train.Saver() # saves variables learned during training


# create write and merge them together
print('Creating writer at logs...')
writer = tf.summary.FileWriter("logs", sess.graph) # for 1.0
merged = tf.summary.merge_all()


# initialize all the variables....
tf.global_variables_initializer().run()
#saver.restore(sess, "C:/TensorFlow/draw-master/data/drawmodel_attn.ckpt") # to restore from model, uncomment this line
#saver.restore(sess, "data/drawmodel_attn.ckpt") # to restore from model, uncomment this line
#saver.restore(sess, "drawmodel1.ckpt") # to restore from model, uncomment this line

# train!!!

for i in range(train_iters):
  xtrain,_=train_data.next_batch(batch_size) # xtrain is (batch_size x img_size)
  feed_dict={x:xtrain}
  [results,summary]=sess.run([fetches,merged],feed_dict)
  writer.add_summary(summary, i)
  
  if i%50==0:
    print("iter=%d" % (i) )

## TRAINING FINISHED ##


sess.close()



