
# coding: utf-8

# In[ ]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import genfromtxt


# In[2]:

import tensorflow as tf

sess = tf.InteractiveSession()


# In[3]:

# initialize weights W as Variables. A Variable can be used and modified by TF
# Second index is number of output


def weight_variable(shape,stddev):
    initial = tf.truncated_normal(shape,stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape,stddev=1.0, dtype=tf.float32)
    ## Initializing as normal is dumb-sauce, don't do it bruv
    #initial = 0.1 + tf.truncated_normal(shape, stddev=0.05, dtype=tf.float32)
    return tf.Variable(initial)



# In[4]:

# x is a placeholder for spectra, using it we can input any number of spectra
# Specifically, 'None' indicates that no number of input spectra is set

spectra_length = 1024
num_categories = 32

x_spectra = tf.placeholder(tf.float32, [None,spectra_length])

y_ = tf.placeholder(tf.float32, [None,num_categories])


# In[5]:

# Initalize variables 


# Input to layer 1

nodes_1 = 1000 # number of neurons in first layer

W1 = weight_variable([ spectra_length, nodes_1 ], stddev = 1/np.sqrt(spectra_length))
b1 = bias_variable([nodes_1])




#### Layer N to layer M

#nodes_M = 10

#WM = weight_variable([ nodes_N, nodes_M ], stddev = 1/np.sqrt(nodes_N))
#bM = bias_variable([nodes_M],0)


nodes_2 = 500 # number of neurons in first layer

W2 = weight_variable([ nodes_1, nodes_2 ], stddev = 1/np.sqrt(nodes_1))
b2 = bias_variable([nodes_2])



nodes_3 = 500 # number of neurons in first layer

W3 = weight_variable([ nodes_2, nodes_3 ], stddev = 1/np.sqrt(nodes_2))
b3 = bias_variable([nodes_3])

# Layer 1 to Output


W_out = weight_variable([ nodes_2, num_categories ], stddev = 1/np.sqrt(nodes_2))
b_out = bias_variable([num_categories])


# Prepare saver to save model


saver = tf.train.Saver()
# In[6]:

# Create network structure


keep_prob = tf.placeholder(tf.float32)



N1 = tf.nn.tanh( tf.matmul(x_spectra, W1) + b1 )
N1_drop = tf.nn.dropout(N1, keep_prob)





# N_M = tf.nn.tanh( tf.matmul(N_N, N_M) + bM)

N2 = tf.nn.tanh( tf.matmul(N1_drop, W2) + b2 )
N2_drop = tf.nn.dropout(N2, keep_prob)


#N3 = tf.nn.tanh( tf.matmul(N2, W3) + b3 )



N_out = tf.nn.softmax( tf.matmul(N2 , W_out) + b_out )


# In[7]:

# Define cost function, optimizer for training 

scale_factor = 5.0
batch_size = 100000.0
L2_Reg = (scale_factor/batch_size)*( tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W_out)  )



#cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( tf.matmul(N1 , W_out) + b_out  , y_))+L2_Reg


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( tf.matmul(N2_drop , W_out) + b_out  , y_ ))+L2_Reg


learning_rate=0.01

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


# In[ ]:




# In[ ]:




# In[ ]:




# In[8]:




# In[9]:




# In[14]:

# Implement the model

def ANN_full(_spectra):
    
    N1 = tf.nn.tanh( tf.matmul(_spectra, W1) + b1 )
    N1_drop = tf.nn.dropout(N1, keep_prob)
    
    N2 = tf.nn.tanh( tf.matmul(N1      , W2) + b2 )
    N2_drop = tf.nn.dropout(N2, keep_prob)

    N3 = tf.nn.tanh( tf.matmul(N2      , W3) + b3 )
    
    N_out = tf.nn.softmax( tf.matmul(N2_drop , W_out) + b_out )

    return N_out


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[15]:

#### Start session ####
sess.run(tf.initialize_all_variables())




# In[ ]:




# In[ ]:




# In[16]:

split_fraction = 10


# In[17]:

batch_key = np.split(TrnData_key,split_fraction )
batch = np.split(TrnData,split_fraction,axis=0)


# In[18]:

iters = 2000

loss_train = np.zeros(iters)
loss_test = np.zeros(iters)

L2_all = np.zeros(iters)


# split Test data into 4 batches 
#batch = np.split(TestData,4,axis=1)
#batch_key = np.split(TestData_key,4)
    

for i in range(iters):

    train_step.run( feed_dict = {x_spectra: batch[i%split_fraction], y_: batch_key[i%split_fraction], keep_prob: 0.5} )   
    # train and record training loss
    loss_train[i] = sess.run(cross_entropy, feed_dict={x_spectra: batch[0][0:1000], y_: batch_key[0][0:1000], keep_prob: 1.0})        
    
    # record evaluate loss value
    loss_test[i] = sess.run(cross_entropy, feed_dict={x_spectra: TestData[0:1000], y_: TestData_key[0:1000], keep_prob: 1.0})

    L2_all[i] = sess.run(L2_Reg)

    print '\1b[2k\r',    
    print('Epoch %s of %s, train error is %s, test error is %s' %(i,iters,(loss_train-L2_all)[i], loss_test[i])),

    
    if (i>1000 and np.abs(loss_test[i-1]-loss_test[i]) < 1e-3):
        break
        
        
loss_train = loss_train[0:i]
loss_test = loss_test[0:i] 

#L2_all = L2_all[0:i]


# In[19]:


loss_train = loss_train[0:i]
loss_test = loss_test[0:i]

#L2_all = L2_all[0:i]


# In[20]:

plt.plot(loss_train[0:])
plt.plot(loss_train[0:],'--r')
plt.title('Train Error')
#plt.savefig('test_%s_%s/Train_Error.png' %(I ,J))


# In[ ]:




# In[22]:

plt.plot(L2_all[50:])


# In[ ]:




# In[ ]:




# In[40]:


#index = 16

#print TestData_key[index]

#derp1 = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: #(TestData.T[index]).reshape(1,1024) })[0]

#plt.plot(derp1)
#plt.figure()

#plt.plot(TestData.T[index])
#print np.argmax(derp1)


# In[ ]:




# In[1]:

#weird_error = np.empty([3,1])


# In[ ]:

#np.trim_zeros((TestData_key[:,2]).astype(int))


# In[24]:

#
#correct = 0

#err = 10

#for i in range(1000):
#    ANN_result_temp = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (TestData.T[i]).reshape(1,1024) })[0]
#    Data_temp = TestData_key[i]
    
    
#    correct += np.any(np.arange(Data_temp-err,Data_temp+err) == np.argmax(ANN_result_temp))
    
    


# In[25]:

#weird_error[0] = correct/1000.0


# In[ ]:




# In[ ]:




# In[ ]:




# In[48]:


#index2 = 55

#print TrnData_key[index2]

#derp2 = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (TrnData.T[index2]).reshape(1,1024) })[0]

#plt.plot(derp2)
#plt.figure()

#plt.plot(TrnData.T[index2])
#print np.argmax(derp2)


# In[ ]:




# In[49]:

#correct = 0

#err = 10

#for i in range(1000):
#    ANN_result_temp = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (TrnData.T[i]).reshape(1,1024) })[0]
#    Data_temp = TrnData_key[i]
    
    
#    correct += np.any(np.arange(Data_temp-err,Data_temp+err) == np.argmax(ANN_result_temp))
    
    


# In[50]:

#weird_error[1] = correct/1000.0 


# In[ ]:




# In[ ]:

#correct = 0

#err = 10

#for i in range(1000):
#    ANN_result_temp = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (ValData.T[i]).reshape(1,1024) })[0]
#    Data_temp = ValData_key[i]
    
    
#    correct += np.any(np.arange(Data_temp-err,Data_temp+err) == np.argmax(ANN_result_temp))
    
    


# In[ ]:

#weird_error[2] = correct/1000.0 


# In[ ]:




# In[ ]:




# In[ ]:

#np.savetxt('test_%s_%s/results.txt %(I ,J)', weird_error, delimiter=',')
#np.savetxt('test_%s_%s/results.txt' %(I ,J), loss_test, delimiter=',')
#np.savetxt('test_%s_%s/results2.txt' %(I ,J), loss_train, delimiter=',')
# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

#saver = tf.train.Saver()
#save_path = saver.save(sess, 'test_%s_%s/model.ckpt' %(I ,J) )


# In[ ]:

# close session to unallocate variables

