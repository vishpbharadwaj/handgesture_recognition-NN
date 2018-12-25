
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py

import tensorflow.examples.tutorials.mnist.input_data as input_data

LOGDIR = "/home/vishnu/Dropbox/intel_works/nn_files/tf_tests/modular_py/log_com_gesture_cnn_mnist"



########################################################################################
def load_dataset():
    train_dataset = h5py.File('signs_dataset/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('signs_dataset/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

########################################################################################



########################################################################################
def committee_full_gesture_cnn(alpha,keeprobs):
 
    #tf.reset_default_graph()
    sess = tf.Session()
    
#input placeholders
    x = tf.placeholder(tf.float32, [None, train_x.shape[1],train_x.shape[2],train_x.shape[3]])
    y = tf.placeholder(tf.float32, [None, 10])
    #x = tf.cast(x,tf.float32) # casting as the original was uint8

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # param dicts
    com1_W_params = {}
    com1_B_params = {}
    com1_Z_params = {}
    com1_A_params = {}
    com1_Maxpool_params = {}

    com1_A_params["a0"] = x
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# CNN layers
    for iter_i in range(1,com1_num_cnn_layers+1,1):
        "('string_%d'%iter_1) is the syntax to make it dynamic"
        # convolution
        with tf.name_scope('convolution'):
            com1_W_params["w{0}".format(iter_i)] = tf.Variable( tf.truncated_normal([ com1_Fsize_params["fsize{0}".format(iter_i)], com1_Fsize_params["fsize{0}".format(iter_i)], com1_ChanNum_params["chan{0}".format(iter_i)], com1_Fnum_params["fnum{0}".format(iter_i)] ], stddev=0.1) )

            com1_B_params["b{0}".format(iter_i)] = tf.Variable( tf.constant(0.1, shape=[com1_Fnum_params["fnum{0}".format(iter_i)]]) )

            com1_Z_params["z{0}".format(iter_i)] = tf.nn.conv2d( com1_A_params["a{0}".format(iter_i-1)], com1_W_params["w{0}".format(iter_i)], com1_Stride_params["stride{0}".format(iter_i)], padding="VALID" ) + com1_B_params["b{0}".format(iter_i)]

        # maxpool
        if (com1_Maxpool_ops["pool{0}".format(iter_i)]) == True:
            with tf.name_scope('max_pool'):
                com1_Maxpool_params["max_pool{0}".format(iter_i)] = tf.nn.max_pool( com1_Z_params["z{0}".format(iter_i)], com1_Pool_Ksize_params["ksize{0}".format(iter_i)], com1_Pool_stride_params["kstride{0}".format(iter_i)], padding='VALID' )


        # relu
        with tf.name_scope('a_conv--relu'):
            if (com1_Maxpool_ops["pool{0}".format(iter_i)]) == True:
                com1_A_params["a{0}".format(iter_i)] = tf.nn.relu( com1_Maxpool_params["max_pool{0}".format(iter_i)], name=('a_conv%d'%iter_i) )
            else:
                com1_A_params["a{0}".format(iter_i)] = tf.nn.relu( com1_Z_params["z{0}".format(iter_i)], name=('a_conv%d'%iter_i) )

        
        # for Z and not after max pool and relu
        tf.summary.histogram("weights", com1_W_params["w{0}".format(iter_i)])
        tf.summary.histogram("biases", com1_B_params["b{0}".format(iter_i)])

        tf.summary.histogram("activations", com1_A_params["a{0}".format(iter_i)])
        
        print("\nshape after z_conv%d : "%iter_i)
        print( com1_Z_params["z{0}".format(iter_i)].shape ) 
        print("\n")
        
        # after pooling & relu
        print("\nshape after pool & relu a_conv%d : "%iter_i)
        print( com1_A_params["a{0}".format(iter_i)].shape ) 
        print("\n")
               
    
    """"""""""""""""""""""""""""""
    
# Fully connected layers
    
    #flattening for fc 
    with tf.name_scope('flattening'):
        com1_fc_flat = tf.reshape( com1_A_params["a{0}".format(com1_num_cnn_layers)], [-1, (tf.shape(com1_A_params["a{0}".format(com1_num_cnn_layers)])[1]) * (tf.shape(com1_A_params["a{0}".format(com1_num_cnn_layers)])[2]) * (tf.shape(com1_A_params["a{0}".format(com1_num_cnn_layers)])[3])] )

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #fully-connected layer 1:
    with tf.name_scope('fc1') as scope:
        W_fc1 = tf.Variable(tf.truncated_normal([com1_fc_shape_last_cnn, com1_n_fc1], stddev=0.1),name="w_fc1")
        b_fc1 = tf.Variable(tf.truncated_normal([com1_n_fc1], stddev=0.1),name="b_fc1")
        h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(com1_fc_flat, W_fc1) + b_fc1), keeprobs )
        
        tf.summary.histogram("weights", W_fc1)
        tf.summary.histogram("biases", b_fc1)
        tf.summary.histogram("activations", h_fc1)
        print("\nshape of fc1: ")
        print(h_fc1.shape)
        print("\n")
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
       
    #fully-connected layer 2:
    with tf.name_scope('fc2') as scope:
        W_fc2 = tf.Variable(tf.truncated_normal([com1_n_fc1, com1_n_fc2], stddev=0.1),name="w_fc2")
        b_fc2 = tf.Variable(tf.truncated_normal([com1_n_fc2], stddev=0.1),name="b_fc2")
        h_fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2), keeprobs )
        
        tf.summary.histogram("weights", W_fc2)
        tf.summary.histogram("biases", b_fc2)
        tf.summary.histogram("activations", h_fc2)
        print("\nshape of fc2: ")
        print(h_fc2.shape)
        print("\n")
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
    #fully-connected layer 3:
    with tf.name_scope('fc3') as scope:
        W_fc3 = tf.Variable(tf.truncated_normal([com1_n_fc2, com1_n_fc3], stddev=0.1),name="w_fc3")
        b_fc3 = tf.Variable(tf.truncated_normal([com1_n_fc3], stddev=0.1),name="b_fc3")
        h_fc3 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3), keeprobs )
        
        tf.summary.histogram("weights", W_fc3)
        tf.summary.histogram("biases", b_fc3)
        tf.summary.histogram("activations", h_fc3)
        print("\nshape of fc3: ")
        print(h_fc3.shape)
        print("\n")
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
     
    #softmax layer:
    with tf.name_scope('final_softmax') as scope:
        W_fc4 = tf.Variable(tf.truncated_normal([com1_n_fc3, 10], stddev=0.1),name="w_fc4")
        b_fc4 = tf.Variable(tf.truncated_normal([10], stddev=0.1),name="b_fc3")
        com1_y_pred = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)
        #print(com1_y_pred.get_shape())
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #loss functions:
    with tf.name_scope('com1_cross_entropy'):
        com1_cross_entropy = -tf.reduce_sum(y * tf.log(com1_y_pred))
        
        tf.summary.scalar("com1_cross_entropy", com1_cross_entropy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #training:
    with tf.name_scope('train'):    
        com1_optimizer = tf.train.AdamOptimizer(alpha).minimize(com1_cross_entropy)
        #optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(com1_cross_entropy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    #com1_accuracy:
    with tf.name_scope('com1_accuracy'):
        correct_prediction1 = tf.equal(tf.argmax(com1_y_pred, 1), tf.argmax(y, 1))
        com1_accuracy = tf.reduce_mean(tf.cast(correct_prediction1, 'float'))
        
        tf.summary.scalar("com1_accuracy", com1_accuracy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # param dicts
    com2_W_params = {}
    com2_B_params = {}
    com2_Z_params = {}
    com2_A_params = {}
    com2_Maxpool_params = {}

    com2_A_params["a0"] = x
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# CNN layers
    for iter_i in range(1,com2_num_cnn_layers+1,1):
        "('string_%d'%iter_1) is the syntax to make it dynamic"
        # convolution
        with tf.name_scope('convolution'):
            com2_W_params["w{0}".format(iter_i)] = tf.Variable( tf.truncated_normal([ com2_Fsize_params["fsize{0}".format(iter_i)], com2_Fsize_params["fsize{0}".format(iter_i)], com2_ChanNum_params["chan{0}".format(iter_i)], com2_Fnum_params["fnum{0}".format(iter_i)] ], stddev=0.1) )

            com2_B_params["b{0}".format(iter_i)] = tf.Variable( tf.constant(0.1, shape=[com2_Fnum_params["fnum{0}".format(iter_i)]]) )

            com2_Z_params["z{0}".format(iter_i)] = tf.nn.conv2d( com2_A_params["a{0}".format(iter_i-1)], com2_W_params["w{0}".format(iter_i)], com2_Stride_params["stride{0}".format(iter_i)], padding="VALID" ) + com2_B_params["b{0}".format(iter_i)]

        # maxpool
        if (com2_Maxpool_ops["pool{0}".format(iter_i)]) == True:
            with tf.name_scope('max_pool'):
                com2_Maxpool_params["max_pool{0}".format(iter_i)] = tf.nn.max_pool( com2_Z_params["z{0}".format(iter_i)], com2_Pool_Ksize_params["ksize{0}".format(iter_i)], com2_Pool_stride_params["kstride{0}".format(iter_i)], padding='VALID' )


        # relu
        with tf.name_scope('a_conv--relu'):
            if (com2_Maxpool_ops["pool{0}".format(iter_i)]) == True:
                com2_A_params["a{0}".format(iter_i)] = tf.nn.relu( com2_Maxpool_params["max_pool{0}".format(iter_i)], name=('a_conv%d'%iter_i) )
            else:
                com2_A_params["a{0}".format(iter_i)] = tf.nn.relu( com2_Z_params["z{0}".format(iter_i)], name=('a_conv%d'%iter_i) )

        
        # for Z and not after max pool and relu
        tf.summary.histogram("weights", com2_W_params["w{0}".format(iter_i)])
        tf.summary.histogram("biases", com2_B_params["b{0}".format(iter_i)])

        tf.summary.histogram("activations", com2_A_params["a{0}".format(iter_i)])
        
        print("\nshape after z_conv%d : "%iter_i)
        print( com2_Z_params["z{0}".format(iter_i)].shape ) 
        print("\n")
        
        # after pooling & relu
        print("\nshape after pool & relu a_conv%d : "%iter_i)
        print( com2_A_params["a{0}".format(iter_i)].shape ) 
        print("\n")
               
    
    """"""""""""""""""""""""""""""
    
# Fully connected layers
    
    #flattening for fc 
    with tf.name_scope('flattening'):
        com2_fc_flat = tf.reshape( com2_A_params["a{0}".format(com2_num_cnn_layers)], [-1, (tf.shape(com2_A_params["a{0}".format(com2_num_cnn_layers)])[1]) * (tf.shape(com2_A_params["a{0}".format(com2_num_cnn_layers)])[2]) * (tf.shape(com2_A_params["a{0}".format(com2_num_cnn_layers)])[3])] )

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #fully-connected layer 1:
    with tf.name_scope('fc1') as scope:
        W_fc1 = tf.Variable(tf.truncated_normal([com2_fc_shape_last_cnn, com2_n_fc1], stddev=0.1),name="w_fc1")
        b_fc1 = tf.Variable(tf.truncated_normal([com2_n_fc1], stddev=0.1),name="b_fc1")
        h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(com2_fc_flat, W_fc1) + b_fc1), keeprobs )
        
        tf.summary.histogram("weights", W_fc1)
        tf.summary.histogram("biases", b_fc1)
        tf.summary.histogram("activations", h_fc1)
        print("\nshape of fc1: ")
        print(h_fc1.shape)
        print("\n")
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
       
    #fully-connected layer 2:
    with tf.name_scope('fc2') as scope:
        W_fc2 = tf.Variable(tf.truncated_normal([com2_n_fc1, com2_n_fc2], stddev=0.1),name="w_fc2")
        b_fc2 = tf.Variable(tf.truncated_normal([com2_n_fc2], stddev=0.1),name="b_fc2")
        h_fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2), keeprobs )
        
        tf.summary.histogram("weights", W_fc2)
        tf.summary.histogram("biases", b_fc2)
        tf.summary.histogram("activations", h_fc2)
        print("\nshape of fc2: ")
        print(h_fc2.shape)
        print("\n")
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
    #fully-connected layer 3:
    with tf.name_scope('fc3') as scope:
        W_fc3 = tf.Variable(tf.truncated_normal([com2_n_fc2, com2_n_fc3], stddev=0.1),name="w_fc3")
        b_fc3 = tf.Variable(tf.truncated_normal([com2_n_fc3], stddev=0.1),name="b_fc3")
        h_fc3 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3), keeprobs )
        
        tf.summary.histogram("weights", W_fc3)
        tf.summary.histogram("biases", b_fc3)
        tf.summary.histogram("activations", h_fc3)
        print("\nshape of fc3: ")
        print(h_fc3.shape)
        print("\n")
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
     
    #softmax layer:
    with tf.name_scope('final_softmax') as scope:
        W_fc4 = tf.Variable(tf.truncated_normal([com2_n_fc3, 10], stddev=0.1),name="w_fc4")
        b_fc4 = tf.Variable(tf.truncated_normal([10], stddev=0.1),name="b_fc3")
        com2_y_pred = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)
        #print(com2_y_pred.get_shape())
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #loss functions:
    with tf.name_scope('com2_cross_entropy'):
        com2_cross_entropy = -tf.reduce_sum(y * tf.log(com2_y_pred))
        
        tf.summary.scalar("com2_cross_entropy", com2_cross_entropy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #training:
    with tf.name_scope('train'):    
        com2_optimizer = tf.train.AdamOptimizer(alpha).minimize(com2_cross_entropy)
        #optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(com2_cross_entropy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    #com2_accuracy:
    with tf.name_scope('com2_accuracy'):
        correct_prediction2 = tf.equal(tf.argmax(com2_y_pred, 1), tf.argmax(y, 1))
        com2_accuracy = tf.reduce_mean(tf.cast(correct_prediction2, 'float'))
        
        tf.summary.scalar("com2_accuracy", com2_accuracy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    with tf.name_scope('full_accuracy'):
        y_pred = tf.add(com1_y_pred,com2_y_pred) / tf.constant(2.0, shape=[1,1]) 
        full_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        full_accuracy = tf.reduce_mean(tf.cast(full_prediction, 'float'))
        
        tf.summary.scalar("full_accuracy", full_accuracy)
 
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    summ = tf.summary.merge_all()
    
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #write here batch size and runs
    ep = 15
    batch_size = 400
    
    for i in range(ep):
        for batch_i in range(train_x.shape[0] // batch_size):
            if batch_i % 1 == 0:
                batch_xs, batch_ys = train_x[(batch_i+batch_i*i):batch_size,:,:,:], train_y[(batch_i+batch_i*i):batch_size,:]
                train_accuracy1 = sess.run(com1_accuracy, feed_dict={x:batch_xs, y:batch_ys})
                train_accuracy2 = sess.run(com2_accuracy, feed_dict={x:batch_xs, y:batch_ys})
                full_accuracy_2 = sess.run(full_accuracy, feed_dict={x:batch_xs, y:batch_ys})
                
                print(train_accuracy1)
                print(train_accuracy2)
                print(full_accuracy_2)
                print("\n")
                #s = sess.run(summ, feed_dict={x:batch_xs, y:batch_ys})
                sess.run(com1_optimizer, feed_dict={x:batch_xs, y:batch_ys})
                sess.run(com2_optimizer, feed_dict={x:batch_xs, y:batch_ys})
                
                #writer.add_summary(s, i)


    print(">>>>>>>>>>>>>>>>>>>>>>>>>>> testing phase <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # test accuracy
    #for num_i in  range(test_x.shape[0]):
    test_accuracy = sess.run(full_accuracy, feed_dict={x:test_x, y:test_y})
    print(test_accuracy)


########################################################################################


##############################################
"""
Inputs X and Y def:
"""
train_img_size = 10000
test_img_size = 1000

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

train_x =  mnist.train.images[1:train_img_size,:]
test_x =  mnist.test.images[1:test_img_size,:]

train_x = train_x.reshape(train_x.shape[0], 28,28,1)
train_y = mnist.train.labels[1:train_img_size,:] 

test_x = test_x.reshape(test_x.shape[0], 28,28,1)
test_y = mnist.test.labels[1:test_img_size,:]

train_x = np.pad(train_x,((0,0),(36,0),(36,0),(0,0)),'constant')
test_x = np.pad(test_x,((0,0),(36,0),(36,0),(0,0)),'constant')

classes = 10

#train_x, train_y, test_x, test_y, classes = load_dataset()
##index = 1079
##plt.imshow(train_set_x_orig[index])
#print("\n\nNo. of Training samples (batch): %d"%train_x.shape[0])
#print("No. of Test samples: %d"%test_x.shape[0])
#print("No. of gestures: %d"%classes.size)
#print("Image format (Len, Wid, Chan): %d * %d * %d"%(train_x.shape[1],train_x.shape[2],train_x.shape[3]))
#
#train_x = train_x/255
#test_x = test_x/255
#
#y_org = np.eye(6)[train_y] #converting it into one-hot encoding  # this is equal to y
#y_org = np.reshape(y_org,[y_org.shape[1],y_org.shape[2]])
#
#y_val = np.eye(6)[test_y] #converting it into one-hot encoding   # this is equal to y_test
#y_val = np.reshape(y_val,[y_val.shape[1],y_val.shape[2]])

##############################################


############################# Com1 params ##################################################
com1_ChanNum_params = {}
com1_Fsize_params = {}
com1_Fnum_params = {}
com1_Stride_params = {}
com1_Maxpool_ops = {}
com1_Pool_Ksize_params ={}
com1_Pool_stride_params = {}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
### CNN layer params

com1_num_cnn_layers = 3

#layer 1 --> conv, maxpool & relu
com1_ChanNum_params["chan1"] = tf.cast(train_x.shape[3],tf.int32)
com1_Fsize_params["fsize1"] = 3 # filter size of n*n for convolution
com1_Fnum_params["fnum1"] = 6 # No. of filters required
com1_Stride_params["stride1"] = [1,2,2,1]
com1_Maxpool_ops["pool1"] = False #change here if pooling not required
if com1_Maxpool_ops["pool1"] == True:
    com1_Pool_Ksize_params["ksize1"] = [1,2,2,1]
    com1_Pool_stride_params["kstride1"] = [1,2,2,1]
#layer 1 ends 

#layer 2 --> conv, maxpool & relu
com1_ChanNum_params["chan2"] = com1_Fnum_params["fnum1"]
com1_Fsize_params["fsize2"] = 5 
com1_Fnum_params["fnum2"] = 10 
com1_Stride_params["stride2"] = [1,1,1,1]
com1_Maxpool_ops["pool2"] = True
if com1_Maxpool_ops["pool2"] == True:
    com1_Pool_Ksize_params["ksize2"] = [1,2,2,1]
    com1_Pool_stride_params["kstride2"] = [1,2,2,1]
#layer 2 ends 

#layer 3 --> conv, maxpool & relu
com1_ChanNum_params["chan3"] = com1_Fnum_params["fnum2"]
com1_Fsize_params["fsize3"] = 5 
com1_Fnum_params["fnum3"] = 16 
com1_Stride_params["stride3"] = [1,1,1,1] 
com1_Maxpool_ops["pool3"] = True
if com1_Maxpool_ops["pool3"] == True:
    com1_Pool_Ksize_params["ksize3"] = [1,2,2,1]
    com1_Pool_stride_params["kstride3"] = [1,2,2,1]
#layer 3 ends 

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
### FC layer params
com1_num_fc_layers = 2
com1_fc_shape_last_cnn = 256 #"this is done because tf.reshape had some issues"

#fc layer 1
com1_n_fc1 = 128

#fc layer 2
com1_n_fc2 = 64

#fc layer 3
com1_n_fc3 = 10#classes.size

### FC layer params ends


########################################################################################

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

############################# Com2 params ##################################################
com2_ChanNum_params = {}
com2_Fsize_params = {}
com2_Fnum_params = {}
com2_Stride_params = {}
com2_Maxpool_ops = {}
com2_Pool_Ksize_params ={}
com2_Pool_stride_params = {}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
### CNN layer params

com2_num_cnn_layers = 2

#layer 1 --> conv, maxpool & relu
com2_ChanNum_params["chan1"] = tf.cast(train_x.shape[3],tf.int32)
com2_Fsize_params["fsize1"] = 5 # filter size of n*n for convolution
com2_Fnum_params["fnum1"] = 6 # No. of filters required
com2_Stride_params["stride1"] = [1,2,2,1]
com2_Maxpool_ops["pool1"] = True #change here if pooling not required
if com2_Maxpool_ops["pool1"] == True:
    com2_Pool_Ksize_params["ksize1"] = [1,2,2,1]
    com2_Pool_stride_params["kstride1"] = [1,2,2,1]
#layer 1 ends 

#layer 2 --> conv, maxpool & relu
com2_ChanNum_params["chan2"] = com2_Fnum_params["fnum1"]
com2_Fsize_params["fsize2"] = 5 
com2_Fnum_params["fnum2"] = 16 
com2_Stride_params["stride2"] = [1,1,1,1]
com2_Maxpool_ops["pool2"] = True
if com2_Maxpool_ops["pool2"] == True:
    com2_Pool_Ksize_params["ksize2"] = [1,2,2,1]
    com2_Pool_stride_params["kstride2"] = [1,2,2,1]
#layer 2 ends 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
### FC layer params
com2_num_fc_layers = 2
com2_fc_shape_last_cnn = 400 #"this is done because tf.reshape had some issues"

#fc layer 1
com2_n_fc1 = 128

#fc layer 2
com2_n_fc2 = 64

#fc layer 3
com2_n_fc3 = 10#classes.size

### FC layer params ends



########################################################################################


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#call the model here with alpha and multiple runs with for loop to change alpha
committee_full_gesture_cnn(alpha=0.0005, keeprobs=0.9)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
