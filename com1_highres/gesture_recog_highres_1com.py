import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py
from scipy import ndimage

from cnn_utils import *

LOGDIR = "/home/vishnu/Documents/nn_files/tf_tests/com1_highres/log_1com_gesture_cnn"


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
########################################################################################
def gesture_recog_cnn_1com(train_x, train_y, test_x, test_y, classes, alpha, keeprobs):
 
    #tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
#input placeholders
    x = tf.placeholder(tf.float32, [None, train_x.shape[1],train_x.shape[2],train_x.shape[3]])
    y = tf.placeholder(tf.float32, [None, classes.size])
    #x = tf.cast(x,tf.float32) # casting as the original was uint8



    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print("\n --------------- com params ------------------\n")
    # param dicts
    com1_W_params = {}
    com1_B_params = {}
    com1_Z_params = {}
    com1_A_params = {}
    com1_Maxpool_params = {}

    com1_A_params["a0"] = x
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# CNN layers
    for iter_i in range(1,num_cnn_layers+1,1):
        "('string_%d'%iter_1) is the syntax to make it dynamic"
        # convolution
        with tf.name_scope('convolution_com1'):
            com1_W_params["w{0}".format(iter_i)] = tf.Variable( tf.truncated_normal([ com1_Fsize_params["fsize{0}".format(iter_i)], com1_Fsize_params["fsize{0}".format(iter_i)], com1_ChanNum_params["chan{0}".format(iter_i)], com1_Fnum_params["fnum{0}".format(iter_i)] ], stddev=0.1) )

            com1_B_params["b{0}".format(iter_i)] = tf.Variable( tf.constant(0.1, shape=[com1_Fnum_params["fnum{0}".format(iter_i)]]) )

            com1_Z_params["z{0}".format(iter_i)] = tf.nn.conv2d( com1_A_params["a{0}".format(iter_i-1)], com1_W_params["w{0}".format(iter_i)], com1_Stride_params["stride{0}".format(iter_i)], padding="VALID" ) + com1_B_params["b{0}".format(iter_i)]

        # maxpool
        if (com1_Maxpool_ops["pool{0}".format(iter_i)]) == True:
            with tf.name_scope('max_pool_com1'):
                com1_Maxpool_params["max_pool{0}".format(iter_i)] = tf.nn.max_pool( com1_Z_params["z{0}".format(iter_i)], com1_Pool_Ksize_params["ksize{0}".format(iter_i)], com1_Pool_stride_params["kstride{0}".format(iter_i)], padding='VALID' )


        # relu
        with tf.name_scope('a_conv--relu_com1'):
            if (com1_Maxpool_ops["pool{0}".format(iter_i)]) == True:
                com1_A_params["a{0}".format(iter_i)] = tf.nn.relu( com1_Maxpool_params["max_pool{0}".format(iter_i)], name=('a_conv%d'%iter_i) )
            else:
                com1_A_params["a{0}".format(iter_i)] = tf.nn.relu( com1_Z_params["z{0}".format(iter_i)], name=('a_conv%d'%iter_i) )

        
        # for Z and not after max pool and relu
        tf.summary.histogram("weights_com1", com1_W_params["w{0}".format(iter_i)])
        tf.summary.histogram("biases_com1", com1_B_params["b{0}".format(iter_i)])

        tf.summary.histogram("activations_com1", com1_A_params["a{0}".format(iter_i)])
        
        # Visualize conv kernels
        #grid = put_conv_img_ongrid(com1_A_params["a{0}".format(iter_i)])
        #print(grid.shape)
        tf.summary.image('com1_kernal', com1_A_params["a{0}".format(iter_i)][:,:,:,0:1], max_outputs=1)

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
    with tf.name_scope('flattening_com1'):
        com1_fc_flat = tf.reshape( com1_A_params["a{0}".format(num_cnn_layers)], [-1, (tf.shape(com1_A_params["a{0}".format(num_cnn_layers)])[1]) * (tf.shape(com1_A_params["a{0}".format(num_cnn_layers)])[2]) * (tf.shape(com1_A_params["a{0}".format(num_cnn_layers)])[3])] )

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #fully-connected layer 1:
    with tf.name_scope('fc1_com1') as scope:
        W_fc1 = tf.Variable(tf.truncated_normal([com1_fc_shape_last_cnn, com1_n_fc1], stddev=0.1),name="w_fc1")
        b_fc1 = tf.Variable(tf.truncated_normal([com1_n_fc1], stddev=0.1),name="b_fc1")
        z_fc1 = tf.nn.relu(tf.matmul(com1_fc_flat, W_fc1) + b_fc1)
        h_fc1 = tf.nn.dropout(z_fc1, keeprobs )
        
        tf.summary.histogram("weights_com1", W_fc1)
        tf.summary.histogram("biases_com1", b_fc1)
        tf.summary.histogram("activations_com1", h_fc1)
        print("\nshape of fc1: ")
        print(h_fc1.shape)
        print("\n")
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
       
    #fully-connected layer 2:
    with tf.name_scope('fc2_com1') as scope:
        W_fc2 = tf.Variable(tf.truncated_normal([com1_n_fc1, com1_n_fc2], stddev=0.1),name="w_fc2")
        b_fc2 = tf.Variable(tf.truncated_normal([com1_n_fc2], stddev=0.1),name="b_fc2")
        z_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc2 = tf.nn.dropout(z_fc2, keeprobs )
        
        tf.summary.histogram("weights_com1", W_fc2)
        tf.summary.histogram("biases_com1", b_fc2)
        tf.summary.histogram("activations_com1", h_fc2)
        print("\nshape of fc2: ")
        print(h_fc2.shape)
        print("\n")
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
    #fully-connected layer 3 - softmax:
    with tf.name_scope('fc3-softmax_com1') as scope:
        W_fc3 = tf.Variable(tf.truncated_normal([com1_n_fc2, com1_n_fc3], stddev=0.1),name="w_fc3")
        b_fc3 = tf.Variable(tf.truncated_normal([com1_n_fc3], stddev=0.1),name="b_fc3")
        #h_fc3 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3), keeprobs )
        z_y = tf.matmul(h_fc2, W_fc3) + b_fc3
        com1_y_pred = tf.nn.softmax(z_y)
        
        tf.summary.histogram("weights_com1", W_fc3)
        tf.summary.histogram("biases_com1", b_fc3)
        tf.summary.histogram("activations_com1", com1_y_pred)
        print("\nshape of fc3-softmax: ")
        print(com1_y_pred.shape)
        print("\n")
 
 
    #loss functions:
    with tf.name_scope('cross_entropy_com1'):
        com1_cross_entropy = -tf.reduce_sum(y * tf.log(com1_y_pred))
        
        tf.summary.scalar("com1_cross_entropy", com1_cross_entropy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #training:
    with tf.name_scope('train_com1'):    
        com1_optimizer = tf.train.AdamOptimizer(alpha).minimize(com1_cross_entropy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    #com1_accuracy:
    with tf.name_scope('accuracy_com1'):
        correct_prediction1 = tf.equal(tf.argmax(com1_y_pred, 1), tf.argmax(y, 1))
        com1_accuracy = tf.reduce_mean(tf.cast(correct_prediction1, 'float'))
        
        tf.summary.scalar("com1_accuracy", com1_accuracy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    summ = tf.summary.merge_all()
    
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    #saver object
    all_saver = tf.train.Saver()

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    
    for i in range(ep):
        for batch_i in range(train_x.shape[0] // batch_size_train):
            if batch_i % 1 == 0:
                batch_xs, batch_ys = train_x[(batch_i+batch_i*i):batch_size_train,:,:,:], y_org[(batch_i+batch_i*i):batch_size_train,:]
                train_accuracy1 = sess.run(com1_accuracy, feed_dict={x:batch_xs, y:batch_ys})
                
                print("batch %d of epoch %d/%d"%(batch_i, i, ep))
                print("accuracy: %f"%train_accuracy1)
                print("\n")
                s = sess.run(summ, feed_dict={x:batch_xs, y:batch_ys})
                sess.run(com1_optimizer, feed_dict={x:batch_xs, y:batch_ys})
                
                writer.add_summary(s, i)
                #saver object
                #all_saver.save(sess, LOGDIR+'/weigths')


    "-------------------- testing phase -------------------------------"
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>> testing phase <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    for batch_i in range(test_x.shape[0] // batch_size_test):
        batch_xs, batch_ys = test_x[batch_i*batch_size_test:batch_size_test+batch_i*batch_size_test,:,:,:], y_val[batch_i*batch_size_test:batch_size_test+batch_i*batch_size_test ,:]
        # test accuracy
        test_accuracy = sess.run(com1_accuracy, feed_dict={x:batch_xs, y:batch_ys})
        print("batch %d/%d"%(batch_i, test_x.shape[0] // batch_size_test))
        print("test accuracy: %f"%test_accuracy)
        print("\n")


    "---------------- weights & bias writes ---------------------------"
    w_cnn = {}
    b_cnn = {}
    z_cnn = {}
    a_cnn = {}
 
    w_cnn = sess.run(com1_W_params)
    b_cnn = sess.run(com1_B_params)
    #z_cnn = sess.run(com1_Z_params)
    #a_cnn = sess.run(com1_A_params) #Z and A needs feed dict also as they are intermediate parameters

    w_fc1 = sess.run(W_fc1)
    b_fc1 = sess.run(b_fc1)
    #z_fc1 = sess.run(z_fc1)
    #h_fc1 = sess.run(h_fc1) #Z and A needs feed dict also as they are intermediate parameters

    w_fc2 = sess.run(W_fc2)
    b_fc2 = sess.run(b_fc2)
    #z_fc2 = sess.run(z_fc2)
    #h_fc2 = sess.run(h_fc2) #Z and A needs feed dict also as they are intermediate parameters

    w_fc3 = sess.run(W_fc3)
    b_fc3 = sess.run(b_fc3)
    #z_y = sess.run(z_y) #Z and A needs feed dict also as they are intermediate parameters

    
    #return w_cnn, b_cnn, z_cnn, a_cnn, w_fc1, b_fc1, z_fc1, h_fc1, w_fc2, b_fc2, z_fc2, h_fc2, w_fc3, b_fc3, z_y #return is required to save the variable Z and A needs feeddict also as they are intermediate parameters
    return w_cnn, b_cnn, w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3 #return is required to save the variable

########################################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



##############################################
"""
Inputs X and Y def:
"""
#train_x, train_y, test_x, test_y, classes = load_dataset_rgb()
train_x, train_y, test_x, test_y, classes = load_dataset_gray()

"reshaping to 4d for operation -- only for gray image"
train_x = np.reshape(train_x,[train_x.shape[0],train_x.shape[1],train_x.shape[2],1])
test_x = np.reshape(test_x,[test_x.shape[0],test_x.shape[1],test_x.shape[2],1])

#plt.imshow(train_set_x_orig[8])
print("\n\nNo. of Training samples (batch): %d"%train_x.shape[0])
print("No. of Test samples: %d"%test_x.shape[0])
print("No. of gestures: %d"%classes.size)
print("Image format (Len, Wid, Chan): %d * %d * %d"%(train_x.shape[1],train_x.shape[2],train_x.shape[3]))

#train_x = train_x/255
#test_x = test_x/255

train_x = train_x/1.0
test_x = test_x/1.0

y_org = np.eye(6)[train_y] #converting it into one-hot encoding  # this is equal to y
y_org = np.reshape(y_org,[y_org.shape[1],y_org.shape[2]])

y_val = np.eye(6)[test_y] #converting it into one-hot encoding   # this is equal to y_test
y_val = np.reshape(y_val,[y_val.shape[1],y_val.shape[2]])

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

num_cnn_layers = 3

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
com1_n_fc3 = classes.size

### FC layer params ends


########################################################################################



########################################################################################

" final weight and biases of the network"
w_cnn = {}
b_cnn = {}
z_cnn = {}
a_cnn = {}


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#write here batch size and runs
ep = 3
batch_size_train = 360
batch_size_test = 20
learning_rate = 0.00084
keep_prob_fc_node = 0.85

#call the model here with alpha and multiple runs with for loop to change alpha
#w_cnn, b_cnn, z_cnn, a_cnn, w_fc1, b_fc1, z_fc1, h_fc1, w_fc2, b_fc2, z_fc2, h_fc2, w_fc3, b_fc3, z_y = gesture_recog_cnn_1com(train_x, y_org, test_x, y_val, classes, alpha=learning_rate, keeprobs=keep_prob_fc_node)
w_cnn, b_cnn, w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3 = gesture_recog_cnn_1com(train_x, y_org, test_x, y_val, classes, alpha=learning_rate, keeprobs=keep_prob_fc_node)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
########################################################################################

write_cnn_nparray_to_file(w_cnn["w1"],"w_b_params_csv/cnn_l1_w")
write_cnn_nparray_to_file(w_cnn["w2"],"w_b_params_csv/cnn_l2_w")
write_cnn_nparray_to_file(w_cnn["w3"],"w_b_params_csv/cnn_l3_w")

write_fc_nparray_to_file(b_cnn["b1"],"w_b_params_csv/cnn_l1_b")
write_fc_nparray_to_file(b_cnn["b2"],"w_b_params_csv/cnn_l2_b")
write_fc_nparray_to_file(b_cnn["b3"],"w_b_params_csv/cnn_l3_b")


write_fc_nparray_to_file(w_fc1,"w_b_params_csv/fc_l1_w")
write_fc_nparray_to_file(b_fc1,"w_b_params_csv/fc_l1_b")

write_fc_nparray_to_file(w_fc2,"w_b_params_csv/fc_l2_w")
write_fc_nparray_to_file(b_fc2,"w_b_params_csv/fc_l2_b")

write_fc_nparray_to_file(w_fc3,"w_b_params_csv/fc_l3_w")
write_fc_nparray_to_file(b_fc3,"w_b_params_csv/fc_l3_b")
