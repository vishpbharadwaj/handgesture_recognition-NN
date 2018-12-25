
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py
from scipy import ndimage

LOGDIR = "/home/vishnu/Documents/nn_files/tf_tests/modular_py/log_com_gesture_cnn"



########################################################################################
def load_dataset_rgb():
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
def load_dataset_gray():
    train_dataset_gray = h5py.File('signs_dataset/train_signs_gray.h5', "r")
    train_set_x_orig = np.array(train_dataset_gray["train_set_x_gray"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset_gray["train_set_y_gray"][:]) # your train set labels

    test_dataset_gray = h5py.File('signs_dataset/test_signs_gray.h5', "r")
    test_set_x_orig = np.array(test_dataset_gray["test_set_x_gray"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset_gray["test_set_y_gray"][:]) # your test set labels

    classes = np.array(test_dataset_gray["list_classes"][:]) # the list of classes
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

########################################################################################



########################################################################################
def filteredimg_gray(img,filt):
    return ndimage.convolve(img, filt, mode='constant',cval=1.0)

########################################################################################



########################################################################################
def put_conv_img_ongrid(z_conv):
    '''
    Visualize convolved image with filters .
    Arranges filters into a grid, with some paddings between adjacent filters.
    
    Args:
        kernel: tensor of shape [Y, X, NumChannels, NumKernels]
    
    Return:
        Tensor of shape [1, Y, X, NumChannels].
    '''
    
    pad = 1
    padding = tf.constant([[0,0],[pad, pad], [pad, pad],[0,0]]) #padding of 1 around first 2d filter of n*n
    padded_img = tf.pad(z_conv,padding,"CONSTANT")
    print(padded_img)
    
    grey_img_size = tf.shape(padded_img)[1]  #or tf.shape(padded_img)[2]
    num_img_chan = tf.shape(padded_img)[3]
    num_img_batch = tf.shape(padded_img)[0]
    
    num_grey_img = num_img_batch * num_img_chan
    print(num_grey_img)
    
    #if num_grey_img % 2 ==0:
    #    num_grey_img_half = num_grey_img // 2
    #else
    #    num_grey_img_half
    
    #conv_img = tf.reshape(padded_img,[1, grey_img_size*grey_img_size*num_img_chan*num_grey_img_half,grey_img_size*grey_img_size*num_img_chan*num_grey_img_half,1])
    
    conv_img = tf.reshape(padded_img,[1, grey_img_size*grey_img_size*num_img_chan,num_img_batch,1])

    return conv_img
########################################################################################



########################################################################################
def committee_full_gesture_cnn(alpha,keeprobs):
 
    #tf.reset_default_graph()
    sess = tf.Session()
    
#input placeholders
    x = tf.placeholder(tf.float32, [None, train_x.shape[1],train_x.shape[2],train_x.shape[3]])
    y = tf.placeholder(tf.float32, [None, classes.size])
    #x = tf.cast(x,tf.float32) # casting as the original was uint8

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print("\n --------------- com1 params ------------------\n")
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
        com1_fc_flat = tf.reshape( com1_A_params["a{0}".format(com1_num_cnn_layers)], [-1, (tf.shape(com1_A_params["a{0}".format(com1_num_cnn_layers)])[1]) * (tf.shape(com1_A_params["a{0}".format(com1_num_cnn_layers)])[2]) * (tf.shape(com1_A_params["a{0}".format(com1_num_cnn_layers)])[3])] )

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #fully-connected layer 1:
    with tf.name_scope('fc1_com1') as scope:
        W_fc1 = tf.Variable(tf.truncated_normal([com1_fc_shape_last_cnn, com1_n_fc1], stddev=0.1),name="w_fc1")
        b_fc1 = tf.Variable(tf.truncated_normal([com1_n_fc1], stddev=0.1),name="b_fc1")
        h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(com1_fc_flat, W_fc1) + b_fc1), keeprobs )
        
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
        h_fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2), keeprobs )
        
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
        com1_y_pred = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
        
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

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print("\n --------------- com2 params ------------------\n")
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
        with tf.name_scope('convolution_com2'):
            com2_W_params["w{0}".format(iter_i)] = tf.Variable( tf.truncated_normal([ com2_Fsize_params["fsize{0}".format(iter_i)], com2_Fsize_params["fsize{0}".format(iter_i)], com2_ChanNum_params["chan{0}".format(iter_i)], com2_Fnum_params["fnum{0}".format(iter_i)] ], stddev=0.1) )

            com2_B_params["b{0}".format(iter_i)] = tf.Variable( tf.constant(0.1, shape=[com2_Fnum_params["fnum{0}".format(iter_i)]]) )

            com2_Z_params["z{0}".format(iter_i)] = tf.nn.conv2d( com2_A_params["a{0}".format(iter_i-1)], com2_W_params["w{0}".format(iter_i)], com2_Stride_params["stride{0}".format(iter_i)], padding="VALID" ) + com2_B_params["b{0}".format(iter_i)]

        # maxpool
        if (com2_Maxpool_ops["pool{0}".format(iter_i)]) == True:
            with tf.name_scope('max_pool_com2'):
                com2_Maxpool_params["max_pool{0}".format(iter_i)] = tf.nn.max_pool( com2_Z_params["z{0}".format(iter_i)], com2_Pool_Ksize_params["ksize{0}".format(iter_i)], com2_Pool_stride_params["kstride{0}".format(iter_i)], padding='VALID' )


        # relu
        with tf.name_scope('a_conv--relu_com2'):
            if (com2_Maxpool_ops["pool{0}".format(iter_i)]) == True:
                com2_A_params["a{0}".format(iter_i)] = tf.nn.relu( com2_Maxpool_params["max_pool{0}".format(iter_i)], name=('a_conv%d'%iter_i) )
            else:
                com2_A_params["a{0}".format(iter_i)] = tf.nn.relu( com2_Z_params["z{0}".format(iter_i)], name=('a_conv%d'%iter_i) )

        
        # for Z and not after max pool and relu
        tf.summary.histogram("weights_com2", com2_W_params["w{0}".format(iter_i)])
        tf.summary.histogram("biases_com2", com2_B_params["b{0}".format(iter_i)])

        tf.summary.histogram("activations_com2", com2_A_params["a{0}".format(iter_i)])
         
        # Visualize conv kernels
        #grid = put_conv_img_ongrid(com2_A_params["a{0}".format(iter_i)])
        #print(grid.shape)
        #tf.summary.image('com2_kernal', grid, max_outputs=1)

       
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
    with tf.name_scope('flattening_com2'):
        com2_fc_flat = tf.reshape( com2_A_params["a{0}".format(com2_num_cnn_layers)], [-1, (tf.shape(com2_A_params["a{0}".format(com2_num_cnn_layers)])[1]) * (tf.shape(com2_A_params["a{0}".format(com2_num_cnn_layers)])[2]) * (tf.shape(com2_A_params["a{0}".format(com2_num_cnn_layers)])[3])] )

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #fully-connected layer 1:
    with tf.name_scope('fc1_com2') as scope:
        W_fc1 = tf.Variable(tf.truncated_normal([com2_fc_shape_last_cnn, com2_n_fc1], stddev=0.1),name="w_fc1")
        b_fc1 = tf.Variable(tf.truncated_normal([com2_n_fc1], stddev=0.1),name="b_fc1")
        h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(com2_fc_flat, W_fc1) + b_fc1), keeprobs )
        
        tf.summary.histogram("weights_com2", W_fc1)
        tf.summary.histogram("biases_com2", b_fc1)
        tf.summary.histogram("activations_com2", h_fc1)
        print("\nshape of fc1: ")
        print(h_fc1.shape)
        print("\n")
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
       
    #fully-connected layer 2:
    with tf.name_scope('fc2_com2') as scope:
        W_fc2 = tf.Variable(tf.truncated_normal([com2_n_fc1, com2_n_fc2], stddev=0.1),name="w_fc2")
        b_fc2 = tf.Variable(tf.truncated_normal([com2_n_fc2], stddev=0.1),name="b_fc2")
        h_fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2), keeprobs )
        
        tf.summary.histogram("weights_com2", W_fc2)
        tf.summary.histogram("biases_com2", b_fc2)
        tf.summary.histogram("activations_com2", h_fc2)
        print("\nshape of fc2: ")
        print(h_fc2.shape)
        print("\n")
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
    #fully-connected layer 3 - softmax:
    with tf.name_scope('fc3-softmax_com2') as scope:
        W_fc3 = tf.Variable(tf.truncated_normal([com2_n_fc2, com2_n_fc3], stddev=0.1),name="w_fc3")
        b_fc3 = tf.Variable(tf.truncated_normal([com2_n_fc3], stddev=0.1),name="b_fc3")
        #h_fc3 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3), keeprobs )
        com2_y_pred = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
        
        tf.summary.histogram("weights_com2", W_fc3)
        tf.summary.histogram("biases_com2", b_fc3)
        tf.summary.histogram("activations_com2", com2_y_pred)
        print("\nshape of fc3-softmax: ")
        print(com2_y_pred.shape)
        print("\n")
    

    #loss functions:
    with tf.name_scope('cross_entropy_com2'):
        com2_cross_entropy = -tf.reduce_sum(y * tf.log(com2_y_pred))
        
        tf.summary.scalar("com2_cross_entropy", com2_cross_entropy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #training:
    with tf.name_scope('train_com2'):    
        com2_optimizer = tf.train.AdamOptimizer(alpha).minimize(com2_cross_entropy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    #com2_accuracy:
    with tf.name_scope('accuracy_com2'):
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

    #saver object
    all_saver = tf.train.Saver()

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    
    for i in range(ep):
        for batch_i in range(train_x.shape[0] // batch_size_train):
            if batch_i % 1 == 0:
                batch_xs, batch_ys = train_x[(batch_i+batch_i*i):batch_size_train,:,:,:], y_org[(batch_i+batch_i*i):batch_size_train,:]
                train_accuracy1 = sess.run(com1_accuracy, feed_dict={x:batch_xs, y:batch_ys})
                train_accuracy2 = sess.run(com2_accuracy, feed_dict={x:batch_xs, y:batch_ys})
                full_accuracy_2 = sess.run(full_accuracy, feed_dict={x:batch_xs, y:batch_ys})
                
                print("batch %d of epoch %d/%d"%(batch_i, i, ep))
                print("com1 accuracy: %f"%train_accuracy1)
                print("com2 accuracy: %f"%train_accuracy2)
                print("full_accuracy: %f"%full_accuracy_2)
                print("\n")
                s = sess.run(summ, feed_dict={x:batch_xs, y:batch_ys})
                sess.run(com1_optimizer, feed_dict={x:batch_xs, y:batch_ys})
                sess.run(com2_optimizer, feed_dict={x:batch_xs, y:batch_ys})
                
                writer.add_summary(s, i)
                #saver object
                #all_saver.save(sess, LOGDIR+'/weigths')


    "-------------------- testing phase -------------------------------"
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>> testing phase <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    for batch_i in range(test_x.shape[0] // batch_size_test):
        batch_xs, batch_ys = test_x[batch_i*batch_size_test:batch_size_test+batch_i*batch_size_test,:,:,:], y_val[batch_i*batch_size_test:batch_size_test+batch_i*batch_size_test ,:]
        # test accuracy
        test_accuracy = sess.run(full_accuracy, feed_dict={x:batch_xs, y:batch_ys})
        print("batch %d/%d"%(batch_i, test_x.shape[0] // batch_size_test))
        print("test accuracy: %f"%test_accuracy)
        print("\n")


    "---------------- weights & bias writes ---------------------------"
    w_com1 = {}
    b_com1 = {}
    w_com2 = {}
    b_com2 = {}
 
    w_com1 = sess.run(com1_W_params)
    b_com1 = sess.run(com1_B_params)
    
    w_com2 = sess.run(com2_W_params)
    b_com2 = sess.run(com2_B_params)

    return w_com1, b_com1, w_com2, b_com2  #return is required to save the variable

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

train_x = train_x/255
test_x = test_x/255

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
com1_n_fc3 = classes.size

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
    com2_Pool_Ksize_params["ksize2"] = [1,4,4,1]
    com2_Pool_stride_params["kstride2"] = [1,2,2,1]
#layer 2 ends 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
### FC layer params
com2_num_fc_layers = 2
com2_fc_shape_last_cnn = 256 #"this is done because tf.reshape had some issues"

#fc layer 1
com2_n_fc1 = 128

#fc layer 2
com2_n_fc2 = 64

#fc layer 3
com2_n_fc3 = classes.size

### FC layer params ends



########################################################################################

" final weight and biases of the network"
w_com1 = {}
b_com1 = {}
w_com2 = {}
b_com2 = {}


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#write here batch size and runs
ep = 100
batch_size_train = 360
batch_size_test = 20
learning_rate = 0.00085
keep_prob_fc_node = 0.85

#call the model here with alpha and multiple runs with for loop to change alpha
w_com1, b_com1, w_com2, b_com2 = committee_full_gesture_cnn(alpha=learning_rate, keeprobs=keep_prob_fc_node)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
