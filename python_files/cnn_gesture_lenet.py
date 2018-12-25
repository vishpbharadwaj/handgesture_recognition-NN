
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py

LOGDIR = "/home/vishnu/Dropbox/intel_works/nn_files/tf_tests/python_files/log_mnist_wideTOnarow"



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
def put_filters_on_grid(w_filter):
    '''
    Visualize conv. filters as an image.
    Arranges filters into a grid, with some paddings between adjacent filters.
    
    Args:
        kernel: tensor of shape [Y, X, NumChannels, NumKernels]
    
    Return:
        Tensor of shape [1, Y, X, NumChannels].
    '''
    pad = 1
    padding = tf.constant([[pad, pad], [pad, pad],[0,0],[0,0]]) #padding of 1 around first 2d filter of n*n
    padded_filt = tf.pad(w_filter,padding,"CONSTANT")
    
    padded_filt = tf.transpose(padded_filt) #this is done so that filter of n*n goes to last dimension
    
    #filt_size = padded_filt.get_shape()[3]
    #num_chan = padded_filt.get_shape()[0]
    #num_outfilt = padded_filt.get_shape()[1]
    
    #filt_size = padded_filt.shape[3]
    #num_chan = padded_filt.shape[0]
    #num_outfilt = padded_filt.shape[1]
    
    filt_size = tf.shape(padded_filt)[3]
    num_chan = tf.shape(padded_filt)[0]
    num_outfilt = tf.shape(padded_filt)[1]
    
    filt_size = tf.cast(filt_size, tf.int32)   #typecasting and stacking because issue with tf.reshape command
    num_chan = tf.cast(num_chan,tf.int32)
    num_outfilt = tf.cast(num_outfilt,tf.int32)
    roll_shape = tf.stack([1,num_chan*filt_size,num_outfilt*filt_size,1])
    
    #roll_shape = tf.constant([1,num_chan*filt_size,num_outfilt*filt_size,1])
    
   
    grid = tf.reshape(padded_filt,roll_shape)
    #print(grid.get_shape())
        
    return grid
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
"""
Inputs X and Y def:
"""
train_x, train_y, test_x, test_y, classes = load_dataset()
#index = 1079
#plt.imshow(train_set_x_orig[index])
print("No. of Training samples (batch): %d"%train_x.shape[0])
print("No. of Test samples: %d"%test_x.shape[0])
print("No. of gestures: %d"%classes.size)
print("Image format (Len, Wid, Chan): %d * %d * %d"%(train_x.shape[1],train_x.shape[2],train_x.shape[3]))

y_org = np.eye(6)[train_y] #converting it into one-hot encoding  # this is equal to y
y_org = np.reshape(y_org,[y_org.shape[1],y_org.shape[2]])

y_val = np.eye(6)[test_y] #converting it into one-hot encoding   # this is equal to y_test
y_val = np.reshape(y_val,[y_val.shape[1],y_val.shape[2]])



########################################################################################



########################################################################################
def gesture_Lenet(alpha):
    
    tf.reset_default_graph()
    sess = tf.Session()
    
    #input placeholders
    x = tf.placeholder(tf.float32, [None, train_x.shape[1],train_x.shape[2],train_x.shape[3]])
    y = tf.placeholder(tf.float32, [None, classes.size])
    #x = tf.cast(x,tf.float32) # casting as the original was uint8
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #conv1 layer to convert the input to Lenet format:
    with tf.name_scope('conv_1'):
        ch_1 = tf.cast(x.shape[3],tf.int32)
        l1_filter_size = 3 # filter size of n*n for convolution
        l1_filter_Num = 3 # No. of filters required
        stride_1 = [1,2,2,1]
        
        w = tf.Variable(tf.truncated_normal([l1_filter_size, l1_filter_size, ch_1, l1_filter_Num], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[l1_filter_Num]), name="B")
        
        z_conv = tf.nn.conv2d(x,w,stride_1,padding="SAME") + b
        act1 = tf.nn.relu(z_conv,name="act1")
    
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act1)
        print(act1.shape)
        
        # Visualize conv1 kernels
        grid = put_conv_img_ongrid(z_conv)
        print(grid.shape)
        tf.summary.image('conv1_kernal', grid, max_outputs=1)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #conv2 layer to convert the input to Lenet format:
    with tf.name_scope('conv_2'):
        ch_1 = tf.cast(act1.shape[3],tf.int32)
        l2_filter_size = 5 # filter size of n*n for convolution
        l2_filter_Num = 6 # No. of filters required
        stride_2 = [1,1,1,1]
        
        w = tf.Variable(tf.truncated_normal([l2_filter_size, l2_filter_size, ch_1, l2_filter_Num], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[l2_filter_Num]), name="B")
        
        z_conv = tf.nn.conv2d(act1,w,stride_2,padding="VALID") + b
        act2 = tf.nn.relu(z_conv,name="act2")
    
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act2)
        print(w.get_shape())
        
        grid = put_conv_img_ongrid(z_conv)
        print(grid.get_shape())
        tf.summary.image('conv2_kernal', grid, max_outputs=1)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #avg pooling layer:
    with tf.name_scope('pool1'):
        ch_2 = tf.cast(act2.shape[3],tf.int32)
        k_size = [1,2,2,1] #filter size
        act2_pool = tf.nn.avg_pool(act2,k_size,strides=[1, 2, 2, 1],padding='VALID',name="act2_pool")
        print(act2_pool.get_shape())
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #conv3 layer to convert the input to Lenet format:
    with tf.name_scope('conv_3'):
        ch_3 = tf.cast(act2_pool.shape[3],tf.int32)
        l3_filter_size = 5 # filter size of n*n for convolution
        l3_filter_Num = 16 # No. of filters required
        stride_3 = [1,1,1,1]
        
        w = tf.Variable(tf.truncated_normal([l3_filter_size, l3_filter_size, ch_3, l3_filter_Num], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[l3_filter_Num]), name="B")
        
        z_conv = tf.nn.conv2d(act2_pool,w,stride_3,padding="VALID") + b
        act3 = tf.nn.relu(z_conv,name="act3")
    
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act3)
        #print(w.get_shape())
        
        grid = put_conv_img_ongrid(z_conv)
        print(grid.shape)
        tf.summary.image('conv3_kernal', grid, max_outputs=1)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    #conv4 layer to convert the input to Lenet format:
    with tf.name_scope('conv_4'):
        ch_4 = tf.cast(act3.shape[3],tf.int32)
        l4_filter_size = 2 # filter size of n*n for convolution
        l4_filter_Num = 16 # No. of filters required
        stride_4 = [1,2,2,1]
        
        w = tf.Variable(tf.truncated_normal([l4_filter_size, l4_filter_size, ch_4, l4_filter_Num], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[l4_filter_Num]), name="B")
        
        z_conv = tf.nn.conv2d(act3,w,stride_4,padding="VALID") + b
        act4 = tf.nn.relu(z_conv,name="act4")
    
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act4)
        print(act4.get_shape())
        
        grid = put_conv_img_ongrid(z_conv)
        print(grid.get_shape())
        tf.summary.image('conv4_kernal', grid, max_outputs=1)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    #reshaping to connect fully-connected layer:
    act4_flat = tf.reshape(act4, [-1, 5 * 5 * 16])
    print(act4_flat.get_shape())
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    #fully-connected layer 1:
    with tf.name_scope('fc1') as scope:
        n_fc1 = 120
        W_fc1 = tf.Variable(tf.truncated_normal([5 * 5 * 16, n_fc1], stddev=0.1),name="w_fc1")
        b_fc1 = tf.Variable(tf.truncated_normal([n_fc1], stddev=0.1),name="b_fc1")
        h_fc1 = tf.nn.relu(tf.matmul(act4_flat, W_fc1) + b_fc1)
        
        tf.summary.histogram("weights", W_fc1)
        tf.summary.histogram("biases", b_fc1)
        tf.summary.histogram("activations", h_fc1)
        print(h_fc1.get_shape())
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   
    #fully-connected layer 2:
    with tf.name_scope('fc2') as scope:
        n_fc2 = 84
        W_fc2 = tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1),name="w_fc2")
        b_fc2 = tf.Variable(tf.truncated_normal([n_fc2], stddev=0.1),name="b_fc2")
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        
        tf.summary.histogram("weights", W_fc2)
        tf.summary.histogram("biases", b_fc2)
        tf.summary.histogram("activations", h_fc2)
        print(h_fc2.get_shape())
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    #softmax layer:
    with tf.name_scope('softmax') as scope:
        W_fc3 = tf.Variable(tf.truncated_normal([n_fc2, 6], stddev=0.1),name="w_fc3")
        b_fc3 = tf.Variable(tf.truncated_normal([6], stddev=0.1),name="b_fc3")
        y_pred = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
        print(y_pred.get_shape())
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #loss functions:
    with tf.name_scope('cross_entropy'):
        cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
        
        tf.summary.scalar("cross_entropy", cross_entropy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #training:
    with tf.name_scope('train'):    
        optimizer = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    #accuracy:
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print(correct_prediction.get_shape())
        tf.summary.scalar("accuracy", accuracy)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    summ = tf.summary.merge_all()
    
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    #write here batch size and runs
    ep = 20
    batch_size = 50
    
    for i in range(ep):
        for batch_i in range(train_x.shape[0] // batch_size):
            if batch_i % 5 == 0:
                batch_xs, batch_ys = train_x[(batch_i+batch_i*i):batch_size,:,:,:], y_org[(batch_i+batch_i*i):batch_size,:]
                train_accuracy = sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys})
                print(train_accuracy)
                s = sess.run(summ, feed_dict={x:batch_xs, y:batch_ys})
                writer.add_summary(s, i)
    
            #if i % 4 == 0:
                sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})

########################################################################################




""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#call the model here with alpha and multiple runs with for loop to change alpha
gesture_Lenet(0.0005)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
