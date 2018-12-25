import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py
from scipy import ndimage
from tensorflow.examples.tutorials.mnist import input_data



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
def load_mnist_unnormalized():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_x = (mnist.train.images) * 255
    train_y = mnist.train.labels
    test_x = (mnist.test.images) * 255
    test_y = mnist.test.labels

    return train_x, train_y, test_x, test_y
########################################################################################


########################################################################################
def filteredimg_gray(img,filt):
    return ndimage.convolve(img, filt, mode='constant',cval=1.0)

########################################################################################



########################################################################################
def conv_forward(X, W, b, stride=1, padding=1):
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col + b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding, X_col)

    return out, cache
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
def write_cnn_nparray_to_file(np_array, file_name):
    # Write the array to disk
    with open(file_name+'.csv', 'wb') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(np_array.shape).encode())
        outfile.write('#\n'.encode())

        # Iterating through a ndimensional array produces slices along
        # slice is 3rd dimension and block is 4th dimension
        dim3 = np_array.shape[2]
        dim4 = np_array.shape[3]

        for i in range(dim4):
            # writing out a break to indicate different block
            outfile.write('#\n'.encode())
            outfile.write('# ========================== Block %d/%d ==========================\n'.encode()%(i+1,dim4))
    
            for j in range(dim3):
                # writing out a break to indicate different slice
                outfile.write('#\n'.encode())
                outfile.write('# -------- Slice %d/%d --------\n'.encode()%(j+1,dim3))
                np.savetxt(outfile, np_array[:,:,j,i], fmt='%f')
                outfile.write('# 2-D slice -> max_value: %f, min_value:%f \n'.encode()%(np_array[:,:,j,i].max(),np_array[:,:,j,i].min()))

#######################################################################################


########################################################################################
def write_fc_nparray_to_file(np_array, file_name):
    # Write the array to disk
    with open(file_name+'.csv', 'wb') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(np_array.shape).encode())
        outfile.write('#\n'.encode())

        # Iterating through a ndimensional array produces slices along
        # slice is 3rd dimension and block is 4th dimension
        np.savetxt(outfile, np_array, fmt='%s')
        #outfile.write('# 2-D slice -> max_value: %f, min_value:%f \n'.encode()%(np_array.max(),np_array.min()))

#######################################################################################


#######################################################################################
def tf_param_round(x, decimals):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier
#######################################################################################



#######################################################################################
def tf_2bits_round(x, in_frac_bits=2):
    multiplier = tf.constant(10**in_frac_bits, dtype=x.dtype)
    last_digits = (x * multiplier) % multiplier
    a = np.zeros([x.shape[0]])
    for i in range(a.shape[0]):
        if (last_digits[i] >= 0 and last_digits[i] < 25 ):
            a[i] = np.round(x[i]) + 0.25

        elif (last_digits[i] >= 25 and last_digits[i] < 50 ):
            a[i] = np.round(x[i]) + 0.5

        elif (last_digits[i] >= 50 and last_digits[i] < 75 ):
            a[i] = np.round(x[i]) + 0.75

        elif (last_digits[i] >= 0.75 ):
            a[i] = np.round(x[i]) + 1

        #def f1(): return 0.25
        #def f2(): return 0.5 
        #def f3(): return 0.75
        #def f4(): return 1.0
        #def fn(): return 0.0

        #a[i] =  tf.round(x[i]) + tf.case({tf.less(last_digits[i],25):f1, tf.less(last_digits[i],50):f2, tf.less(last_digits[i],75):f3, tf.greater(last_digits[i],75):f4}, default=fn, exclusive=True)
    
    return tf.stack(a)
#######################################################################################



#######################################################################################
def norm_conv(img_2d, kern):
    #convolution with no padding and stride 1
    #stride x & y dimension should be same
    stride = 1
    jmp = kern.shape[0]
    y_axis = math.floor( (img_2d.shape[0] - kern.shape[0])/stride + 1)
    x_axis = math.floor( (img_2d.shape[1] - kern.shape[1])/stride + 1)
    conv_out = np.zeros([x_axis, y_axis])
    conv_slices = np.zeros([x_axis, y_axis, kern.shape[0], kern.shape[1]])
    #conv_slices = np.zeros([kern.shape[0], kern.shape[1], x_axis, y_axis])

    for y in range(y_axis):
        for x in range(x_axis):
            conv_out[x][y] = (img_2d[x:(x+jmp),y:(y+jmp)] * kern).sum()
            if (conv_out[x][y] > 0):
                print(x)
                print(y)
            conv_slices[x][y] = img_2d[x:(x+jmp),y:(y+jmp)]

    return conv_out, conv_slices

#######################################################################################


#######################################################################################
def rand_array_0_1(k, m, n):
    # K no. of ones in mxn
    arr = np.zeros(m*n)
    arr[:k]  = 1
    np.random.shuffle(arr)
    arr = np.reshape(arr,[m,n])
    return arr
#######################################################################################


## np.random.seed(42) ##everytime np.random.randint is run, seed should be run first
##random matrix of (mxn): np.random.randint(0,255,(m,n))


## typecasting val = val.astype('int')


## decimal to hex file conversion for 2D array
## hex_val  = [hex(x)[2:] for x in np.reshape(dec_val,[mxn])]
