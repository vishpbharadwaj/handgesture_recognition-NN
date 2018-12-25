import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py
from scipy import ndimage




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
        np.savetxt(outfile, np_array, fmt='%f')
        outfile.write('# 2-D slice -> max_value: %f, min_value:%f \n'.encode()%(np_array.max(),np_array.min()))

#######################################################################################

