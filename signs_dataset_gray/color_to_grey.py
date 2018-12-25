import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py

LOGDIR = "/home/vishnu/Dropbox/intel_works/nn_files/tf_tests/modular_py/log_com_gesture_cnn"



########################################################################################
def load_dataset():
    train_dataset = h5py.File('train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

########################################################################################



########################################################################################
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
########################################################################################



########################################################################################
def rgb2gray_loop(dat4d):
    dat_gray = np.zeros(shape=(dat4d.shape[0],dat4d.shape[1],dat4d.shape[2]))
    for i in range(dat4d.shape[0]):
        dat_gray[i] = rgb2gray(dat4d[i])

    return dat_gray

########################################################################################


"""
Inputs X and Y def:
"""
train_x, train_y, test_x, test_y, classes = load_dataset()
#index = 1079
#plt.imshow(train_set_x_orig[index])
print("\n\nNo. of Training samples (batch): %d"%train_x.shape[0])
print("No. of Test samples: %d"%test_x.shape[0])
print("No. of gestures: %d"%classes.size)
print("Image format (Len, Wid, Chan): %d * %d * %d"%(train_x.shape[1],train_x.shape[2],train_x.shape[3]))

train_x_gray = rgb2gray_loop(train_x)
test_x_gray = rgb2gray_loop(test_x)


########################################################################################
" writing to h5py file"
train_dataset_gray = h5py.File('train_signs_gray.h5','w')
train_dataset_gray.create_dataset('train_set_x_gray',data=train_x_gray)
train_dataset_gray.create_dataset('train_set_y_gray',data=train_y)
train_dataset_gray.close()


test_dataset_gray = h5py.File('test_signs_gray.h5','w')
test_dataset_gray.create_dataset('test_set_x_gray',data=test_x_gray)
test_dataset_gray.create_dataset('test_set_y_gray',data=test_y)
test_dataset_gray.create_dataset('list_classes',data=classes)
test_dataset_gray.close()

########################################################################################
