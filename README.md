# handgesture_recognition-NN
Hand gesture recognition using CNN and FC implemented in tensorflow.

This is scalable where number, stride, kernel and pool size varies depending on the configuration. 
Number of nodes and number of layers in FC also can be altered.

Folder structure:
	- MNIST_data			: MNSIT data set folder
	- signs_dataset			: RGB signs data set
	- signs_dataset_gray	: gray images of signs dataset
    - com1_highres  		: has 2 committees structure
    - com1_stride_2 		: has stride 2 but no pooling
    - com2_lowres   		: has 2nd committee graph with less computation
    - lenet_arch    		: Lenet architecture graph
    - working_bkp   		: working backup of committee
	- modular_py, notebooks, python_files    : misc examples to tryout few scenarious


Software:
    - Python 3
    - Numpy
    - Scipy
    - Tensorflow


Run procedure:
    - Open Python3 console
    - RUN: exec(open("<FILE_NAME.py>").read())
	
	-- appropriate path for MNIST_data, signs_dataset and signs_dataset_gray need to be given.
	