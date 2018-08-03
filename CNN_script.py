# Importing Dependencies
# This code is optimized for python 3
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from datetime import timedelta
import math
from sklearn.metrics import roc_curve
import tensorflow as tf 
import numpy as np 
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import os 
from PIL import Image
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
# get_ipython().magic(u'matplotlib inline')
import time
from sklearn.preprocessing import scale
ion = tf.Session()
plt.style.use('ggplot')
## Importing our Images
ellipse =  'Ellipse_Images/'
spiral = 'Spiral_Images/'
path_sp = os.listdir(spiral)
path_el = os.listdir(ellipse)

## View how many images in each folder
print len(path_sp)
print len(path_el)
##### Building a Training Data
print('Reading Training data')
el_train = []
for i in range(1,5000):
    im = mpimg.imread(ellipse + path_el[i])
    el_train.append(np.ravel(im))
sp_train = [] 
for i in range(1,5000):
    im = mpimg.imread(spiral + path_sp[i])
    sp_train.append(np.ravel(im))
print(scale(el_train))

## Creating a training set
y_el_train = []
y_sp_train = []
for i in range(0,len(el_train)):
    y_el_train.append([0,1])
for i in range(0,len(sp_train)):
    y_sp_train.append([1,0])
X_train = el_train + sp_train
y_train = y_el_train + y_sp_train
del el_train, y_el_train, sp_train, y_sp_train
print('Training Data Done')
print('Reading Test Data')

######## Building the Test Data
el_test = []
sp_test = []
for i in range(20000,21000):
    im = mpimg.imread(ellipse + path_el[i])
    el_test.append(np.ravel(im))

for i in range(20000,21000):
    im = mpimg.imread(spiral + path_sp[i])
    sp_test.append(np.ravel(im))

## Creating a training set
y_el_test = []
y_sp_test = []
for i in range(0,len(el_test)):
    y_el_test.append([0,1])
for i in range(0,len(sp_test)):
    y_sp_test.append([1,0])
X_test = el_test + sp_test
y_test = y_el_test + y_sp_test

y_test_cls = []

#for i in range(len(y_test)):
#    if y_test[i] == [0,1]:
#        y_test_cls.append('Ellipse')
#    elif y_test[i] == [1,0]:
#        y_test_cls.append('Spiral')
# Generating Labels
for i in range(len(y_test)):
    if y_test[i] == [0,1]:
        y_test_cls.append(1)
    elif y_test[i] == [1,0]:
        y_test_cls.append(0)
y_test_cls = np.array(y_test_cls)



# To save memory
'''
In hindsight the "del" function does not do
anything. set the variables below to "None" instead
'''
del sp_test, el_test, y_sp_test, y_el_test
print('Test Data Done')
print('Test', len(X_test))
print('Train', len(X_train))

X, y = shuffle(X_train, y_train)
######################################
X_train, y_train = X[:int(len(X)*.8)], y[:int(len(X)*.8)]
#X_test,y_test_cls = X[int(len(X)*.8):len(X)], y[int(len(X)*.8):len(X)]


# In[3]:

## Data dimensions
# We know that Galaxy images are 100x100x3 pixels in each dimension.
img_size = 100

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 3

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size, num_channels)



# Number of classes, one class for each of 2 Galaxy types.
num_classes = 2

print("Size of:")
print("- Training-set:\t\t{}".format(len(X_train)))
print("- Test-set:\t\t{}".format(len(y_test)))

y_test_cls = y_test_cls
# Convolutional Layer 1.
filter_size1 = 10          # Convolution filters are 5 x 5 pixels.13
num_filters1 = 36         # There are 36 of these filters. 36

# Convolutional Layer 2.
filter_size2 = 8         # Convolution filters are 5 x 5 pixels.
num_filters2 = 36        # There are 36 of these filters.

######## Convolutional Layer 3 ###########
filter_size3 = 7
num_filters3 = 36

######## Convolutional Layer 4 ##########
filter_size4 = 5
num_filters4 = 36

# Fully-connected layer.
fc_size = 150 # Number of neurons in fully-connected layer.

# 100 pixels each dimension for each image
img_size = 100

# Images are stored in 1-D array of this length
img_size_flat = 100**2

# Tuple with height & width of iages used to reshape these arrays.
img_shape = (img_size, img_size, 3)

# Number of colour channels for the images: 1 for gray-scale
num_channels = 3

# Number of classes 'elliptical' and 'spiral'
num_channel = 2

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            if cls_true[i] == 0:
                xlabel ='Spiral'
            elif cls_true[i] == 1:
                xlabel = 'Ellipse'
        else:
            if cls_true[i] == 0:
                if cls_pred[i] == 0:
                    xlabel = "True: {0}, Pred: {1}".format('Spiral', 'Spiral')
                elif cls_pred[i] == 1:
                    xlabel = "True: {0}, Pred: {1}".format('Spiral', 'Ellipse')
            elif cls_true[i] == 1:
                if cls_pred[i] == 0:
                    xlabel = "True: {0}, Pred: {1}".format('Ellipse', 'Spiral')
                elif cls_pred[i] == 1:
                    xlabel = "True: {0}, Pred: {1}".format('Ellipse', 'Ellipse')
                
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig('Test8.png')
# Parameters about the our neural network
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)
    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases
    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)  
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

x = tf.placeholder(tf.float32, shape=[None, img_size_flat*3], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

print('dimension of x: ', x)
print('-------------------------------------')
print('dimension of x_image: ', x_image)
print('-------------------------------------')
print('dimension of y_true: ', y_true)
print('-------------------------------------')
print('dimension of y_true: ', y_true_cls)

######### 1st Conv layer ###############
layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
######### 2nd Conv layer ################
layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
    
########## 3rd Conv Layer ##################
layer_conv3, weights_conv3 =     new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters = num_filters3,
                   use_pooling=True)
########## 4th Conv Layer ###################
layer_conv4, weights_conv4 =     new_conv_layer(input=layer_conv3,
		   num_input_channels=num_filters3,
		   filter_size = filter_size4,
		   num_filters = num_filters4,
		   use_pooling=True)

dropout = 0.95
layer_flat, num_features = flatten_layer(layer_conv4)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

#layer_fc1 = tf.contrib.layers.batch_norm(layer_fc1, center = True, scale =False)
layer_fc1 = tf.nn.dropout(layer_fc1, dropout)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=fc_size,
                         use_relu=True)
#layer_fc2 = tf.contrib.layers.batch_norm(layer_fc2, center = True, scale =False)
                                         ### Switch to False
################ Layer_fc3 #################
layer_fc2 = tf.nn.dropout(layer_fc2, dropout)

layer_fc3 = new_fc_layer(input=layer_fc2,
                        num_inputs=fc_size,
                        num_outputs=fc_size,
                        use_relu = True)
layer_fc3 = tf.nn.dropout(layer_fc3, dropout)
################ Layer_fc4 ##################
layer_fc4 = new_fc_layer(input=layer_fc3, 
			num_inputs=fc_size,
			num_outputs=num_classes,
			use_relu = False)
layer_fc4 = tf.nn.dropout(layer_fc4, dropout)
y_pred = tf.nn.softmax(layer_fc4)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc4,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
ion = tf.Session()
ion.run(tf.global_variables_initializer())
saver = tf.train.Saver()
''' 
This batch function is not the best functioning at the moment 
need to fix some issues
'''
bs = 100
first = 0 
last = bs 
def nxt_batch(data1, data2):
    global first
    global last
    #print first, last
    tr = data1[first:last]
    ytr = data2[first:last]
    first = first + bs
    last = last + bs 
    return tr, ytr

total_iterations = 100

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations
    
    # Start-time used for printing time-usage below.
    start_time = time.time()
    a, b = nxt_batch(X_train, y_train)
    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        #x_batch, y_true_batch = X_train, y_train
	x_batch, y_true_batch = a, b
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        ion.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 1 == 0:
            # Calculate the accuracy on the training-set.
            acc = ion.run(accuracy, feed_dict=feed_dict_train)
           
            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
    
            print_test_accuracy()

            # Print it.
            print(msg.format(i + 1, acc))
    	save_path = saver.save(ion, 'modelling.ckpt') # Saving model	
    # Update the total number of iterations performed.
    total_iterations += num_iterations
   
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    return print_test_accuracy()
            



def plot_example_errors(cls_pred, correct):
  
    incorrect = np.array(correct == False)

    images = np.array(X_test)[incorrect]
    
    cls_pred = cls_pred[incorrect]
    cls_true = y_test_cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# In[20]:

def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = y_test_cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    # tick_marks = np.array(['Spiral', 'Elliptical'])
    plt.xticks(tick_marks, ['Spiral', 'Elliptical'], rotation=45)
    plt.yticks(tick_marks, ['Spiral', 'Elliptical'])
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    #plt.show()
    plt.savefig('confusion_matrix.png')

# Split the test-set into smaller batches of this size.
test_batch_size = 50

def print_test_accuracy(show_example_errors=True,
                        show_confusion_matrix=True):

    # Number of images in the test-set.
    num_test = len(X_test)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)

        images = np.array(X_test)[i:j, :]

        labels = np.array(y_test)[i:j, :]

        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = ion.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = y_test_cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    correct_sum = np.array(correct).sum()
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)
    fpr, tpr, thresholds = roc_curve(cls_true , cls_pred)

    plt.plot(fpr, tpr, linewidth = 2)#, label = label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel('True Positive Rate')
    plt.savefig('ROC_curve.png')
    return correct

print('Test Accuracy')
print_test_accuracy()
print('DONE')
print_test_accuracy()

for i in range(100):
    optimize(num_iterations=120)

print_test_accuracy(show_example_errors=True, show_confusion_matrix = True)