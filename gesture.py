
#get_ipython().magic('matplotlib inline')
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.contrib import rnn


batch_size = 32

#Prepare input data
classes = ['Abort','Circle', 'Hello', 'No', 'Stop', 'Turn Left', 'Turn Right', 
          'Stop', 'Turn', 'Warn']
num_classes = len(classes)

values = np.array(classes)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
print(inverted)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = (64, 48)
num_channels = 3
train_path='training_data'

##Read_Data
x_data = []
y_label = []
path = r'C:\\Study\\Sem 3\\ChrisTseng\\gesture\\Save'
for g in os.listdir(path):
    print (g)
    gp = os.path.join(path,g)
    for s in os.listdir(gp): #sample 1_0
        sp = os.path.join(gp, s)
        for i in os.listdir(sp):
            a = np.asarray(Image.open(os.path.join(sp, i)))
            x_data.append(a)
            y_label.append(g)
#     img = cv2.imread(os.path.join(path, f))


y_enc_label = onehot_encoder.fit_transform(label_encoder.fit_transform(y_label).reshape(len(y_label), 1))

x_data_a = np.array(x_data)

#x_data_a.shape

#y_enc_label.shape

X = x_data_a
y = y_enc_label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

x = tf.placeholder(tf.float32, shape=[None, img_size[0], img_size[1], num_channels], name='x')

##Network graph params
filter_size_conv1 = 11
num_filters_conv1 = 5

filter_size_conv2 = 6
num_filters_conv2 = 10

lstm_units = 500
    
fc_layer_size = 500

learning_rate = 1e-4

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.tanh(layer)

    return layer


def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_flat = create_flatten_layer(layer_conv2)


#layer_flat

################## Error regarding tensor dimention for LSTM input...
flat_out =tf.unstack(layer_flat ,1920 ,1)

lstm_layer=rnn.BasicLSTMCell(lstm_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer, flat_out, dtype="float32")

#weights and biases of appropriate shape to accomplish the task
out_weights=tf.Variable(tf.random_normal([lstm_units,num_classes]))
out_bias=tf.Variable(tf.random_normal([num_classes]))

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs[-1],out_weights)+out_bias

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
train = opt
#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cnt = 0
    total = x_train.shape[0]
    
    for e in range(epoch):
        start = timeit.default_timer()
        for i in range(0, total, batch_size):
            end = i + batch_size
            if(end >= total):
                end = total
            batch = X_train[i:end]
            y_ = y_train[i:end]
            cnt += batch_size
            
            feed_dict_tr = {x: batch,
                           y: y_}
            feed_dict_val = {x: X_val,
                                  y: y_val}
            
            train.run(feed_dict=feed_dict_tr)
             
            if cnt % 1000 == 0:
                print cnt," items trained"
                val_loss = session.run(cost, feed_dict=feed_dict_val)    
                show_progress(e, feed_dict_tr, feed_dict_val, val_loss)
#                 saver.save(sess, 'gesture-model')

        stop = timeit.default_timer()

        print "Total testing time for epoch: ", e, " =>", stop - start, " seconds"

    print "Processing complete!"
    print "Total number items trained on : ", total

