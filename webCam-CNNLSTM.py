import numpy as np
import cv2
import time
import tensorflow as tf
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import tflearn
from tflearn.layers.recurrent import lstm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected, time_distributed, flatten, activation

from image_utils import ImageText
from PIL import Image

classes = ['Circle','Turn Left', 'Turn Right']#['Abort', 'Circle', 'Hello', 'No', 'Stop', 'Turn Left', 'Turn Right', 'Turn', 'Warn']
num_classes = len(classes)
model_file = ""
layer_fc2 = ""
label_encoder = None
logits = None
model = None

def encoder():
    global label_encoder
    values = np.array(classes)
    # print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)
    # invert first example
    inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    # print(inverted)

def convert_to_oneHot(index):
    p = [0]*num_classes #np.zeros(num_classes)
    p[index] = 1
    return p



def decode(preds):
    labels = label_encoder.inverse_transform(preds)
    return labels

def cnnLSTM_model():
    global model
    filter_size_conv1 = 11
    num_filters_conv1 = 5

    filter_size_conv2 = 6
    num_filters_conv2 = 10

    filter_size_conv3 = 5
    num_filters_conv3 = 5

    filter_size_conv4 = 2
    num_filters_conv4 = 2

    lstm_units = 500

    learning_rate = 1e-4
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

    net = tflearn.input_data([None, 22, 64, 48, 1], name="input")
    net = time_distributed(net, conv_2d, args=[num_filters_conv1,
                                               filter_size_conv1, 1, 'same',
                                               'tanh'])
    net = time_distributed(net, max_pool_2d, args=[2])
    net = time_distributed(net, conv_2d, args=[num_filters_conv2,
                                               filter_size_conv2, 1, 'same',
                                               'tanh'])
    net = time_distributed(net, max_pool_2d, args=[2])
    net = time_distributed(net, conv_2d, args=[num_filters_conv3,
                                               filter_size_conv3, 1, 'same',
                                               'tanh'])
    net = time_distributed(net, max_pool_2d, args=[2])
    net = time_distributed(net, flatten, args=['flat'])
    net = lstm(net, lstm_units)
    fc_layer = tflearn.fully_connected(net, num_classes, activation='softmax')
    loss = tflearn.objectives.categorical_crossentropy(fc_layer, y_true)
    network = regression(fc_layer, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='gestureCNNLSTM.tfl.ckpt')

def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)

def getdiffList(imgs):
    # imgs, f = getimages(path)
    # print ('File: ', f)
    diff = []
    # os.chdir('C:\\Study\\Sem 3\\ChrisTseng\\GRIT_DATASET\\Images\\abort')
    for i, img in enumerate(imgs[2:-1]):
        im = diffImg(imgs[i-1], img, imgs[i+1])
        # print (f[i-1], ", ", f[i], ", ", f[i+1])
        # plt.figure()

        # plt.imshow(im)
        # plt.show()
        # print (im)
        # print (np.shape(im))
        # cv2.imwrite(str(i)+".jpg", im)
        diff.append(im)
    return diff

def predict(imgs):
    return 0

# def load_model():

def parse_args():
    global model_file
    model_file = str(sys.argv[1])
    # train_folder_name = str(sys.argv[2])


if __name__ == '__main__':
    # load_model()
    # parse_args()
    encoder()
    img_size = (64, 48)
    num_channels = 1
    with tf.Session() as sess:
        # x = tf.placeholder(tf.float32, shape=[None, img_size[0], img_size[1], num_channels], name='x')
        # cnn_model()
        cnnLSTM_model()
        model.load(model_file)#("gestureCNNLSTM.tfl")
        print (os.getcwd())
        # cwd = os.getcwd()
        # file = os.path.join(cwd, 'model', model_file)
        #
        # saver = tf.train.Saver()
        # saver.restore(sess, file)
        # sess.run(tf.global_variables_initializer())
        # test_accuracy = accuracy.eval(feed_dict={x: x_data, y_true: y_label})
        # print "Test accuracy: ", test_accuracy

        cap = cv2.VideoCapture(0)
        l = []
        # tf.reset_default_graph()
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            if(len(l) < 26):
                l.append(np.transpose(cv2.resize(gray, (64,48))))
            else:
                # print (np.shape(l))
                # l = np.resize(l, q(-1, 64, 48, 1))
                diff = np.resize(getdiffList(l), (-1, 64, 48, 1))
                diff = diff.reshape(1,-1,64,48,1)
                prediction = model.predict(diff)#sess.run(logits, feed_dict={x: diff})
                # print (prediction)

                # d = dict(Counter(prediction))
                # s = sorted(d.items(), key=lambda x: x[1])
                # s.reverse()
                # print (s)
                # # prediction = [convert_to_oneHot(i) for i in prediction]
                # # print (prediction)
                # prediction = [t[0] for t in s]
                pred = np.argmax(prediction)
                text = ""
                color = (50, 50, 50)
                font = "arial.ttf"#'unifont.ttf'
                img = ImageText(Image.fromarray(frame))
                if prediction[0][pred] > 0.5:
                    predict = decode([pred])
                    print (str(predict) + " " + str(prediction[0][pred]))
                    text = str(predict[0])
                    # print(prediction[0][pred])
                    # print(pred)
                else:
                    print ("Predicting...")
                    text = "Predicting..."
                # img.write_text_box((300, 125), text, box_width=200, font_filename=font,
                #                    font_size=15, color=color, place='right')
                #
                # cv2.imshow('frame+prediction', cv2.cvtColor(np.array(img.image), cv2.COLOR_RGB2BGR))

                # img.image.show()
                # print(tf.argmax(prediction, 1))
                # send(l)
                l = []

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()