import tensorflow as tf
import os

from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121, DenseNet201, VGG16, VGG19, ResNet50, ResNet152, Xception
from tensorflow.keras import Model, layers



class pifia_network(Model):
    def __init__(self, num_classes, k=1, dropout_rate=0, \
                 dense1_size=512, num_features=512, last_block=True):
        super(pifia_network, self).__init__()
        self.dropout_rate = dropout_rate
        self.last_block = last_block

        self.conv1 = Conv2D(filters=int(64*k), kernel_size=(3,3), strides=(1, 1), padding='valid', activation=None)
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')

        self.conv2 = Conv2D(filters=int(64*k), kernel_size=(3,3), strides=(1, 1), padding='valid', activation=None)
        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')

        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        self.conv3 = Conv2D(filters=int(128*k), kernel_size=(3,3), strides=(1, 1), padding='valid', activation=None)
        self.bn3 = BatchNormalization()
        self.relu3 = Activation('relu')


        self.conv4 = Conv2D(filters=int(128*k), kernel_size=(3,3), strides=(1, 1), padding='valid', activation=None)
        self.bn4 = BatchNormalization()
        self.relu4 = Activation('relu')

        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        self.conv5 = Conv2D(filters=int(256*k), kernel_size=(3,3), strides=(1, 1), padding='valid', activation=None)
        self.bn5 = BatchNormalization()
        self.relu5 = Activation('relu')

        self.conv6 = Conv2D(filters=int(256*k), kernel_size=(3,3), strides=(1, 1), padding='valid', activation=None)
        self.bn6 = BatchNormalization()
        self.relu6 = Activation('relu')

        self.conv7 = Conv2D(filters=int(256*k), kernel_size=(3,3), strides=(1, 1), padding='valid', activation=None)
        self.bn7 = BatchNormalization()
        self.relu7 = Activation('relu')

        self.conv8 = Conv2D(filters=int(256*k), kernel_size=(3,3), strides=(1, 1), padding='valid', activation=None)
        self.bn8 = BatchNormalization()
        self.relu8 = Activation('relu')

        self.pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        self.flatten = Flatten()
        self.d1 = Dense(dense1_size, activation='relu')
        self.d2 = Dense(num_features, activation='relu')
        self.dropout = Dropout(self.dropout_rate)
        self.d3 = Dense(num_classes)

    def nn(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.pool2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        if self.last_block:
            x = self.conv6(x)
            x = self.bn6(x)
            x = self.relu6(x)

            x = self.conv7(x)
            x = self.bn7(x)
            x = self.relu7(x)

            x = self.conv8(x)
            x = self.bn8(x)
            x = self.relu8(x)

        x = self.pool3(x)

        x = self.flatten(x)

        return x


    def call(self, x):
        x = self.nn(x)
        x = self.d1(x)
        x = self.d2(x)
        if self.dropout_rate>0:
            x = self.dropout(x)
        x = self.d3(x)
        return x


    def get_features(self, x, layer='d2'):
        x = self.nn(x)
        x = self.d1(x)
        if layer=='d1':
            return x

        x = self.d2(x)
        if layer=='d2':
            return x

        x = self.d3(x)
        if layer=='d3':
            return x
        return x





class custom_network(Model):
    def __init__(self, num_classes, num_channels=1, dropout_rate=0, \
                 backbone='dense_net_121', dense1_size=512, num_features=512, \
                 pool=False, h=64, w=64):
        super(custom_network, self).__init__()
        self.dropout_rate = dropout_rate
        self.dense1_size = dense1_size
        self.avg_pool = pool

        if backbone == 'dense_net_121':
            self.nn = DenseNet121(weights= None, include_top=False, input_shape= (h,w,num_channels))
        if backbone == 'dense_net_201':
            self.nn = DenseNet201(weights= None, include_top=False, input_shape= (h,w,num_channels))
        if backbone == 'vgg_16':
            self.nn = VGG16(weights= None, include_top=False, input_shape= (h,w,num_channels))
        if backbone == 'vgg_19':
            self.nn = VGG19(weights= None, include_top=False, input_shape= (h,w,num_channels))
        if backbone == 'res_net_50':
            self.nn = ResNet50(weights= None, include_top=False, input_shape= (h,w,num_channels))
        if backbone == 'res_net_152':
            self.nn = ResNet50(weights= None, include_top=False, input_shape= (h,w,num_channels))
        #if backbone == 'efficient_net':
        #    self.nn = EfficientNetB6(weights= None, include_top=False, input_shape= (h,w,num_channels))
        if backbone == 'xception':
            self.nn = Xception(weights= None, include_top=False, input_shape= (h,w,num_channels))

        if self.avg_pool:
            self.global_avg_pool = GlobalAveragePooling2D()
        self.flatten = Flatten()
        if self.dense1_size>0:
            self.d1 = Dense(dense1_size, activation='relu')
        self.d2 = Dense(num_features, activation='relu')
        self.dropout = Dropout(self.dropout_rate)
        self.d3 = Dense(num_classes)

    def call(self, x):
        x = self.nn(x)
        if self.avg_pool:
            x = self.global_avg_pool(x)
        x = self.flatten(x)
        if self.dense1_size>0:
            x = self.d1(x)
        x = self.d2(x)
        if self.dropout_rate>0:
            x = self.dropout(x)
        return self.d3(x)


    def get_features(self, x, layer='d2'):
        x = self.nn(x)
        if self.avg_pool:
            x = self.global_avg_pool(x)
        x = self.flatten(x)
        if layer=='flatten':
            return x

        if self.dense1_size>0:
            x = self.d1(x)

        if layer=='d1':
            return x

        if layer=='d2':
            x = self.d2(x)
            return x
