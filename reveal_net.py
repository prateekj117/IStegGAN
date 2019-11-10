import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D


def reveal_net(container):
    print(container)
    conv1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        container)
    conv2 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv2)
    conv3 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv3)
    conv3 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv3)

    print(conv3)

    v1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv3)

    maxpool1 = MaxPool2D(pool_size=2, data_format='channels_last')(conv3)

    conv4 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        maxpool1)
    conv5 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv4)
    conv6 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv5)
    conv6 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv6)
    conv6 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv6)
    print(conv6)
    v2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv6)

    maxpool2 = MaxPool2D(pool_size=2, data_format='channels_last')(conv6)

    conv7 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        maxpool2)
    conv8 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv7)
    conv9 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv8)
    conv9 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv9)
    conv9 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv9)

    print(conv9)

    v3 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv9)

    maxpool3 = MaxPool2D(pool_size=2, data_format='channels_last')(conv9)

    conv10 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        maxpool3)
    conv11 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv10)
    conv12 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv11)
    conv12 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv12)
    conv12 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv12)

    print(conv12)

    v4 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv12)

    maxpool4 = MaxPool2D(pool_size=2, data_format='channels_last')(conv12)

    conv13 = Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        maxpool4)
    conv14 = Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv13)
    conv15 = Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv14)
    conv15 = Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv15)
    conv15 = Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv15)

    print(conv15)

    upsample1 = UpSampling2D(size=2, data_format='channels_last')(conv15)

    print(upsample1)
    print(v4)

    concat1 = tf.concat([v4, upsample1], axis=3, name='concat1')

    conv16 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat1)
    conv17 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv16)
    conv18 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv17)
    conv18 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv18)
    conv18 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv18)

    print(conv18)

    upsample2 = UpSampling2D(size=2, data_format='channels_last')(conv18)

    concat2 = tf.concat([v3, upsample2], axis=3, name='concat2')

    conv19 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat2)
    conv20 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv19)
    conv21 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv20)
    conv21 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv21)
    conv21 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv21)

    print(conv21)

    upsample3 = UpSampling2D(size=2, data_format='channels_last')(conv21)

    concat3 = tf.concat([v2, upsample3], axis=3, name='concat3')

    conv22 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat3)
    conv23 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv22)
    conv24 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv23)
    conv24 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv24)
    conv24 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv24)

    print(conv24)

    upsample4 = UpSampling2D(size=2, data_format='channels_last')(conv24)

    concat4 = tf.concat([v1, upsample4], axis=3, name='concat4', )

    conv25 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat4)
    conv26 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv25)
    output = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv26)
    output = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        output)
    output = Conv2D(filters=3, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        output)

    print(output)

    print("Reveal_Net")

    return output


"""
input_shape=(None, 224, 224, 3)
cover = tf.placeholder(shape=input_shape, dtype=tf.float32, name='container')   
reveal_net(container)    
"""
