import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D


def hiding_net(cover, secret):
    concat_input = tf.concat([cover, secret], axis=3, name='concat')
    conv1_1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat_input)
    conv1_2 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv1_1)
    conv1_3 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv1_2)
    conv1_4 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv1_3)
    conv1_5 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv1_4)

    v1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv1_5)

    maxpool1 = MaxPool2D(pool_size=2, data_format='channels_last')(conv1_5)

    conv2_1 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        maxpool1)
    conv2_2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv2_1)
    conv2_3 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv2_2)
    conv2_4 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv2_3)
    conv2_5 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv2_4)

    v2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv2_5)

    maxpool2 = MaxPool2D(pool_size=2, data_format='channels_last')(conv2_5)

    conv3_1 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        maxpool2)
    conv3_2 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv3_1)
    conv3_3 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv3_2)
    conv3_4 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv3_3)
    conv3_5 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv3_4)

    v3 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv3_5)

    maxpool3 = MaxPool2D(pool_size=2, data_format='channels_last')(conv3_5)

    conv4_1 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        maxpool3)
    conv4_2 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv4_1)
    conv4_3 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv4_2)
    conv4_4 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv4_3)
    conv4_5 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv4_4)

    v4 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv4_5)

    maxpool4 = MaxPool2D(pool_size=2, data_format='channels_last')(conv4_5)

    conv5_1 = Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        maxpool4)
    conv5_2 = Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv5_1)
    conv5_3 = Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv5_2)
    conv5_4 = Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv5_3)
    conv5_5 = Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv5_4)

    upsample1 = UpSampling2D(size=2, data_format='channels_last')(conv5_5)

    concat1 = tf.concat([v4, upsample1], axis=3, name='concat1')

    conv6_1 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat1)
    conv6_2 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv6_1)
    conv6_3 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv6_2)
    conv6_4 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv6_3)
    conv6_5 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv6_4)

    upsample2 = UpSampling2D(size=2, data_format='channels_last')(conv6_5)

    v6 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv6_5)

    concat2 = tf.concat([v3, upsample2], axis=3, name='concat2')

    conv7_1 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat2)
    conv7_2 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv7_1)
    conv7_3 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv7_2)
    conv7_4 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv7_3)
    conv7_5 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv7_4)

    maxpool5 = MaxPool2D(pool_size=2, data_format='channels_last')(conv7_5)

    v7 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv7_5)

    concat3 = tf.concat([v4, v6, maxpool5], axis=3, name='concat3')

    conv8_1 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat3)
    conv8_2 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv8_1)
    conv8_3 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv8_2)
    conv8_4 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv8_3)
    conv8_5 = Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv8_4)

    upsample3 = UpSampling2D(size=2, data_format='channels_last')(conv8_5)

    concat4 = tf.concat([v3, v7, upsample3], axis=3, name='concat4')

    conv9_1 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat4)
    conv9_2 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv9_1)
    conv9_3 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv9_2)
    conv9_4 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv9_3)
    conv9_5 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv9_4)

    v9 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv9_5)

    upsample4 = UpSampling2D(size=2, data_format='channels_last')(conv9_5)

    concat5 = tf.concat([v2, upsample4], axis=3, name='concat5')

    conv10_1 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat5)
    conv10_2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv10_1)
    conv10_3 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv10_2)
    conv10_4 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv10_3)
    conv10_5 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv10_4)

    maxpool6 = MaxPool2D(pool_size=2, data_format='channels_last')(conv10_5)

    v10 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(conv10_5)

    concat6 = tf.concat([v3, v7, v9, maxpool6], axis=3, name='concat6')

    conv11_1 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat6)
    conv11_2 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv11_1)
    conv11_3 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv11_2)
    conv11_4 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv11_3)
    conv11_5 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv11_4)

    upsample5 = UpSampling2D(size=2, data_format='channels_last')(conv11_5)

    concat7 = tf.concat([v2, v10, upsample5], axis=3, name='concat7')

    conv12_1 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat7)
    conv12_2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv12_1)
    conv12_3 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv12_2)
    conv12_4 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv12_3)
    conv12_5 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv12_4)

    upsample6 = UpSampling2D(size=2, data_format='channels_last')(conv12_5)

    concat8 = tf.concat([v1, upsample6], axis=3, name='concat8')

    conv13_1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        concat8)
    conv13_2 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv13_1)
    conv13_3 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv13_2)
    conv13_4 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv13_3)
    output = Conv2D(filters=3, kernel_size=3, padding='same', activation=tf.nn.relu, data_format='channels_last')(
        conv13_4)

    print("Hiding_Net")

    return output


"""
input_shape=(None, 224, 224, 3)
cover = tf.placeholder(shape=input_shape, dtype=tf.float32, name='cover_input')
secret = tf.placeholder(shape=input_shape, dtype=tf.float32, name='secret_input')   
hiding_net(cover,secret)    
"""
