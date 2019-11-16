import hide_net
import reveal_net
import tensorflow as tf
from get_batch import get_img_batch
import os
from PIL import Image
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class SingleSizeModel:
    """ A convolution model that handles only same size cover
    and secret images.
    """

    #   def get_noise_layer_op(self, tensor, std=.1):
    #       with tf.variable_scope("noise_layer"):
    #           return tensor + tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=std, dtype=tf.float32)

    def get_loss_op(self, secret_true, secret_pred, cover_true, cover_pred, beta=0.8):

        with tf.variable_scope("losses"):
            batch_size = 4
            beta = tf.constant(beta, name="beta")
            secret_mse = batch_size * (tf.losses.mean_squared_error(secret_true, secret_pred))
            cover_mse = batch_size * (tf.losses.mean_squared_error(cover_true, cover_pred))
            final_loss = cover_mse + beta * secret_mse

            return final_loss, secret_mse, cover_mse

    #def get_tensor_to_img_op(self, tensor):
    #    with tf.variable_scope("", reuse=True):
    #        t = tensor * tf.convert_to_tensor([0.229, 0.224, 0.225]) + tf.convert_to_tensor([0.485, 0.456, 0.406])
    #        return tf.clip_by_value(t, 0, 1)

    def prepare_training_graph(self, secret_tensor, cover_tensor, global_step_tensor):

        hiding_output = hide_net.hiding_net(cover_tensor, secret_tensor)
        #       noise_add_op = self.get_noise_layer_op(hiding_output_op)
        reveal_output = reveal_net.reveal_net(hiding_output)

        loss_op, secret_loss_op, cover_loss_op = self.get_loss_op(secret_tensor,
                                                                  reveal_output, cover_tensor,
                                                                  hiding_output,
                                                                  beta=self.beta)

        minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op, global_step=global_step_tensor)

        tf.summary.scalar('loss', loss_op, family='train')
        tf.summary.scalar('reveal_net_loss', secret_loss_op, family='train')
        tf.summary.scalar('cover_net_loss', cover_loss_op, family='train')

        tf.summary.image('secret', secret_tensor, max_outputs=1, family='train')
        tf.summary.image('cover', cover_tensor, max_outputs=1, family='train')
        tf.summary.image('hidden', hiding_output, max_outputs=1, family='train')
        tf.summary.image('revealed', reveal_output, max_outputs=1, family='train')

        merged_summary_op = tf.summary.merge_all()

        return minimize_op, merged_summary_op, loss_op, secret_loss_op, cover_loss_op

    def prepare_test_graph(self, secret_tensor, cover_tensor):
        with tf.variable_scope("", reuse=True):
            hiding_output = hide_net.hiding_net(cover_tensor, secret_tensor)
            reveal_output = reveal_net.reveal_net(hiding_output)

            loss_op, secret_loss_op, cover_loss_op = self.get_loss_op(secret_tensor,
                                                                      reveal_output,
                                                                      cover_tensor,
                                                                      hiding_output)

            tf.summary.scalar('loss', loss_op, family='test')
            tf.summary.scalar('reveal_net_loss', secret_loss_op, family='test')
            tf.summary.scalar('cover_net_loss', cover_loss_op, family='test')

            tf.summary.image('secret', self.secret_tensor, max_outputs=1, family='test')
            tf.summary.image('cover', cover_tensor, max_outputs=1, family='test')
            tf.summary.image('hidden', hiding_output, max_outputs=1, family='test')
            tf.summary.image('revealed', reveal_output, max_outputs=1, family='test')

            merged_summary_op = tf.summary.merge_all()

            return hiding_output, reveal_output, merged_summary_op, loss_op, secret_loss_op, cover_loss_op

    #def get_tensor_to_img_op(self, tensor):
    #    with tf.variable_scope("", reuse=True):
    #        t = tensor * tf.convert_to_tensor([0.229, 0.224, 0.225]) + tf.convert_to_tensor([0.485, 0.456, 0.406])
    #        return tf.clip_by_value(t, 0, 1)

    def __init__(self, beta, log_path, input_shape=(None, 224, 224, 3), input_shape1=(None, 224, 224, 3)):

        self.beta = beta
        self.i = 0
        self.j = 0
        self.learning_rate = 0.0001
        self.sess = tf.InteractiveSession()

        self.secret_tensor = tf.placeholder(shape=input_shape1, dtype=tf.float32, name="input_prep")
        self.cover_tensor = tf.placeholder(shape=input_shape, dtype=tf.float32, name="input_hide")
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        self.train_op, self.summary_op, self.loss_op, self.secret_loss_op, self.cover_loss_op = self.prepare_training_graph(
            self.secret_tensor, self.cover_tensor, self.global_step_tensor)

        self.writer = tf.summary.FileWriter(log_path, self.sess.graph)

        self.hiding_output_op, self.reveal_output_op, self.summary_op, self.loss_op, self.secret_loss_op, self.cover_loss_op = self.prepare_test_graph(
            self.secret_tensor, self.cover_tensor)

        self.sess.run(tf.global_variables_initializer())

        print("OK")

    def make_chkp(self, saver, path):
        global_step = self.sess.run(self.global_step_tensor)
        saver.save(self.sess, path, global_step)

    def load_chkp(self, saver, path):
        print("LOADING")
        global_step = self.sess.run(self.global_step_tensor)
        tf.reset_default_graph()
        imported_meta = tf.train.import_meta_graph(
            "./Wnet/Checkpoints/my-model.ckpt-1201.meta")
        imported_meta.restore(self.sess,
                              tf.train.latest_checkpoint('./Wnet/Checkpoints/'))
        # saver.restore(self.sess, path)
        print("LOADED")

    def train(self, saver, steps, files_list, batch_size):

        for step in range(steps):
            print('epoch:', step)
            p = 0
            covers, secrets = get_img_batch(files_list, batch_size, p)
            self.sess.run([self.train_op], feed_dict={"input_prep:0": secrets, "input_hide:0": covers})

            if step % 10 == 0:
                summary, global_step, total_loss, secret_loss, cover_loss = self.sess.run(
                    [self.summary_op, self.global_step_tensor, self.loss_op, self.secret_loss_op, self.cover_loss_op],
                    feed_dict={"input_prep:0": secrets, "input_hide:0": covers})
                self.writer.add_summary(summary, global_step)
                print("total loss at step %s: %s" % (step, total_loss))
                print("cover loss at step %s: %s" % (step, cover_loss))
                print("secret loss at step %s: %s" % (step, secret_loss))
            if step % 2000 == 0: 
                self.make_chkp(saver, "./Wnet/Checkpoints/my-model.ckpt")

    def test(self, saver, files_list, batch_size, path):
        self.load_chkp(saver, path)
        for step in range(1):
            print("Epoch:\n", step)
            p = 1
            covers, secrets = get_img_batch(files_list, batch_size, p)
            self.sess.run(
                [self.hiding_output_op, self.reveal_output_op, self.summary_op, self.loss_op, self.secret_loss_op,
                 self.cover_loss_op],
                feed_dict={"input_prep:0": secrets, "input_hide:0": covers})
            hiding_output_op, reveal_output_op, summary, total_loss, cover_loss, secret_loss = self.sess.run(
                [self.hiding_output_op, self.reveal_output_op, self.summary_op, self.loss_op, self.cover_loss_op,
                 self.secret_loss_op],
                feed_dict={"input_prep:0": secrets, "input_hide:0": covers})
            self.writer.add_summary(summary)

            # hiding_output_op = self.get_tensor_to_img_op(hiding_output_op)

            for k in range(batch_size):
                im = tf.reshape(tf.cast(hiding_output_op[k], tf.uint8), [224, 224, 3])
                images_encode = tf.image.encode_jpeg(im)
                fname = tf.constant('%s.jpeg' % self.i)
                self.i += 1
                fwrite = tf.write_file(fname, images_encode)

            for k in range(batch_size):
                im = Image.fromarray(np.uint8((hiding_output_op[k]) * 255))
                im.save('./Wnet/container/%s.jpg' % self.i)
                self.i += 1

            for k in range(batch_size):
                im = Image.fromarray(np.uint8((reveal_output_op[k]) * 255))
                im.save('./Wnet/secret/%s.jpg' % self.j)
                self.j += 1

            print("total loss at step %s: %s" % (step, total_loss))
            print("cover loss at step %s: %s" % (step, cover_loss))
            print("secret loss at step %s: %s" % (step, secret_loss))


m = SingleSizeModel(beta=0.8, log_path="./Wnet/logs")
saver = tf.train.Saver()
train_list = os.listdir('./imagenet50k/train')
test_list = os.listdir('./imagenet50k/test')
m.train(saver, 50001, train_list, 4)
#m.test(saver, test_list, 1, './Wnet/Checkpoints/my-model.ckpt')
