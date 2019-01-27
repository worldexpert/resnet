from utils import *
import time
from datetime import datetime
#import pandas as pd
#import itertools



num_residual_blocks = 5
batch_size = 256

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('version', 'compareTest', '''A version number defining the directory to save
logs and checkpoints''')
tf.app.flags.DEFINE_string('optimizer', 'Adam', '''Opimizer : AdamOptimizer/ MomentumOptimizer / RMSPropOptimizer''')

train_dir = './model_' + FLAGS.version + FLAGS.optimizer
log_dir = './log_' + FLAGS.version + FLAGS.optimizer


class Train:
    def __init__(self) -> None:
        self.placeholders()

    def placeholders(self) -> None:
        self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, 10])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, 10])

        self.learningrate_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
        self.istraining = tf.placeholder(dtype=tf.bool, shape=[])

    '''
    learning rate decay : 
    '''

    def build_train_validation_graph(self, opt, global_step):
        '''
        This function builds the train graph and validation graph at the same time.

        '''

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph

        #model = []
        with tf.variable_scope('conv0'):
            #in_channel = self.image_placeholder.get_shape().as_list()[-1]
            #bn_layer = batch_normalization_layer(self.image_placeholder, in_channel)
            bn_layer= tf.layers.batch_normalization(self.image_placeholder, training=self.istraining)
            conv0 = tf.layers.conv2d(bn_layer, 16, [3, 3], activation= tf.nn.relu, padding='SAME')
            activation_summary(conv0)

        for i in range(num_residual_blocks):
            with tf.variable_scope('conv1_%d' % i):
                if i == 0:
                    conv1 = residual_block(conv0, 16, self.istraining, first_block=True)
                else:
                    conv1 = residual_block(conv1, 16, self.istraining)
                activation_summary(conv1)


        for i in range(num_residual_blocks):
            with tf.variable_scope('conv2_%d' % i):
                conv2 = residual_block(conv1, 32, self.istraining)
                activation_summary(conv2)

        for i in range(num_residual_blocks):
            with tf.variable_scope('conv3_%d' % i):
                conv3 = residual_block(conv2, 64, self.istraining)
            assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

        with tf.variable_scope('fc'): #fully connected layer
            #in_channel = conv3.get_shape().as_list()[-1]
            #bn_layer = batch_normalization_layer(conv3, in_channel)
            bn_layer= tf.layers.batch_normalization(conv3, training=self.istraining)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])
            #global_pool = tf.contrib.layers.flatten(relu_layer)

            #assert global_pool.get_shape().as_list()[-1:] == [64]
            model = tf.layers.dense(global_pool, 10, activation= None, name='model') # in 10 classes


        # one hot encode ??

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=self.label_placeholder))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            optimizers= {'adam': tf.train.AdamOptimizer(learning_rate=self.learningrate_placeholder).minimize(cost,global_step=global_step),
                'mome': tf.train.MomentumOptimizer(learning_rate=self.learningrate_placeholder, momentum=0.9).minimize(cost,global_step=global_step),
                'rmsp': tf.train.RMSPropOptimizer(learning_rate=self.learningrate_placeholder, momentum=0.9).minimize(cost,global_step=global_step)}
            optimizer = optimizers.get(opt)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss


        return optimizer, cost, model


    def train(self, init_lr, steps, opt) :

        training_data, training_labels = read_training_data()
        test_ori_data, test_labels = read_validation_data()
        test_data = whitening_image(test_ori_data)

        print('training_data.shape : ', training_data.shape)
        print('training_labels.shape : ', training_labels.shape)

        print('test_data.shape : ', test_data.shape)
        print('test_labels.shape : ', test_labels.shape)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer, cost, model = self.build_train_validation_graph(opt, global_step)
        tf.summary.scalar('cost', cost)

        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(self.vali_label_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())

            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restored from checkpoint...')
            else:
                sess.run(tf.global_variables_initializer())

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            #step_list = []
            #train_error_list = []
            #val_error_list = []

            for step in range(steps+1):  #
                report_freq= 500

                train_batch_data, train_batch_labels = generate_augment_train_batch(training_data, training_labels,
                                                                            batch_size)

                start_time = time.time()

                _, train_loss_value = sess.run([optimizer, cost],
                                                    feed_dict={self.image_placeholder: train_batch_data,
                                                    self.label_placeholder: train_batch_labels,
                                                    self.learningrate_placeholder: init_lr,
                                                    self.istraining : True })
                duration = time.time() - start_time


                if step % report_freq == 0:
                    summary_str = sess.run(summary_op, feed_dict={self.image_placeholder: train_batch_data,
                                                     self.label_placeholder: train_batch_labels,
                                                     self.learningrate_placeholder: init_lr,
                                                    self.istraining : True })
                    summary_writer.add_summary(summary_str, global_step=sess.run(global_step))

                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                    print(sess.run(global_step))
                    print(format_str % (datetime.now(), train_loss_value, examples_per_sec,
                                        sec_per_batch))
                    #print('Train top1 error = ', train_error_value)
                    #print('Validation top1 error = %.4f' % validation_error_value)
                    print('Validation loss = ', train_loss_value)
                    print('----------------------------')

                    #step_list.append(step)
                    #train_error_list.append(train_error_value)

                if step % 2000 == 0:

                    # sess = tf.Session()
                    result1 = sess.run(accuracy, feed_dict={self.image_placeholder: test_data,
                                                           self.vali_label_placeholder: test_labels,
                                                    self.istraining : True })
                    print('Accuracy1 :', result1)

                    result2 = sess.run(accuracy, feed_dict={self.image_placeholder: test_data,
                                                           self.vali_label_placeholder: test_labels,
                                                    self.istraining : False })
                    print('Accuracy2 :', result2)


                if step == steps*0.5 or step == steps*0.8 :
                    init_lr = 0.1 * init_lr
                    print('Learning rate decayed to ', init_lr)



            saver.save(sess, train_dir+'/resnet.ckpt', global_step=global_step )
            '''
            df = pd.DataFrame(data={'step': step_list, 'train_error': train_error_list,
                                    'validation_error': val_error_list})
            df.to_csv(train_dir + '/error.csv')
            '''

    #   def test(self):
            validation_step = tf.Variable(0, trainable=False, name='validation_step')
            #model = tf.get_variable("model", [1])


    def test(self) :
        global_step = tf.Variable(0, trainable=False, name='global_step')
        test_ori_data, test_labels = read_validation_data()
        test_data = whitening_image(test_ori_data)

        print('test_data.shape : ', test_data.shape)
        print('test_labels.shape : ', test_labels.shape)

        optimizer, cost, model = self.build_train_validation_graph(global_step)

        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(self.vali_label_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables())

            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restored from checkpoint...')

            # sess = tf.Session()
            result = sess.run(accuracy, feed_dict={self.image_placeholder: test_data,
                                                    self.vali_label_placeholder: test_labels,
                                                    self.istraining: False})
            print('Accuracy :', result)

            labels = sess.run(model, feed_dict={self.image_placeholder: test_data,
                                                self.istraining: False})

        display_random_image(test_ori_data, labels)

train = Train()
train.train(0.1, 80000, 'rmsp') #'adam', 'mome'

#train.test()