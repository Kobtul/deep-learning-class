#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._voxels = data["voxels"]
        self._labels = data["labels"] if "labels" in data else None

        self._shuffle_batches = shuffle_batches
        self._new_permutation()

    def _new_permutation(self):
        if self._shuffle_batches:
            self._permutation = np.random.permutation(len(self._voxels))
        else:
            self._permutation = np.arange(len(self._voxels))

    def split(self, ratio):
        split = int(len(self._voxels) * ratio)

        first, second = Dataset.__new__(Dataset), Dataset.__new__(Dataset)
        first._voxels, second._voxels = self._voxels[:split], self._voxels[split:]
        if self._labels is not None:
            first._labels, second._labels = self._labels[:split], self._labels[split:]
        else:
            first._labels, second._labels = None, None

        for dataset in [first, second]:
            dataset._shuffle_batches = self._shuffle_batches
            dataset._new_permutation()

        return first, second

    @property
    def voxels(self):
        return self._voxels

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._voxels[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._new_permutation()
            return True
        return False


class Network:
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))
    def build_network_part(self,start_layer,network):
        hidden_layer = start_layer
        for element in network.split(','):
            t_args = element.split('-')
            if (t_args[0] == 'C'):
                filters = t_args[1]
                kernel_size = int(t_args[2])
                stride = int(t_args[3])
                padding = t_args[4]
                hidden_layer = tf.layers.conv3d(hidden_layer, filters, kernel_size, stride, padding,
                                                activation=tf.nn.relu)
            elif (t_args[0] == 'M'):
                kernel_size = int(t_args[1])
                stride = int(t_args[2])
                hidden_layer = tf.layers.max_pooling3d(hidden_layer, kernel_size, stride, padding='same')
            elif (t_args[0] == 'F'):
                hidden_layer = tf.layers.flatten(hidden_layer, name="flatten")
            elif (t_args[0] == 'R'):
                hidden_layer_size = t_args[1]
                hidden_layer = tf.layers.dense(hidden_layer, hidden_layer_size, activation=tf.nn.relu)
            elif(t_args[0] == 'D'):
                hidden_layer = tf.layers.dropout(inputs=hidden_layer, rate=0.5, training=self.is_training)
            elif (t_args[0] == 'CB'):
                filters = t_args[1]
                kernel_size = int(t_args[2])
                stride = int(t_args[3])
                padding = t_args[4]
                hidden_layer = tf.layers.conv3d(hidden_layer, filters, kernel_size, stride, padding,
                                                activation=None, use_bias=False)
                hidden_layer = tf.layers.batch_normalization(hidden_layer, training=self.is_training)
                hidden_layer = tf.nn.relu(hidden_layer)
        return hidden_layer
    def cnn_model(self,x_train_data,training=True, keep_rate=0.7, seed=None):
        with tf.name_scope("layer_a"):
            # conv => 32*32*32
            conv1 = tf.layers.conv3d(inputs=x_train_data, filters=16, kernel_size=[3, 3, 3], padding='same',
                                     activation=tf.nn.relu)
            # conv => 32*32*32
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3, 3, 3], padding='same',
                                     activation=tf.nn.relu)
            # pool => 16*16*16
            pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
        with tf.name_scope("layer_b"):
            # conv => 16*16*16
            conv4 = tf.layers.conv3d(inputs=pool3, filters=64, kernel_size=[3, 3, 3], padding='same',
                                     activation=tf.nn.relu)
            # conv => 16*16*16
            conv5 = tf.layers.conv3d(inputs=conv4, filters=128, kernel_size=[3, 3, 3], padding='same',
                                     activation=tf.nn.relu)
            # pool => 8*8*8
            pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2, 2, 2], strides=2)
        # with tf.name_scope("layer_c"):
        #     # conv => 8*8*8
        #     conv7 = tf.layers.conv3d(inputs=pool6, filters=256, kernel_size=[3, 3, 3], padding='same',
        #                              activation=tf.nn.relu)
        #     # conv => 8*8*8
        #     conv8 = tf.layers.conv3d(inputs=conv7, filters=512, kernel_size=[3, 3, 3], padding='same',
        #                              activation=tf.nn.relu)
        #     # pool => 4*4*4
        #     pool9 = tf.layers.max_pooling3d(inputs=conv8, pool_size=[2, 2, 2], strides=2)

        with tf.name_scope("batch_norm"):
            cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=training)

        with tf.name_scope("fully_con"):
            flattening = tf.reshape(cnn3d_bn, [-1, 5 * 5 * 5 * 128])
            dense = tf.layers.dense(inputs=flattening, units=1024, activation=tf.nn.relu)
            # (1-keep_rate) is the probability that the node will be kept
            dropout = tf.layers.dropout(inputs=dense, rate=keep_rate, training=training)

        with tf.name_scope("y_conv"):
            y_conv = tf.layers.dense(inputs=dropout, units=self.LABELS)

        return y_conv
    def construct(self, args):
        with self.session.graph.as_default():
            #stride 4 konvoluce, a convoluce 7
            #od 4 vrstev RNN teprve resiudal connection
            # Inputs
            self.voxels = tf.placeholder(
                tf.float32, [None, args.modelnet_dim, args.modelnet_dim, args.modelnet_dim, 1], name="voxels")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `self.loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`
            #last_layer = self.cnn_model(self.voxels,training=self.is_training)
            last_layer = self.build_network_part(self.voxels,args.cnn)
            last_layer = tf.layers.dense(last_layer, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(last_layer, axis=1)
            loss= tf.losses.sparse_softmax_cross_entropy(self.labels, last_layer, scope="loss")

            # self.predictions = self.cnn_model(self.voxels, keep_rate=0.7, seed=1)
            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.labels))
            # self.predictions = tf.cast(self.predictions,tf.float32)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                global_step = tf.train.create_global_step()
                if args.learning_rate_final == None:
                    learning_rate = args.learning_rate
                else:
                    decay_rate = (args.learning_rate_final / args.learning_rate) ** (1 / (args.epochs - 1))
                    decay_steps = len(train._voxels) // args.batch_size
                    learning_rate = tf.train.exponential_decay(staircase=True, learning_rate=args.learning_rate,
                                                               global_step=global_step, decay_rate=decay_rate,
                                                               decay_steps=decay_steps)
                self.training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,
                                                                                             global_step=global_step,
                                                                                             name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(8):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name="given_loss")
                self.given_accuracy = tf.placeholder(tf.float32, [], name="given_accuracy")
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.given_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_batch(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.voxels: voxels, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        loss, accuracy = 0, 0

        while not dataset.epoch_finished():
            batch_voxels, batch_labels = dataset.next_batch(batch_size)
            batch_loss, batch_accuracy = self.session.run(
                [self.loss, self.accuracy], {self.voxels: batch_voxels, self.labels: batch_labels, self.is_training: False})
            loss += batch_loss * len(batch_voxels) / len(dataset.voxels)
            accuracy += batch_accuracy * len(batch_voxels) / len(dataset.voxels)
        self.session.run(self.summaries[dataset_name], {self.given_loss: loss, self.given_accuracy: accuracy})

        return accuracy

    def predict(self, dataset, batch_size):
        labels = []
        while not dataset.epoch_finished():
            voxels, _ = dataset.next_batch(batch_size)
            labels.append(self.session.run(self.predictions, {self.voxels: voxels, self.is_training: False}))
        return np.concatenate(labels)

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=220, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--modelnet_dim", default=20, type=int, help="Dimension of ModelNet data.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--train_split", default=0.7, type=float, help="Ratio of examples to use as train.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--learning_rate_final", default=0.0005, type=float, help="Final learning rate.")
    parser.add_argument("--cnn", default='CB-64-3-1-same,CB-64-3-1-same,M-3-2,CB-256-3-1-same,CB-256-3-1-same,M-3-2,F,R-1024,D', type=str, help="Description of the CNN architecture.")
    #final parameters: "--epochs 40 --batch_size 60 --learning_rate 0.001 --learning_rate_final 0.0005 --cnn CB-64-7-4-same,CB-64-3-1-same,M-3-2,F,R-1024"
    args = parser.parse_args()
    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train, dev = Dataset("modelnet{}-train.npz".format(args.modelnet_dim)).split(args.train_split)
    test = Dataset("modelnet{}-test.npz".format(args.modelnet_dim), shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            voxels, labels = train.next_batch(args.batch_size)
            network.train_batch(voxels, labels)
        network.evaluate("dev", dev, args.batch_size)

    # Predict test data
    with open("{}/3d_recognition_test.txt".format(args.logdir), "w") as test_file:
        labels = network.predict(test, args.batch_size)
        for label in labels:
            print(label, file=test_file)
