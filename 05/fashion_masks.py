#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None
        self._masks = data["masks"] if "masks" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def masks(self):
        return self._masks

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None, self._masks[batch_perm] if self._masks is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(len(self._images))
            return True
        return False

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10


    # from     # from http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
    def get_bilinear_filter(self,filter_shape, upscale_factor):
        ##filter_shape is [width, height, num_in_channels, num_out_channels]
        kernel_size = filter_shape[1]
        ### Centre location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5

        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location) / upscale_factor)) * (
                            1 - abs((y - centre_location) / upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                           shape=weights.shape)
        return bilinear_weights

    # from http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
    def upsample_layer(self,bottom,
                       n_channels, name, upscale_factor):

        kernel_size = 2 * upscale_factor - upscale_factor % 2
        stride = upscale_factor
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            # Shape of the bottom tensor
            in_shape = tf.shape(bottom)

            h = ((in_shape[1]) * stride)
            w = ((in_shape[2]) * stride)
            new_shape = [in_shape[0], h, w, n_channels]
            output_shape = tf.stack(new_shape)

            filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

            weights = self.get_bilinear_filter(filter_shape, upscale_factor)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

        return deconv
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
                hidden_layer = tf.layers.conv2d(hidden_layer, filters, kernel_size, stride, padding,
                                                activation=tf.nn.relu)
            elif (t_args[0] == 'M'):
                kernel_size = int(t_args[1])
                stride = int(t_args[2])
                hidden_layer = tf.layers.max_pooling2d(hidden_layer, kernel_size, stride, padding='same')
            elif (t_args[0] == 'F'):
                hidden_layer = tf.layers.flatten(hidden_layer, name="flatten")
            elif (t_args[0] == 'R'):
                hidden_layer_size = t_args[1]
                hidden_layer = tf.layers.dense(hidden_layer, hidden_layer_size, activation=tf.nn.relu)
            elif (t_args[0] == 'CB'):
                filters = t_args[1]
                kernel_size = int(t_args[2])
                stride = int(t_args[3])
                padding = t_args[4]
                hidden_layer = tf.layers.conv2d(hidden_layer, filters, kernel_size, stride, padding,
                                                activation=None, use_bias=False)
                hidden_layer = tf.layers.batch_normalization(hidden_layer, training=self.is_training)
                hidden_layer = tf.nn.relu(hidden_layer)
            elif (t_args[0] == 'CU'):
                filters = int(t_args[1])
                upscale_factor = int(t_args[2])
                hidden_layer = self.upsample_layer(hidden_layer,filters,'test0',upscale_factor)
        return hidden_layer


    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.masks = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="masks")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.labels_predictions` of shape [None] and type tf.int64
            # - mask predictions are stored in `self.masks_predictions` of shape [None, 28, 28, 1] and type tf.float32
            #   with values 0 or 1

            hidden_layer = self.images
            hidden_layer_s = self.build_network_part(hidden_layer,args.cnn_common)
            final_layer_masks = self.build_network_part(hidden_layer_s,args.cnn_masks)
            final_layer_labels = self.build_network_part(hidden_layer_s,args.cnn_labels)

            output_layer_labels = tf.layers.dense(final_layer_labels, self.LABELS, activation=None, name="output_layer")
            self.labels_predictions = tf.argmax(output_layer_labels, axis=1)
            loss_labels = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer_labels, scope="loss")

            #This is because I do not know how to use tf.nn.sparse_softmax_cross_entropy_with_logits
            predictions_for_cross_entropy = tf.reshape(final_layer_masks, [-1, self.HEIGHT, self.WIDTH, 2])
            loss_masks = tf.losses.sparse_softmax_cross_entropy(tf.cast(self.masks, tf.int64), predictions_for_cross_entropy, scope="mask_loss")


            loss = tf.add(loss_labels,loss_masks)

            # Training
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                global_step = tf.train.create_global_step()
                if args.learning_rate_final == None:
                    learning_rate = args.learning_rate
                else:
                    decay_rate = (args.learning_rate_final / args.learning_rate) ** (1 / (args.epochs - 1))
                    decay_steps = len(train._images) // args.batch_size
                    learning_rate = tf.train.exponential_decay(staircase=True, learning_rate=args.learning_rate,
                                                               global_step=global_step, decay_rate=decay_rate,
                                                               decay_steps=decay_steps)
                self.training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step, name="training")

            # Reformating data for the Sumaries
            self.masks_predictions = tf.argmax(predictions_for_cross_entropy, axis=3)
            self.masks_predictions = tf.cast(self.masks_predictions,tf.float32)
            self.masks_predictions = tf.reshape(self.masks_predictions,[-1,28,28,1])
            # Summaries
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.labels_predictions), tf.float32))
            only_correct_masks = tf.where(tf.equal(self.labels, self.labels_predictions),
                                          self.masks_predictions, tf.zeros_like(self.masks_predictions))
            intersection = tf.reduce_sum(only_correct_masks * self.masks, axis=[1,2,3])
            iou = tf.reduce_mean(
                intersection / (tf.reduce_sum(only_correct_masks, axis=[1,2,3]) + tf.reduce_sum(self.masks, axis=[1,2,3]) - intersection)
            )

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", accuracy),
                                           tf.contrib.summary.scalar("train/iou", iou),
                                           tf.contrib.summary.image("train/images", self.images),
                                           tf.contrib.summary.image("train/masks", self.masks_predictions)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset+"/loss", loss),
                                               tf.contrib.summary.scalar(dataset+"/accuracy", accuracy),
                                               tf.contrib.summary.scalar(dataset+"/iou", iou),
                                               tf.contrib.summary.image(dataset+"/images", self.images),
                                               tf.contrib.summary.image(dataset+"/masks", self.masks_predictions)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels, masks):
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})

    def evaluate(self, dataset, images, labels, masks):
        self.session.run(self.summaries[dataset],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: False})

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.masks_predictions],
                                {self.images: images, self.is_training: False})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--learning_rate_final", default=0.0005, type=float, help="Final learning rate.")
    parser.add_argument("--cnn-common", default='CB-64-3-1-same,CB-64-3-1-same,M-3-2,CB-256-3-1-same,CB-256-3-1-same,M-3-2', type=str, help="Description of the CNN architecture.")
    parser.add_argument("--cnn-masks", default='CB-256-3-1-same,CB-256-3-1-same,CU-256-4,C-2-1-1-same', type=str, help="Description of the CNN architecture.")
    parser.add_argument("--cnn-labels", default='F,R-1024', type=str, help="Description of the CNN architecture.")
    #parameters "--epochs 30 --batch_size 180 --cnn-masks CB-256-3-1-same,CB-256-3-1-same,CU-256-4,CB-2-1-1-same"

    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                  for key, value in sorted(vars(args).items()))).replace("/", "-")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("fashion-masks-train.npz")
    dev = Dataset("fashion-masks-dev.npz")
    test = Dataset("fashion-masks-test.npz", shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            images, labels, masks = train.next_batch(args.batch_size)
            network.train(images, labels, masks)

        network.evaluate("dev", dev.images, dev.labels, dev.masks)

    # Predict test data
    with open("{}/fashion_masks_test.txt".format(args.logdir), "w") as test_file:
        while not test.epoch_finished():
            images, _, _ = test.next_batch(args.batch_size)
            labels, masks = network.predict(images)
            for i in range(len(labels)):
                print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)
