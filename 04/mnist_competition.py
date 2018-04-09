#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            hidden_layer = self.images
            for element in args.cnn.split(','):
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
                    hidden_layer = tf.layers.max_pooling2d(hidden_layer, kernel_size, stride)
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

            output_layer = tf.layers.dense(hidden_layer, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")



            # Training
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                global_step = tf.train.create_global_step()
                if args.learning_rate_final == None:
                    learning_rate = args.learning_rate
                else:
                    decay_rate = (args.learning_rate_final / args.learning_rate) ** (1 / (args.epochs - 1))
                    decay_steps = mnist.train.num_examples // args.batch_size
                    learning_rate = tf.train.exponential_decay(staircase=True, learning_rate=args.learning_rate,
                                                               global_step=global_step, decay_rate=decay_rate,
                                                               decay_steps=decay_steps)
                self.training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels, self.is_training:True})

    def evaluate(self, dataset, images, labels):
        accuracy,predictions, _ = self.session.run([self.accuracy,self.predictions, self.summaries[dataset]], {self.images: images, self.labels: labels,self.is_training:False})
        return accuracy,predictions

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
    #parser.add_argument("--cnn", default='CB-20-3-2-same,M-3-2,CB-30-3-2-same,M-3-2,F,R-1024,R-10', type=str, help="Description of the CNN architecture.")
    #parser.add_argument("--cnn", default='CB-10-3-2-same,M-3-2,F,R-100', type=str, help="Description of the CNN architecture.")
    parser.add_argument("--cnn", default='CB-64-7-1-valid,M-3-2-same,CB-128-3-1-same,F,R-1024', type=str, help="Description of the CNN architecture.")

    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--learning_rate_final", default=0.0005, type=float, help="Final learning rate.")

    #final args: "--epochs 20 --batch_size 80 --learning_rate_final 0.0001 --cnn CB-64-3-1-same,CB-64-3-1-same,M-3-2,CB-256-3-1-same,CB-256-3-1-same,M-3-2,CB-512-3-1-same,CB-512-3-1-same,M-3-2,F,R-1024"
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42,
                                            source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        accuracy = network.evaluate("dev", mnist.validation.images, mnist.validation.labels)[0]
        print("{:.2f}".format(100 * accuracy))

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    test_labels =network.evaluate("test", mnist.test.images, mnist.test.labels)[1]

    with open("{}/mnist_competition_test.txt".format(args.logdir), "w") as test_file:
        for label in test_labels:
            print(label, file=test_file)
