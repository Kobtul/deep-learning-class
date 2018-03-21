#!/usr/bin/env python3
#
#4792aab4-bcb8-11e7-a937-00505601122b
#e47d7ca8-23a9-11e8-9de3-00505601122b
import numpy as np
import tensorflow as tf
import random

class Network:
    OBSERVATIONS = 4
    ACTIONS = 2

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS], name="observations")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")

            # TODO: Define the model, with the output layers for actions in `output_layer`

            #flattened_images = tf.layers.flatten(self.images, name="flatten")

            layers = args.layers
            size_of_the_hidden_layer = args.hidden_layer
            function = args.activation
            if function == "relu":
                aktiv = tf.nn.relu
            elif function == "none":
                aktiv = None
            elif function == "tanh":
                aktiv = tf.nn.tanh
            elif function == "sigmoid":
                aktiv = tf.nn.sigmoid

            input_layer =  self.observations
            hidden_layer = input_layer
            for i in range(layers):
                hidden_layer = tf.layers.dense(hidden_layer, size_of_the_hidden_layer, activation=aktiv,
                                               name="hidden_layer" + str(i))
            output_layer = tf.layers.dense(hidden_layer,size_of_the_hidden_layer, activation=None, name="output_layer")

            self.actions = tf.argmax(output_layer, axis=1, name="actions")

            # Global step
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.actions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.accuracy = accuracy

            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries = [tf.contrib.summary.scalar("train/loss", loss),
                                  tf.contrib.summary.scalar("train/accuracy", accuracy)]

            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/actions", self.actions)
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, observations, labels):
        self.session.run([self.training, self.summaries], {self.observations: observations, self.labels: labels})

    def evaluate(self, observations, labels):
        temp = self.session.run(self.summaries, {self.observations: observations, self.labels: labels})

    def save(self, path):
        self.saver.save(self.session, path)

    def get_accuracy(self, observations, labels):
        return self.session.run(self.accuracy, {self.observations: observations, self.labels: labels})

def makebatch(data,labels, number):
    batch = []
    batch_labels = []

    jackpot = random.sample(range(0, len(data)), number)
    for i in jackpot:
        batch.append(data[i])
        batch_labels.append(labels[i])
    return batch,batch_labels


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1200, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
    parser.add_argument("--layers", default=2, type=int, help="Number of layers.")
    parser.add_argument("--hidden_layer", default=4, type=int, help="Size of the hidden layer.")
    parser.add_argument("--activation", default="relu", type=str, help="Activation function.")


    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    observations, labels = [], []
    with open("gym_cartpole-data.txt", "r") as data:
        for line in data:
            columns = line.rstrip("\n").split()
            observations.append([float(column) for column in columns[0:4]])
            labels.append(int(columns[4]))
    observations, labels = np.array(observations), np.array(labels)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    num_of_batches = len(observations) // args.batch_size

    # Train
    for i in range(args.epochs):
        for j in range(0,num_of_batches):
            batch_observation,batch_labels = makebatch(observations,labels,args.batch_size)
            network.train(batch_observation, batch_labels)
        network.evaluate(observations, labels)
    network.evaluate(observations, labels)

    print("{:.2f}".format(network.get_accuracy(observations, labels) * 100))

    # Save the network
    network.save("gym_cartpole/model")
