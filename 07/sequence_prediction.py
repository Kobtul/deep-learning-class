#!/usr/bin/env python3
#
#4792aab4-bcb8-11e7-a937-00505601122b
#e47d7ca8-23a9-11e8-9de3-00505601122b
import numpy as np
import tensorflow as tf

class Network:
    DATA = 144
    TEST = 40
    TRAIN = DATA - TEST

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.sequence = tf.placeholder(tf.float32, [self.TRAIN], name="sequence")

            # TODO: Create RNN cell according to args.rnn_cell (RNN, LSTM and GRU should be supported,
            # using BasicRNNCell, BasicLSTMCell and GRUCell from tf.nn.rnn_cell module),
            # with dimensionality of args.rnn_cell_dim. Store the cell in `rnn_cell`.
            if(args.rnn_cell == 'LSTM'):
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_cell_dim)
            elif(args.rnn_cell == 'RNN'):
                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(args.rnn_cell_dim)
            elif(args.rnn_cell == 'GRU'):
                rnn_cell = tf.nn.rnn_cell.GRUCell(args.rnn_cell_dim)
            else:
                pass
            # TODO: Create zero state using rnn_cell.zero_state call. Use batch size 1.
            state = rnn_cell.zero_state(1,dtype=tf.float32)
            predictions, loss = [], 0
            #dense = # TODO: Create a dense layer object using tf.layers.Dense, with 1 output unit.
            #dense = tf.layers.dense(r, 1, activation=None)

            dense = tf.layers.Dense(1, activation=None)
            output, state = rnn_cell(tf.reshape(tf.constant(0,tf.float32),[1,1]),state)
            predictions.append(tf.squeeze(dense(output)))
            loss += tf.losses.mean_squared_error(self.sequence[0], predictions[-1])

            for i in range(1,self.TRAIN):

                # TODO: Call rnn_cell (the input should be 0.0 on first step and self.sequence[i - 1] otherwise).
                # Note that rnn_cell assumes the input is a batch of vectors, so you need to produce the
                # input with [1, 1] shape.
                output, state = rnn_cell(tf.reshape(self.sequence[i-1],[1,1]), state)
                # Then compute current prediction, by using `dense` layer, and append the scalar prediction
                # (i.e., with shape []) to `predictions`.
                predictions.append(tf.squeeze(dense(output)))
                # Also add mean square error of the prediction and self.sequence[i] to the loss.
                loss+=tf.losses.mean_squared_error(self.sequence[i],predictions[-1])
            for i in range(self.TEST):
                # TODO: Call rnn_cell, the input should be the latest prediction. Generate a new
                # prediction using the `dense` layer and append it to `predictions`.
                output, state = rnn_cell(tf.reshape(predictions[-1],[1,1]), state)
                predictions.append(tf.squeeze(dense(output)))


            # TODO: Generate `self.predictions` tensor (instead of Python list), use `tf.stack`.
            self.predictions = tf.stack(predictions,0)
            # Training
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = tf.contrib.summary.scalar("train/loss", loss)
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.prediction_image = tf.placeholder(tf.uint8, [self.DATA, self.DATA, 3])
                self.prediction_gold = tf.placeholder(tf.float32, [self.TEST], name="prediction_gold")
                self.prediction_loss = tf.losses.mean_squared_error(self.prediction_gold, self.predictions[self.TRAIN:])
                self.summaries["prediction"] = [tf.contrib.summary.scalar("prediction/loss", self.prediction_loss),
                                                tf.contrib.summary.image("prediction", tf.expand_dims(self.prediction_image, 0))]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, sequence):
        self.session.run([self.training, self.summaries["train"]], {self.sequence: sequence})

    def predict(self, sequence):
        return self.session.run(self.predictions, {self.sequence: sequence})

    def prediction_summary(self, gold, predictions):
        min_value = min(np.min(gold), np.min(predictions))
        max_value = max(np.max(gold), np.max(predictions))
        def y(x):
            return int(self.DATA - 1 - (self.DATA - 1) * (x - min_value) / (max_value - min_value))

        prediction_image = np.full([self.DATA, self.DATA, 3], 255, dtype=np.uint8)
        for i in range(self.DATA):
            prediction_image[y(gold[i]), i] = [0, 0, 255] if i < self.TRAIN else [0, 255, 0]
            prediction_image[y(predictions[i]), i] = [255, 0, 0]

        _,prediction_loss =self.session.run([self.summaries["prediction"],self.prediction_loss],
                         {self.predictions: predictions, self.prediction_gold: gold[self.TRAIN:], self.prediction_image: prediction_image})
        return prediction_loss


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=10, type=int, help="RNN cell dimension.")
    parser.add_argument("--steps_per_epoch", default=100, type=int, help="Number of training steps per epoch.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
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
    with open("international-airline-passengers.tsv", "r") as data_file:
        data = [float(line.split("\t")[1]) for line in data_file.readlines()[1:]]
        assert(len(data) == Network.DATA)
        data = np.array(data, dtype=np.float32)
        data -= np.min(data)
        data /= np.max(data)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train and predict
    for epoch in range(args.epochs):
        for step in range(args.steps_per_epoch):
            network.train(data[:Network.TRAIN])

        prediction_loss = network.prediction_summary(data, network.predict(data[:Network.TRAIN]))
        # TODO: Print network.prediction_loss for each epoch, using "{:.2g}" format.
        print("{:.2g}".format(prediction_loss))
