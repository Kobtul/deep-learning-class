#!/usr/bin/env python3
#
#4792aab4-bcb8-11e7-a937-00505601122b
#e47d7ca8-23a9-11e8-9de3-00505601122b
import numpy as np
import tensorflow as tf

import cart_pole_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, state_shape, num_actions):
        with self.session.graph.as_default():
            # Input states
            self.states = tf.placeholder(tf.float32, [None] + state_shape)
            # Chosen actions (used for training)
            self.actions = tf.placeholder(tf.int32, [None])
            # Observed returns (used for training)
            self.returns = tf.placeholder(tf.float32, [None])

            # Compute the action logits

            # TODO: Add a fully connected layer processing self.states, with args.hidden_layer neurons
            # and some non-linear activatin.
            dense = tf.layers.dense(self.states, args.hidden_layer, activation=tf.nn.relu)


            # TODO: Compute `logits` using another dense layer with
            # `num_actions` outputs (utilizing no activation function).
            output_layer = tf.layers.dense(dense,num_actions, activation=None, name="output_layer")


            # TODO: Compute the `self.probabilities` from the `logits`.
            self.probabilities = tf.nn.softmax(output_layer)
            # Training

            # TODO: Compute `loss`, as a softmax cross entropy loss of self.actions and `logits`.
            # Because this is a REINFORCE algorithm, it is crucial to weight the loss of batch
            # elements using `self.returns` -- this can be accomplished using the `weights` parameter.
            loss = tf.losses.sparse_softmax_cross_entropy(self.actions, output_layer,weights=self.returns, scope="loss")

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        return self.session.run(self.probabilities, {self.states: states})

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns})

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=800, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=0.999, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=200, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    evaluating = False
    all_rewards = []
    while True:
        # TODO: Decide if evaluation should start (one possibility is to train for args.episodes,
        # so env.episode >= args.episodes could be used).
        # evaluation = ...
        # if(env.episode >= args.episodes):
        #     evaluating = True

        # Train for a batch of episodes
        batch_states, batch_actions, batch_returns = [], [], []
        brewards = []
        for _ in range(args.batch_size):
            # Perform episode
            state = env.reset(evaluating)
            states, actions, rewards, done = [], [], [], False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # TODO: Compute action distribution using `network.predict`
                actions_distrib = network.predict([state])


                # TODO: Set `action` randomly according to the generated distribution
                # (you can use np.random.choice or any other method).
                action = np.random.choice(env.actions, p=actions_distrib[0])

                next_state, reward, done, _ = env.step(action)

                # TODO: Accumulate states, actions and rewards.
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO: Compute returns from rewards (by summing them up and
            # applying discount by `args.gamma`).
            num_actions = len(actions)
            returns = np.zeros(len(actions))
            previous_reward = 0
            for i in range(num_actions-1,-1,-1):
                actual_reward = rewards[i]
                previous_reward = previous_reward * args.gamma
                reward = actual_reward + previous_reward
                returns[i] = reward
                previous_reward = reward


            # TODO: Extend the batch_{states,actions,returns} using the episodic
            # {states,actions,returns}.
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)
            all_rewards.append(np.sum(rewards))

        #computes mean of last 100 rewards
        mean_all_previous_rewards = np.mean(all_rewards[len(all_rewards)-100:len(all_rewards)])

        # if last 100 episodes reach more than 490, evaluation is launched.
        if (mean_all_previous_rewards > 490):
            evaluating = True
        if(env.episode >= args.episodes):
            # If the score is not enough after env.episodes, the network is reset and the env.episodes doubled
            if(mean_all_previous_rewards <= 490):
                # print(mean_all_previous_rewards)
                network = Network(threads=args.threads)
                network.construct(args, env.state_shape, env.actions)
                args.episodes+=args.episodes

        # TODO: Perform network training using batch_{states,actions,returns}.
        if not evaluating:
            network.train(batch_states, batch_actions, batch_returns)