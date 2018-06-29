import sys
import numpy as np
import math
import random

import gym
import gym_maze


class Agent:
    def __init__(self,
                 learning_rate=0.8,
                 environment='maze-sample-10x10-v0',
                 lambda_factor=0.5,
                 learning_decay=0.02,
                 discount_factor=0.99,
                 epsilon=1,
                 epsilon_decay=0.05,
                 max_episode=200,
                 max_step=1000):
        self.learning_rate = learning_rate
        self.environment = environment
        self.lambda_factor = lambda_factor
        self.learning_decay = learning_decay
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.max_episode = max_episode
        self.max_step = max_step

        self.env = gym.make(self.environment)
        self.MAZE_SIZE = tuple(
            (self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int))
        self.NUM_BUCKETS = self.MAZE_SIZE  # one bucket per grid

        # Number of discrete actions
        self.NUM_ACTIONS = self.env.action_space.n  # ["N", "S", "E", "W"]
        # Bounds for each discrete state
        self.STATE_BOUNDS = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        self.q_table = np.zeros(self.NUM_BUCKETS + (self.NUM_ACTIONS,), dtype=float)

    def state_to_bucket(self, state):
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= self.STATE_BOUNDS[i][0]:
                bucket_index = 0
            elif state[i] >= self.STATE_BOUNDS[i][1]:
                bucket_index = self.NUM_BUCKETS[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = self.STATE_BOUNDS[i][1] - self.STATE_BOUNDS[i][0]
                offset = (self.NUM_BUCKETS[i] - 1) * self.STATE_BOUNDS[i][0] / bound_width
                scaling = (self.NUM_BUCKETS[i] - 1) / bound_width
                bucket_index = int(round(scaling * state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)

    def select_action(self, state, explore_rate):
        # Select a random action
        if random.random() < explore_rate:
            action = self.env.action_space.sample()
        # Select the action with the highest q
        else:
            action = int(np.argmax(self.q_table[state]))
        return action

    def update_learning_rate(self, episode):
        self.learning_rate = max(0.2, min(0.8, 1.0 - math.log10((episode + 1) / 10)))

    def update_explore_rate(self, episode):
        self.epsilon = max(0.001, min(0.8, 1.0 - math.log10((episode + 1) / 10)))

    def learn(self):  # learn a policy based on the paramters
        # 要返回点东西:返回total_reward
        total_reward_record = []
        total_step=[]
        for episode in range(self.max_episode):
            obv = self.env.reset()
            state_0 = self.state_to_bucket(obv)
            total_reward = 0
            # self.env.render()
            for step in range(self.max_step):
                # Select an action
                action = self.select_action(state_0, self.epsilon)

                # execute the action
                obv, reward, done, _ = self.env.step(action)

                # Observe the result
                state = self.state_to_bucket(obv)
                total_reward += reward

                # Update the Q based on the result
                best_q = np.amax(self.q_table[state])
                self.q_table[state_0 + (action,)] += self.learning_rate * (
                    reward + self.discount_factor * (best_q) - self.q_table[state_0 + (action,)])

                # Setting up for the next iteration
                state_0 = state
                if done:
                    print("Episode %d finished after %f time steps with total reward = %f ."
                          % (episode, step, total_reward))
                    total_step.append(step)
                    break
                elif step >= self.max_step - 1:
                    print("Episode %d timed out at %d with total reward = %f."
                          % (episode, step, total_reward))
                    total_step.append(step)
            self.update_explore_rate(episode)
            self.update_learning_rate(episode)
            total_reward_record.append(total_reward)
        return total_reward_record,total_step
