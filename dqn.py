import gym
import numpy as np
from math import floor, ceil
import time
import logging
import warnings
from PIL import Image
import imageio
import matplotlib.pyplot as plt

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, Input, Lambda, multiply
from keras.layers.convolutional import Conv2D, MaxPooling2D

import itertools
import random

ACTIONS = [2, 0, 5]
MAX_GRAYSCALE = 255.0
BATCH_SIZE = 32
UP = 2
NOOP = 0
DOWN = 5

class RingBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, datum):
        if len(self.memory) < self.capacity:
            self.memory.append(datum)
        else:
            self.memory[self.position] = datum

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

def process_data(data, init=False):
    cropped = data[34:-16,:]
    if init:
        return (np.mean(cropped[::2,::2,:], axis=2)/255.0).reshape((1, 80, 80, 1))
    else:
        return (np.mean(cropped[::2,::2,:], axis=2)/255.0).reshape((80, 80, 1))

def one_hot_encode(action):
    if action == UP: return [1, 0, 0]
    elif action == NOOP: return [0, 1, 0]
    else: return [0, 0, 1]

def get_epsilon():
    i = 0
    while (True):
        yield (1 - 0.9 * (i / 50000)) if i < 50000 else 0.1

        i += 1

def q_learn(env, model, gamma=0.9):
    get_ep = get_epsilon()
    games_won = 0
    threshold = 0.0

    for t in itertools.count():
        done = False
        memory = RingBuffer()

        s_t = process_data(env.reset(), init=True)

        S_t = s_t.repeat(4, axis=3)

        round_loss = 0

        while not done:
            if np.random.random() <= next(get_ep):
                action_idx = np.random.randint(0, 3)
            else:
                action_idx = np.argmax(model.predict(S_t))

            s_t_prime, reward, done, info = env.step(ACTIONS[action_idx])
            s_t_prime = process_data(s_t_prime, init=True)

            S_t_prime = np.append(s_t_prime, S_t[:, :, :, :3], axis=3)

            memory.append((S_t, action_idx, reward, S_t_prime, done))

            # env.render()

            if t > 400:
                batch = memory.sample(BATCH_SIZE)

                states_t, actions_t, rewards_t, states_t_prime, done_t = [], [], [], [], []
                for state, action, reward, state_prime, done in batch:
                    states_t.append(state[0])
                    actions_t.append(action)
                    rewards_t.append(reward)
                    states_t_prime.append(state_prime[0])
                    done_t.append(done)

                states_t = np.array(states_t)
                states_t_prime = np.array(states_t_prime)

                Q_sa_prime = model.predict(states_t)
                Q_sa = model.predict(states_t_prime)

                Q_sa_prime[:, actions_t] = rewards_t + gamma * np.max(Q_sa, axis=1) * np.invert(done_t)

                prev_loss = round_loss
                round_loss += model.train_on_batch(states_t, Q_sa_prime)

                print("Loss for batch: {:05f}".format(prev_loss-round_loss), end='\r')

                if (t % 1000) == 0:
                    model.save_weights("weights.h5", overwrite=True)

                if (games_won-threshold >= 0.1):
                    model.save_weights("weights_{:02f}_won.hf".format(games_won))
                    threshold = games_won

                print()

            S_t = S_t_prime

        if reward == 1:
            games_won += 1

        print("Total games: {}\tPct games won: {:05f}\tGame {} loss: {:05f}".format(t+1, games_won/(t+1), t+1, round_loss))

if __name__ == '__main__':
    model = Sequential()
    model.add(Conv2D(16, (8, 8), input_shape=(80, 80, 4), activation='relu', padding='same'))
    model.add(Conv2D(32, (4, 4), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3))
    model.compile(optimizer='rmsprop', loss='mse')

    env = gym.make('Pong-v0')
    q_learn(env, model)
