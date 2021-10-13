from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, ElementClickInterceptedException, ElementNotInteractableException, NoSuchWindowException, InvalidSessionIdException
import time
import math
import os
import threading
import openpyxl
import random
import csv
import pandas as pd
#For jupyterlab
from IPython.display import clear_output, display_html
import tensorflow as tf
import collections
import numpy as np
import statistics
import tensorflow as tf
from tensorflow.errors import InvalidArgumentError
from tensorflow.python.framework import ops
import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import initializers
from typing import Any, List, Sequence, Tuple

#essentially if (debug){print(str)}
def printd(str):
    global DEBUG
    if DEBUG:
        print(str)

#print function to send a bar out on the command line for easy separation of items
def bar(barnum):
    #debugging flag
    if barnum == 1:
        printd("*" * 64)
    if barnum ==2:
        printd(":" * 64)

def loadsite():
    driver.get("https://play2048.co")

    
'''
Our net! We use an actor critic architecture to evaluate the best move at our given state, and our expected return from the given state.
This code is mostly taken from the Tensorflow example "CartPole". This is not a deep neural net, and has 0 hidden layers. Hidden layers have
been tested within this architecture, however training becomes unstable after 100 episodes. The net works simply by generating activations
for the state and then passing those activations to the actor layer and the critic layer SEPARATELY:

EX:

inputs
  |
Dense
  /\
 | |
 A  C
 
 We use the actor values immeadiately to determine the best move to pick at the certain moment. We use the critic values to later edit our
 model by comparing the actual expected returns we recieved against the ones we predicted. If we recieved better returns than expected we
 know we played a good game, if we recieved worse rewards than expected we know we played a bad game.
'''
class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(
        self, 
        num_actions: int, 
        num_hidden_units: int,
        num_hidden_inner: int):
        """Initialize."""
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        #Pass our state through the first layer
        x = self.common(inputs)
        #SEPARATELY pass those activations to the actor and critic layer
        return self.actor(x), self.critic(x)

'''
Code modeled off the function of the same name from the Tensorflow "Cartpole" example. The function works within a 
loop making a single move upon every iteration and gathering the logits, critic values, and rewards upon every move.
We then return these values to generate gradients and edit the model
'''
def run_episode( 
    model: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""
    #initializing tensorArrays for values we need
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    score = 0
    index = 0
    while True:
        # Convert state into a batched tensor (batch size = 1)
        state = getboard()
        modelstate = normalizetensorv2(state)
        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(modelstate)

        # Sample next action from the action probability distribution
        used_logits = generatevalidlist(state, action_logits_t)
        action = tf.random.categorical(used_logits, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        printd("action probs")
        printd(action_probs_t)
        # Store critic values
        values = values.write(index, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(index, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        reward, done, score = domove(action, score)

        # Store reward
        rewards = rewards.write(index, reward)
        index += 1
        #write done identifier
        if done:
            break
    #turn tensorArrays into pure tensors
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards

'''
Function used to run an episode using own abstractions. Quickest method to train the model. Returns the
action probabilities sampled, critic values measured, and the rewards recieved. These values then undergo
computations to determine the gradients and edit the model
'''
def runtrainepisode(
    model: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    
    index = 0
    
    state = generateboard()
    while True:
        # Convert state into a batched tensor (batch size = 1)
        
        modelstate = normalizetensorv2(state)
        
        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(modelstate)

        # Sample next action from the action probability distribution
        used_logits, done = generatetrainingvalidlist(state, action_logits_t)
        if done:
            clear_output()
            print(state)
            break
        action = tf.random.categorical(used_logits, 1)[0, 0]
        
        action_probs_t = tf.nn.softmax(used_logits)
        printd("action probs")
        printd(action_probs_t)
        # Store critic values
        values = values.write(index, tf.squeeze(value))

        # Store log probability of the action chosen
        
        action_probs = action_probs.write(index, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        reward, state = dotrainmove(action, state)

        # Store reward
        rewards = rewards.write(index, reward)
        index += 1
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards

#Defining our loss function
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

'''
Iterates through all the rewards and calculates the returns recieved after said point, 
with each reward multiplied by an exponentially decaying factor, to give higher priority
to sooner rewards. Returns the list of "true" critic values to be compared against our
generated critic values and to, from there, generate gradients
'''
def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
    eps = np.finfo(np.float32).eps.item()
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]
    #Standardize to increase stability
    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / 
               (tf.math.reduce_std(returns) + eps))

    return returns

'''
Computes loss by first computing an "advantage" value, calculated by finding the difference between
our generated expected return and the expected returns we actually recieved. It then multiplys this advantage
by the probabilities of the actions selected. This sets up 4 distinct settings:
1. High probabilities of good actions: Positive loss
2. Low probabilities of good actions: Small positive loss
3. High probabilities of bad actions: High negative loss
4: Low probabilities of bad actions: Small negative loss
Critic loss is then determined using a huber loss function on the actual expecteed returns and the predicted
expected returns
'''
def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""
    eps = np.finfo(np.float32).eps.item()
    values = ((values - tf.math.reduce_mean(values)) / 
               (tf.math.reduce_std(values) + eps))
    advantage = returns - values
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)
    #Write actor and critic loss to csv files for graphing
    csvwrite(float(actor_loss), "loss.csv")
    csvwrite(float(critic_loss), "criticloss.csv")
    return actor_loss + critic_loss

'''
Runs a training episode and calculates gradients based off the data returned from that episode.
'''
def train_step(
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float,
    n: int) -> tf.Tensor:
    """Runs a model training step."""
    global TRAINING
    with tf.GradientTape() as tape:

        # Run the model for one episode to collect training data
        if TRAINING:
            action_probs, values, rewards = runtrainepisode(
            model) 
        else:
            action_probs, values, rewards = run_episode(
            model) 

        # Calculate expected returns
        returns = get_expected_return(rewards, gamma)

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

        # Calculating loss values to update our network
        loss = compute_loss(action_probs, values, returns)

    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    episode_reward = tf.math.reduce_sum(rewards)
    
    
    
    print("------------------------finished game--------------------------")
    print("loss: " + str(loss))
    print("this was our episode_reward: " + str(episode_reward))
    printd("These are the gradients we're applying:")
    printd(grads)
    #Write episode reward to csv file
    csvwrite(float(episode_reward), "episodereward.csv")
    return episode_reward

'''
Restarts the jupyter kernel. Currently unused
'''
def restartkernel() :
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)

'''
Line saver to write data to a csv file, takes an argument of the value being written and the
string file name
'''
def csvwrite(stat, strfilename):
    file = open(strfilename, 'a')
    file.write(str(stat) + "\n")
    file.close()

'''
Function which generates a matplotlib graph of csv data. Takes arguments of the string file
name, color of the graph, label of the graph, and batchsize for each point
'''
def graphcsv(strfilename, color, label, batchsize):
    x = []
    y = []
    with open(strfilename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        counter = 0
        for row in lines:
            if counter == 0:
                val = float(row[0])
            elif counter % batchsize == 0: 
                x.append(counter)
                y.append(val/batchsize)
                val = 0
            else:
                val += float(row[0])
            counter += 1
    plt.plot(x, y, color = color, linestyle = 'dashed', marker = 'o', 
             label = label)
    plt.xticks(rotation = 25)
    plt.xlabel("game")
    plt.ylabel(label)
    plt.title(label, fontsize = 20)
    plt.grid()
    plt.legend()
    plt.show()

'''
Shows all our graphs
'''
def showstats():
    graphcsv('episodereward.csv', 'g', "Episode Reward", 1000)
    graphcsv('loss.csv', 'r', "Actor Loss", 1000)
    graphcsv('criticloss.csv', 'b', "Critic Loss", 1000)

'''
Play 2048 using out own abstractions! called when the global TESTING is set to True
'''
def runtest():
    showstats()
    state = generateboard()
    while True:
        action = int(input())
        reward, state = dotrainmove(action, state)
        clear_output()
        print(state)
        print(reward)

#Verbose
DEBUG = False
#Running in our own abstractions or browser?
TRAINING = True
#Test our abstractions by playing
TESTING = False
######################################
if TESTING:
    runtest()
if not TRAINING:
    #Set up driver if we're playing in browser
    PATH = "/Users/daniel/chromebin/chromedriver"
    driver = webdriver.Chrome(PATH)
#Set random seed
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
#Generate our model
model = ActorCritic(4, 256, 12)
#Generate our optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#Exponentially decaying factor used for the expected returns function
gamma = 0.99
n = 0
#Checkpoint and Checkpoint manager for saving model
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
#If we're playing in brower load the site
if not TRAINING:
    loadsite()
while True:
    #Load the latest checkpoint
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint is None:
        #If the checkpoint is none thats likely because were using a new setup
        #If so delete all the old data so we can start with a fresh graph
        loss = open("loss.csv", "w")
        loss.truncate(0)
        criticloss = open("criticloss.csv", "w")
        criticloss.truncate(0)
        episodereward = open("episodereward.csv", "w")
        episodereward.truncate(0)
    #run a trainstep
    train_step(model, opt, gamma, n)
    n += 1
    if not TRAINING:
        showstats()
    model.summary()
    save_path = manager.save()
    if not TRAINING:
        bar(1)
        print("model saved")
        bar(1)
        restartkernel()
