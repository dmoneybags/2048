#!/usr/bin/env python
# coding: utf-8

# In[1049]:


"off white in python"
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


# In[1050]:


def printd(str):
    global DEBUG
    if DEBUG:
        print(str)


# In[1051]:


def bar(barnum):
    #debugging flag
    if barnum == 1:
        printd("*******************************************************************")
    if barnum ==2:
        printd(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")


# In[1052]:


class Tile():
    def __init__(
        self,
        cordinates,
        value):
        printd(cordinates)
        for val in range(len(cordinates)):
            cordinates[val] = tf.cast(int(cordinates[val]), dtype = tf.int64)
            cordinates[val] -= 1
        x = cordinates[0]
        y = cordinates[1]
        cordinates = [y, x]
        self.cordinates = cordinates
        self.value = int(value)
    def __str__(self):
        return self.value, self.cordinates


# In[1053]:


def addtiles(tile1, tile2):
    #used in the testing version of the game
    tile3value = tile1.value + tile2.value
    tile3cordinates = [tile1.cordinates[1] + 1, tile1.cordinates[0] + 1]
    tile3 = Tile(tile3cordinates, tile3value)
    return tile3


# In[1054]:


def loadsite():
    driver.get("https://play2048.co")


# In[1055]:


def getcordinates(tilediv):
    #used in the testing version to scrape tiles cords from html element
    tiledivclass = tilediv.get_attribute("class")
    printd(tiledivclass)
    cordsindex = tiledivclass.index("tile-position") + 14
    cordsstr = tiledivclass[cordsindex:]
    cordsstr = cordsstr.split(" ")
    cords = cordsstr[0]
    cords = cords.split("-")
    printd("these are cords the cords for tilediv:" + tiledivclass)
    printd(cords)
    for cord in cords:
        cord = int(cord)
    return cords


# In[1056]:


def gettiles(tilelist, y, x):
    cordtiles = []
    for tile in tilelist:
        if tile.cordinates == [y, x]:
            cordtiles.append(tile)
    return cordtiles


# In[1057]:


def generatetiles(tiledivs):
    tilelist = []
    for tilediv in tiledivs:
        tiledivclass = tilediv.get_attribute("class")
        if "tile-merged" in tiledivclass:
            continue
        value = int(tilediv.text)
        printd("this is the value: " + str(value))
        cordinates = getcordinates(tilediv)
        tile = Tile(cordinates, value)
        tilelist.append(tile)
    tile1index = 0
    printd("this is the num tiles")
    printd(len(tilelist))
    matchedindexs = []
    tilelistcopy = tilelist
    for y in range(4):
        for x in range(4):
            cordtiles = gettiles(tilelist, y, x)
            if len(cordtiles) > 1:
                assert(len(cordtiles) < 3)
                tilelist.remove(cordtiles[0])
                tilelist.remove(cordtiles[1])
                tilelist.append(addtiles(cordtiles[0], cordtiles[1]))
    printd("this is our tile list")
    for tile in tilelist:
        printd(tile.__str__())
    printd('these are our matched indices')
    printd(matchedindexs)
    return tilelist


# In[1058]:


def getboard():
    boardshape = [4, 4]
    boardtensor = tf.zeros(boardshape, dtype = np.int64)
    while True:
        try:
            board = driver.find_element_by_class_name("tile-container")
            break
        except NoSuchElementException:
            time.sleep(0.5)
    tiledivs = board.find_elements_by_tag_name("div")
    for tilediv in tiledivs:
        if tilediv.get_attribute("class") == "tile-inner":
            tiledivs.remove(tilediv)
    tiles = generatetiles(tiledivs)
    indices = []
    values = []
    for tile in tiles:
        indices.append(tile.cordinates)
        values.append(tile.value)
    printd("ATTEMPTING TO CREATE SPARSE TENSOR WITH THESE INDICES")
    printd(indices)
    indices = tf.cast(indices, dtype = np.int64)
    values = tf.cast(values, dtype = np.int64)
    printd("AND THESE VALUES")
    printd(values)
    delta = tf.SparseTensor(indices, values, boardshape)
    delta = tf.sparse.reorder(delta)
    boardtensor = tf.cast(boardtensor, dtype = np.int64)
    board = boardtensor + tf.sparse.to_dense(delta)
    printd(board)
    return board


# In[1059]:


def generatevalidlist(board, action_logits):
    moveupwards = False
    movedownwards = False
    moveleft = False
    moveright = False
    for x in range(4):
        prevboardval = None
        foundnum = False
        foundzero = False
        for y in range(4):
            #foundnum and foundzero refer to if we have ever seen a number or zero in the row
            if board[y, x] != 0:
                foundnum = True
            else:
                foundzero = True
            if ((board[y, x] == 0) and foundnum):
                movedownwards = True
            if ((board[y, x] != 0 and foundzero)):
                moveupwards = True
            if board[y, x] == prevboardval and (prevboardval != 0):
                moveupwards = True
                movedownwards = True
            prevboardval = board[y, x]
    for y in range(4):
        prevboardval = None
        foundnum = False
        foundzero = False
        for x in range(4):
            #foundnum and foundzero refer to if we have ever seen a number or zero in the row
            if board[y, x] != 0:
                foundnum = True
            else:
                foundzero = True
            if ((board[y, x] == 0) and foundnum):
                moveright = True
            if ((board[y, x] != 0 and foundzero)):
                moveleft = True
            if board[y, x] == prevboardval and (prevboardval != 0):
                moveright = True
                moveleft = True
            prevboardval = board[y, x]
    #list for values will be in order of 0: up, 1: down, 2: right, 3: left
    values = []
    indices = []
    shape = [1, 4]
    if not moveupwards:
        printd("SCRIPT THINKS WE CANT MOVE UP")
        values.append(float(-50000))
        indices.append([0, 0])
    if not movedownwards:
        printd("SCRIPT THINKS WE CANT MOVE DOWN")
        values.append(float(-50000))
        indices.append([0, 1])
    if not moveright:
        printd("SCRIPT THINKS WE CANT MOVE RIGHT")
        values.append(float(-50000))
        indices.append([0, 2])
    if not moveleft:
        printd("SCRIPT THINKS WE CANT MOVE LEFT")
        values.append(float(-50000))
        indices.append([0, 3])
    if values != []:
        delta = tf.SparseTensor(indices, values, shape)
        action_logits = tf.reshape(action_logits, shape)
        action_logits = action_logits + tf.sparse.to_dense(delta)
    printd("these are our new action logits")
    printd(action_logits)
    return action_logits


# In[1060]:


def generatetrainingvalidlist(board, action_logits):
    moveupwards = False
    movedownwards = False
    moveleft = False
    moveright = False
    done = False
    for x in range(4):
        prevboardval = None
        foundnum = False
        foundzero = False
        for y in range(4):
            #foundnum and foundzero refer to if we have ever seen a number or zero in the row
            if board[y, x] != 0:
                foundnum = True
            else:
                foundzero = True
            if ((board[y, x] == 0) and foundnum):
                movedownwards = True
            if ((board[y, x] != 0 and foundzero)):
                moveupwards = True
            if board[y, x] == prevboardval and (prevboardval != 0):
                moveupwards = True
                movedownwards = True
            prevboardval = board[y, x]
    for y in range(4):
        prevboardval = None
        foundnum = False
        foundzero = False
        for x in range(4):
            #foundnum and foundzero refer to if we have ever seen a number or zero in the row
            if board[y, x] != 0:
                foundnum = True
            else:
                foundzero = True
            if ((board[y, x] == 0) and foundnum):
                moveright = True
            if ((board[y, x] != 0 and foundzero)):
                moveleft = True
            if board[y, x] == prevboardval and (prevboardval != 0):
                moveright = True
                moveleft = True
            prevboardval = board[y, x]
    #list for values will be in order of 0: up, 1: down, 2: right, 3: left
    values = []
    indices = []
    shape = [1, 4]
    if not moveupwards:
        printd("SCRIPT THINKS WE CANT MOVE UP")
        values.append(float(-50000))
        indices.append([0, 0])
    if not movedownwards:
        printd("SCRIPT THINKS WE CANT MOVE DOWN")
        values.append(float(-50000))
        indices.append([0, 1])
    if not moveright:
        printd("SCRIPT THINKS WE CANT MOVE RIGHT")
        values.append(float(-50000))
        indices.append([0, 2])
    if not moveleft:
        printd("SCRIPT THINKS WE CANT MOVE LEFT")
        values.append(float(-50000))
        indices.append([0, 3])
    if (not moveleft and not moveright) and (not moveupwards and not movedownwards):
        done = True
    if values != []:
        delta = tf.SparseTensor(indices, values, shape)
        action_logits = tf.reshape(action_logits, shape)
        action_logits = action_logits + tf.sparse.to_dense(delta)
    printd("these are our new action logits")
    printd(action_logits)
    return action_logits, done


# In[1061]:


def evalboard(score):
    printd("EVALUATING BOARD")
    while True:
        try:
            newscore = int(driver.find_element_by_xpath("/html/body/div[1]/div[1]/div/div[1]").text)
            break
        except ValueError:
            continue
    reward = newscore - score
    printd("THIS IS REWARD")
    printd(reward)
    done = False
    time.sleep(0.5)
    try:
        loserbtn = driver.find_element_by_class_name("retry-button")
        loserbtn.click()
        done = True
    except ElementNotInteractableException:
        printd("we can keep going")
    return reward, done, newscore


# In[1062]:


def domove(action, score):
    while True:
        try:
            board = driver.find_element_by_tag_name("body")
            break
        except NoSuchElementException:
            time.sleep(0.5)
    if action == 0:
        key = Keys.UP
    if action == 1:
        key = Keys.DOWN
    if action == 2:
        key = Keys.RIGHT
    if action == 3:
        key = Keys.LEFT
    board.send_keys(key)
    reward, done, score = evalboard(score)
    return reward, done, score


# In[1063]:


def restartgame():
    while True:
        try:
            loserbtn = driver.find_element_by_class_name("retry-button")
        except NoSuchElementException:
            time.sleep(0.5)
            pass
    loserbtn.click()


# In[1064]:


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
        #self.hidden = layers.Dense(num_hidden_inner)
        #self.hidden1 = layers.Dense(num_hidden_inner)
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        #x = self.hidden(x)
        #x = self.hidden1(x)
        return self.actor(x), self.critic(x)


# In[1065]:


def run_episode( 
    model: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

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
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


# In[1066]:


def generateboard():
    boardshape = [4, 4]
    boardtensor = tf.zeros(boardshape, dtype = np.int64)
    indices = []
    values = []
    indicecounter = 0
    while True:
        indice = [random.randint(0, 3), random.randint(0, 3)]
        if indice not in indices:
            indices.append(indice)
            indicecounter += 1
            if indicecounter == 2:
                break
    values.append(2*random.randint(1,2))
    values.append(2*random.randint(1,2))
    indices = tf.cast(indices, dtype = np.int64)
    values = tf.cast(values, dtype = np.int64)
    delta = tf.SparseTensor(indices, values, boardshape)
    delta = tf.sparse.reorder(delta)
    boardtensor = tf.cast(boardtensor, dtype = np.int64)
    board = boardtensor + tf.sparse.to_dense(delta)
    printd(board)
    return board


# In[1067]:


def moveright(state):
    rowdict = {}
    reward = 0
    for y in range(4):
        #get whatever row were looking at
        row = state[y]
        rowlist = [row[0], row[1], row[2], row[3]]
        for x in range(2, -1, -1):
            if rowlist[x] != 0:
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        rowlist[x1] = 0
                printd(y)
                printd(x)
        for x in range(2, -1, -1):
            if rowlist[x + 1] == rowlist[x]:
                rowlist[x + 1] = rowlist[x] * 2
                reward += rowlist[x] * 2
                rowlist[x] = 0
        #collapse row again
        for x in range(2, -1, -1):
            if rowlist[x] != 0:
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        rowlist[x1] = 0
        #add new rows into dictionary
        rowlist = tf.cast(rowlist, dtype = np.int64)
        rowdict[y] = tf.constant(rowlist)
    state = tf.Variable([rowdict[0], rowdict[1], rowdict[2], rowdict[3]])
    return state, reward


# In[1068]:


def moveleft(state):
    rowdict = {}
    reward = 0
    for y in range(4):
        #get whatever row were looking at
        row = state[y]
        rowlist = [row[3], row[2], row[1], row[0]]
        for x in range(2, -1, -1):
            if rowlist[x] != 0:
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        rowlist[x1] = 0
                printd(y)
                printd(x)
        for x in range(2, -1, -1):
            if rowlist[x + 1] == rowlist[x]:
                rowlist[x + 1] = rowlist[x] * 2
                reward += rowlist[x] * 2
                rowlist[x] = 0
        #collapse row again
        for x in range(2, -1, -1):
            if rowlist[x] != 0:
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        rowlist[x1] = 0
        #add new rows into dictionary
        rowlist.reverse()
        rowlist = tf.cast(rowlist, dtype = np.int64)
        rowdict[y] = tf.constant(rowlist)
    state = tf.Variable([rowdict[0], rowdict[1], rowdict[2], rowdict[3]])
    return state, reward


# In[1069]:


def movedown(state):
    rowdict = {}
    vertrowdict = {}
    reward = 0
    for y in range(4):
        #get whatever row were looking at
        rowlist = [state[0, y], state[1, y], state[2, y], state[3, y]]
        for x in range(2, -1, -1):
            if rowlist[x] != 0:
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        rowlist[x1] = 0
                printd(y)
                printd(x)
        for x in range(2, -1, -1):
            if rowlist[x + 1] == rowlist[x]:
                rowlist[x + 1] = rowlist[x] * 2
                reward += rowlist[x] * 2
                rowlist[x] = 0
        #collapse row again
        for x in range(2, -1, -1):
            if rowlist[x] != 0:
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        rowlist[x1] = 0
        #add new rows into dictionary
        rowlist = tf.cast(rowlist, dtype = np.int64)
        rowdict[y] = tf.constant(rowlist)
    for y in range(4):
        vertrowdict[y] = [rowdict[0][y], rowdict[1][y], rowdict[2][y], rowdict[3][y]]
    state = tf.Variable([vertrowdict[0], vertrowdict[1], vertrowdict[2], vertrowdict[3]])
    return state, reward


# In[1070]:


def moveup(state):
    rowdict = {}
    vertrowdict = {}
    reward = 0
    for y in range(4):
        #get whatever row were looking at
        rowlist = [state[3, y], state[2, y], state[1, y], state[0, y]]
        for x in range(2, -1, -1):
            if rowlist[x] != 0:
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        rowlist[x1] = 0
                printd(y)
                printd(x)
        for x in range(2, -1, -1):
            if rowlist[x + 1] == rowlist[x]:
                rowlist[x + 1] = rowlist[x] * 2
                reward += rowlist[x] * 2
                rowlist[x] = 0
        #collapse row again
        for x in range(2, -1, -1):
            if rowlist[x] != 0:
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        rowlist[x1] = 0
        #add new rows into dictionary
        rowlist.reverse()
        rowlist = tf.cast(rowlist, dtype = np.int64)
        rowdict[y] = tf.constant(rowlist)
    for y in range(4):
        vertrowdict[y] = [rowdict[0][y], rowdict[1][y], rowdict[2][y], rowdict[3][y]]
    state = tf.Variable([vertrowdict[0], vertrowdict[1], vertrowdict[2], vertrowdict[3]])
    return state, reward


# In[1071]:


def gentile(state):
    emptytilelist = []
    indices = []
    decider = random.randint(1, 100)
    if decider < 10:
        values = [4]
    else:
        values = [2]
    for y in range(4):
        for x in range(4):
            if state[y, x] == 0:
                emptytilelist.append([y, x])
    chosentileindex = random.randint(0, len(emptytilelist) - 1)
    indices.append(emptytilelist[chosentileindex])
    indices = tf.cast(indices, dtype = np.int64)
    values = tf.cast(values, dtype = np.int64)
    delta = tf.SparseTensor(indices, values, [4, 4])
    state = state + tf.sparse.to_dense(delta)
    return state


# In[1072]:


def dotrainmove(action, state):
    if action == 0:
        state, reward = moveup(state)
    elif action == 1:
        state, reward = movedown(state)
    elif action == 2:
        state, reward = moveright(state)
    elif action == 3:
        state, reward = moveleft(state)
    reward = tf.cast(reward, dtype = np.float32)
    state = gentile(state)
    return reward, state


# In[1073]:


def normalizetensor(state):
    
    modelstate = tf.reshape(state, [1, 16])

    scaledlist = []
    for i in range(16):
        if modelstate[0, i] != 0:
            scaledlist.append(float(math.log(modelstate[0, i], 2)))
        else:
            scaledlist.append(float(0))
    scaledtensor = tf.Variable(scaledlist)
    scaledtensor = tf.cast(scaledtensor, dtype = np.float32)
    scaledtensor = tf.reshape(scaledtensor, [1, 16])
    return scaledtensor


# In[1074]:


def normalizetensorv2(state):
    modelstate = tf.reshape(state, [1, 16])
    tiledict = {}
    for i in range(16):
        tilelist = np.zeros(16)
        if modelstate[0, i] != 0:
            indexnum = int(math.log(modelstate[0, i], 2) - 1)
            assert((math.log(modelstate[0, i], 2) - 1) % 1 == 0)
            tilelist[indexnum] = 1
        tiledict[i] = tilelist
    rowzero = tf.constant([tiledict[0], tiledict[1], tiledict[2], tiledict[3]])
    rowone = tf.constant([tiledict[4], tiledict[5], tiledict[6], tiledict[7]])
    rowtwo = tf.constant([tiledict[8], tiledict[9], tiledict[10], tiledict[11]])
    rowthree = tf.constant([tiledict[12], tiledict[13], tiledict[14], tiledict[15]])
    binarystate = tf.Variable([rowzero, rowone, rowtwo, rowthree])
    binarystate = tf.cast(binarystate, dtype = np.float32)
    binarystate = tf.reshape(binarystate, [1, 256])
    return binarystate


# In[1075]:


def checkstatefor2048(state):
    for y in range(4):
        for x in range(4):
            if state[y, x] >= 2048:
                assert(False)


# In[1076]:


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


# In[1077]:


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
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

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / 
               (tf.math.reduce_std(returns) + eps))

    return returns


# In[1078]:


def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""
    eps = np.finfo(np.float32).eps.item()
    values = ((values - tf.math.reduce_mean(values)) / 
               (tf.math.reduce_std(values) + eps))
    advantage = returns - values
    print(values)
    print(returns)
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)
    csvwrite(float(actor_loss), "loss.csv")
    csvwrite(float(critic_loss), "criticloss.csv")
    return actor_loss + critic_loss


# In[1079]:


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
    csvwrite(float(episode_reward), "episodereward.csv")
    return episode_reward


# In[1080]:


def restartkernel() :
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)


# In[1081]:


def csvwrite(stat, strfilename):
    file = open(strfilename, 'a')
    file.write(str(stat) + "\n")
    file.close()


# In[1082]:


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


# In[1083]:


def showstats():
    graphcsv('episodereward.csv', 'g', "Episode Reward", 1000)
    graphcsv('loss.csv', 'r', "Actor Loss", 1000)
    graphcsv('criticloss.csv', 'b', "Critic Loss", 1000)


# In[1084]:


def runtest():
    showstats()
    state = generateboard()
    while True:
        action = int(input())
        reward, state = dotrainmove(action, state)
        clear_output()
        print(state)
        print(reward)


# In[ ]:


DEBUG = False
TRAINING = False
TESTING = False
######################################
if TESTING:
    runtest()
if not TRAINING:
    PATH = "/Users/weston/bin/chromedriver"
    driver = webdriver.Chrome(PATH)
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
model = ActorCritic(4, 256, 12)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
gamma = 0.99
n = 0
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
if not TRAINING:
    loadsite()
while True:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint is None:
        loss = open("loss.csv", "w")
        loss.truncate(0)
        criticloss = open("criticloss.csv", "w")
        criticloss.truncate(0)
        episodereward = open("episodereward.csv", "w")
        episodereward.truncate(0)
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


# In[ ]:




