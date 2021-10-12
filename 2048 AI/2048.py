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


'''
A class for the "Tile" object. In 2048 tiles have the attributes of value and the coordinates of the tile. The value will always be a power of 2,
and the coordinates will always be 0-3. The coordinates are in the form [x, y] and represented via a list. This class is only used for the selenium bot,
as within our own abstraction we control and edit our own tensor.
'''
class Tile():
    def __init__(
        self,
        cordinates,
        value):
        printd(cordinates)
        for val in range(len(cordinates)):
            cordinates[val] = tf.cast(int(cordinates[val]), dtype = tf.int64)
            #Because indexs within a list start at 0 we subtract 1 from the coords
            #This is because we recieve board coordinates from the website by scraping
            #the class name from the html, where the origin is 1,1 within the board
            cordinates[val] -= 1
        x = cordinates[0]
        y = cordinates[1]
        cordinates = [y, x]
        self.cordinates = cordinates
        self.value = int(value)
    def __str__(self):
        return self.value, self.cordinates


# In[1053]:
'''
The way the hmtl elements are set up, when you match a 2 with a 2, the element will not show up as "4" with the coordinates of the new tile
but instead two "2" elements with the same coordinates. This function is called through two nested for loops which run through y and x and merge 
the duplicate tiles using this function
'''
def addtiles(tile1, tile2):
    #Used in the testing version of the game
    tile3value = tile1.value + tile2.value
    #Add one to the tile coordinates for the class constructor
    tile3cordinates = [tile1.cordinates[1] + 1, tile1.cordinates[0] + 1]
    tile3 = Tile(tile3cordinates, tile3value)
    return tile3


# In[1054]:

#Grabs the site
def loadsite():
    driver.get("https://play2048.co")


# In[1055]:

'''
This function is used to grab the coordinates from a div html element and return the coordinates for said tile on that element.
This function doesn't return a value for the tile as grabbing the value is a simple one liner of "div.text" 
'''
def getcoordinates(tilediv):
    #used in the testing version to scrape tiles cords from html element
    tiledivclass = tilediv.get_attribute("class")
    #String processing the get the position of the coordinates
    cordsindex = tiledivclass.index("tile-position") + 14
    #Cut off all characters before the string "tile-position"
    cordsstr = tiledivclass[cordsindex:]
    #Splitting by spaces isolates the coordinates
    cordsstr = cordsstr.split(" ")
    cords = cordsstr[0]
    cords = cords.split("-")
    printd("these are cords the cords for tilediv:" + tiledivclass)
    printd(cords)
    for cord in cords:
        #coorinates are in string type, force them to int
        cord = int(cord)
    return cords


# In[1056]:

'''
Returns a list of all Tiles within a certain set of coordinates. This is uesful because as previously stated, when tiles are matched
Ex: 2, 2, the result will not be a 4 tile with the new coordinates but instead we get two tiles with the same coordinates, and their previous values.
Thus this function is called in a nested for loop of y and x that checks for any duplicate values and calls this function. If the length of the list
returned by this function is more than 1 then we merge the tiles
'''
def gettiles(tilelist, y, x):
    cordtiles = []
    for tile in tilelist:
        if tile.cordinates == [y, x]:
            cordtiles.append(tile)
    return cordtiles
'''
This function is called within generating the board. It returns a list of Tile objects after iterating through the tiledivs list and scraping their data.
It also merges duplicate tiles into the resulting tiles formed by a movement.
'''
def generatetiles(tiledivs):
    tilelist = []
    #use the class of the div element to get the tiles
    for tilediv in tiledivs:
        tiledivclass = tilediv.get_attribute("class")
        #duplicates that we're going to ignore
        if "tile-merged" in tiledivclass:
            continue
        #get the value by reading the text on the div
        value = int(tilediv.text)
        cordinates = getcoordinates(tilediv)
        #create the tile object and append it to the list
        tile = Tile(cordinates, value)
        tilelist.append(tile)
    tile1index = 0
    #Run through all possible coordinates and get all duplicate values and merge them
    for y in range(4):
        for x in range(4):
            cordtiles = gettiles(tilelist, y, x)
            if len(cordtiles) > 1:
                tilelist.remove(cordtiles[0])
                tilelist.remove(cordtiles[1])
                tilelist.append(addtiles(cordtiles[0], cordtiles[1]))
    printd("this is our tile list")
    for tile in tilelist:
        printd(tile.__str__())
    return tilelist

'''
Function called after every move to get the resulting board. The function uses the webdriver object to grab a the div which the board lies within and
scrape the data from said div. The tiles coordinates and values are then sorted into a sparse tensor and added to a tensor of 0s with a shape of 4 by 4
to create a complete 4 by 4 tensor which is then sent to a normalization function and sent to the net for prediction
'''
def getboard():
    boardshape = [4, 4]
    boardtensor = tf.zeros(boardshape, dtype = np.int64)
    #Try to grab the html element, if for some reason it's not present Ex: page just loaded,
    #wait half a second and try again
    while True:
        try:
            board = driver.find_element_by_class_name("tile-container")
            break
        except NoSuchElementException:
            time.sleep(0.5)
    #The child divs of the main board are all the tiles
    tiledivs = board.find_elements_by_tag_name("div")
    #tile-inner contains useless info for us
    for tilediv in tiledivs:
        if tilediv.get_attribute("class") == "tile-inner":
            tiledivs.remove(tilediv)
    #Generate the tile objects from the divs
    tiles = generatetiles(tiledivs)
    #Empty lists for the sparse tensor we're going to create
    indices = []
    values = []
    #Add values to the lists
    for tile in tiles:
        indices.append(tile.cordinates)
        values.append(tile.value)
    #create the sparse tensor and add it to our empty tensor
    indices = tf.cast(indices, dtype = np.int64)
    values = tf.cast(values, dtype = np.int64)
    delta = tf.SparseTensor(indices, values, boardshape)
    delta = tf.sparse.reorder(delta)
    boardtensor = tf.cast(boardtensor, dtype = np.int64)
    board = boardtensor + tf.sparse.to_dense(delta)
    printd(board)
    return board


'''
In reinforcement learning, the agent can't always learn invalid moves simply through a negative reward or lack of reward. Because our output layer is a softmax
activation, the probability of an invalid move will never completely be 0. Because of this, my approach is to use simple loops to run over the board and deduce 
whether or not a move is valid. The order of the process is to: recieve logits of our final softmax ouput layer, filter out invalid moves by adding a -50000 in
place of the previous logit value, and then restandarize our outputs to equal 1.
'''
def generatevalidlist(board, action_logits):
    #Four boolean flags to describe how we can move, we are not proving that
    #we cannot move a certain direction, instead we're proving that we can
    moveupwards = False
    movedownwards = False
    moveleft = False
    moveright = False
    #Two nested for loops to check all COLUMNS
    for x in range(4):
        #The last value we've seen in the loop
        prevboardval = None
        foundnum = False
        foundzero = False
        for y in range(4):
            #foundnum and foundzero refer to if we have ever seen a number or zero in the row
            #We care about finding a number because if a row has no number, theres nothing to move
            #if it does we can move something within that row and therefore the move is valid
            #We care about finding a zero because 0 represents empty space, without empty 
            #space theres nowhere to move within that row
            if board[y, x] != 0:
                foundnum = True
            else:
                foundzero = True
            #If we've found a number and AFTER found a 0 we can move down
            if ((board[y, x] == 0) and foundnum):
                movedownwards = True
            #If we've found a zero and AFTER found a number we can move up
            if ((board[y, x] != 0 and foundzero)):
                moveupwards = True
            #If we've found a value that matches out previous value we can
            #move either way
            if board[y, x] == prevboardval and (prevboardval != 0):
                moveupwards = True
                movedownwards = True
            prevboardval = board[y, x]
    for y in range(4):
        prevboardval = None
        foundnum = False
        foundzero = False
        #Same logic as before simply looking at ROWS instead of columns
        for x in range(4):
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
    #As discussed in the block comment above we add a value of -50000 to the other values to ensure when it passes
    #through the second softmax that the probability will be 0
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
    #Only if we have some invalid moves do we want to change anything
    if values != []:
        delta = tf.SparseTensor(indices, values, shape)
        action_logits = tf.reshape(action_logits, shape)
        action_logits = action_logits + tf.sparse.to_dense(delta)
    printd("these are our new action logits")
    printd(action_logits)
    #Return edited logits
    return action_logits

'''
This function is extremely similar to the above function. The only difference is that within this function we also check and return whether or not
the game is done. This is because when working with our own abstactions, we need to check ourself if the game is done, as opposed to looking for an html
element which pops up after the game is finished
'''
def generatetrainingvalidlist(board, action_logits):
    moveupwards = False
    movedownwards = False
    moveleft = False
    moveright = False
    #Start by assuming we are not done
    done = False
    for x in range(4):
        prevboardval = None
        foundnum = False
        foundzero = False
        for y in range(4):
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
    #if we cant move anywheree we're done
    if (not moveleft and not moveright) and (not moveupwards and not movedownwards):
        done = True
    if values != []:
        delta = tf.SparseTensor(indices, values, shape)
        action_logits = tf.reshape(action_logits, shape)
        action_logits = action_logits + tf.sparse.to_dense(delta)
    printd("these are our new action logits")
    printd(action_logits)
    return action_logits, done

'''
Function called after every move when playing with the Selenium bot. The function evaluates whether or not the html element which is
present upon the game ending has shown up, and if so, we know to break our loop and add the gradients to our model. It also returns the
score for our previous move which is then sent as a reward to our model.
'''
def evalboard(score):
    printd("EVALUATING BOARD")
    while True:
        try:
            #Try to get the score, if we can't wait half a second
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
        #If we find the "You lost" element we know we're done
        loserbtn = driver.find_element_by_class_name("retry-button")
        loserbtn.click()
        done = True
    except ElementNotInteractableException:
        printd("we can keep going")
    #return the reward, whether or not we're done, and our new score
    return reward, done, newscore


# In[1062]:

'''
Function for selenium bot which executes the move by sending keys to the browser and then evaluates if
we're done
'''
def domove(action, score):
    while True:
        try:
            #Try to find the board, if we can't wait half a second
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

'''
Function called after the agent loses a game, and uses the webdriver to restart the game
'''
def restartgame():
    while True:
        try:
            #Try to get the button, if we can't wait half a second
            loserbtn = driver.find_element_by_class_name("retry-button")
            break
        except NoSuchElementException:
            time.sleep(0.5)
            pass
    loserbtn.click()

    
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
Function called at the start of an episode when training within our own abstractions. Returns a tensor with 2 tiles
to start the game. 
'''
def generateboard():
    boardshape = [4, 4]
    boardtensor = tf.zeros(boardshape, dtype = np.int64)
    indices = []
    values = []
    #counts how many indices we've successfuly added
    indicecounter = 0
    #Python doesn't have "do" which Im used to from C so while True + if -> break
    while True:
        indice = [random.randint(0, 3), random.randint(0, 3)]
        if indice not in indices:
            indices.append(indice)
            indicecounter += 1
            if indicecounter == 2:
                break
    #add 2 random values
    values.append(2*random.randint(1,2))
    values.append(2*random.randint(1,2))
    indices = tf.cast(indices, dtype = np.int64)
    values = tf.cast(values, dtype = np.int64)
    delta = tf.SparseTensor(indices, values, boardshape)
    delta = tf.sparse.reorder(delta)
    boardtensor = tf.cast(boardtensor, dtype = np.int64)
    #Create sparse tensor with random 2 values
    board = boardtensor + tf.sparse.to_dense(delta)
    printd(board)
    return board


'''
Simulates a rightwards move on the board for our games using our own abstractions
'''
def moveright(state):
    #rowdict is what we will sort the new rows into
    rowdict = {}
    reward = 0
    for y in range(4):
        #Get the row we're looking at
        row = state[y]
        #Turn row from tensor to List
        rowlist = [row[0], row[1], row[2], row[3]]
        #Start at the 2nd index and decrement until we reach -1
        for x in range(2, -1, -1):
            if rowlist[x] != 0:
                #if the value that we're at doesn't equal 0 go through every x after it
                #and check if it needs to be moved over
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        #Place a 0 at the index of the value we moved
                        rowlist[x1] = 0
        #If any values are the same add their combined value to the reward and combine them
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


'''
Simulates a leftwards move on the board for our games using our own abstractions
'''
def moveleft(state):
    #rowdict is what we will sort the new rows into
    rowdict = {}
    reward = 0
    for y in range(4):
        #Get the row we're looking at
        row = state[y]
        #Turn row from tensor to List
        rowlist = [row[3], row[2], row[1], row[0]]
        #Start at the 2nd index and decrement until we reach -1
        for x in range(2, -1, -1):
            if rowlist[x] != 0:
                #if the value that we're at doesn't equal 0 go through every x after it
                #and check if it needs to be moved over
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        #Place a 0 at the index of the value we moved
                        rowlist[x1] = 0
        #If any values are the same add their combined value to the reward and combine them
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

'''
Simulates a downwards move on the board for our games using our own abstractions
'''
def movedown(state):
    #rowdict is what we will sort the new rows into
    rowdict = {}
    vertrowdict = {}
    reward = 0
    for y in range(4):
        #Get the row we're looking at
        rowlist = [state[0, y], state[1, y], state[2, y], state[3, y]]
        #Start at the 2nd index and decrement until we reach -1
        for x in range(2, -1, -1):
            if rowlist[x] != 0:
                #if the value that we're at doesn't equal 0 go through every x after it
                #and check if it needs to be moved over
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        rowlist[x1] = 0
        #If any values are the same add their combined value to the reward and combine them
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

'''
Simulates a upwards move on the board for our games using our own abstractions, same thing as moveDown just reversed
'''
def moveup(state):
    rowdict = {}
    vertrowdict = {}
    reward = 0
    for y in range(4):
        #Get the row we're looking at
        rowlist = [state[3, y], state[2, y], state[1, y], state[0, y]]
        #Start at the 2nd index and decrement until we reach -1
        for x in range(2, -1, -1):
            #if the value that we're at doesn't equal 0 go through every x after it
            #and check if it needs to be moved over
            if rowlist[x] != 0:
                for x1 in range(x, 3):
                    if rowlist[x1 + 1] == 0:
                        rowlist[x1 + 1] = rowlist[x1]
                        rowlist[x1] = 0
        #If any values are the same add their combined value to the reward and combine them
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
        #Reversing because its the same thing as move down we just reverse it
        rowlist.reverse()
        rowlist = tf.cast(rowlist, dtype = np.int64)
        rowdict[y] = tf.constant(rowlist)
    for y in range(4):
        vertrowdict[y] = [rowdict[0][y], rowdict[1][y], rowdict[2][y], rowdict[3][y]]
    state = tf.Variable([vertrowdict[0], vertrowdict[1], vertrowdict[2], vertrowdict[3]])
    return state, reward

'''
Generates a new tile for playing within our own abstractions. Called after every move. 
Takes in the state as an argument and returns a new state with the tile added.
'''
def gentile(state):
    #Start with an empty list that will represent the indexes of the empty tiles
    emptytilelist = []
    #List that will represent the coordinates int which we're going to add a tile
    indices = []
    #Random integer that will determine the value of the tile we're gonna add
    decider = random.randint(1, 100)
    #10% chance for a 4, 90% for a 2
    if decider < 10:
        values = [4]
    else:
        values = [2]
    #Iterate through all spaces and get the empty tiles
    for y in range(4):
        for x in range(4):
            if state[y, x] == 0:
                emptytilelist.append([y, x])
    #Pick a random indice out of the chosen tiles
    chosentileindex = random.randint(0, len(emptytilelist) - 1)
    indices.append(emptytilelist[chosentileindex])
    #Create a sparse tensor representing what we're gonna add to the state
    indices = tf.cast(indices, dtype = np.int64)
    values = tf.cast(values, dtype = np.int64)
    delta = tf.SparseTensor(indices, values, [4, 4])
    #Create our new state
    state = state + tf.sparse.to_dense(delta)
    return state

'''
Called everytime the net outputs a value. Takes an argument of the value the net has output and the state.
The state is taken as an argument so we can edit it and send it back. The function returns the editted state
and the reward recieved by the action
'''
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
    #add a new tile to the state
    state = gentile(state)
    return reward, state

'''
Original way I was normalizing the tensor. Unused now that we normalize using the "v2" method.
The function still remains for testing with other net architectures to see if it works better than 
the log base 2 binary method, ex: if the net was changed to a convolutional kernel to dense architecture.
'''
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

'''
Binary normlization method which turns each value within the tensor into a list of 16 values which are all 0s,
and then places a 1 at the index of the value log base 2 - 1. This method has been found to work best at activating
the proper neurons and reducing noise.
'''
def normalizetensorv2(state):
    #change state to a linear tensor
    modelstate = tf.reshape(state, [1, 16])
    tiledict = {}
    #Iterate through all the values in the linear tensor
    for i in range(16):
        #List of 16 0s which we will place a 1 within
        tilelist = np.zeros(16)
        #keep the value as 0 if its 0
        if modelstate[0, i] != 0:
            #set the index that were putting at by taking the log base 2 of the value and subtracting 1
            indexnum = int(math.log(modelstate[0, i], 2) - 1)
            tilelist[indexnum] = 1
        tiledict[i] = tilelist
    #Reconstruct the rows
    rowzero = tf.constant([tiledict[0], tiledict[1], tiledict[2], tiledict[3]])
    rowone = tf.constant([tiledict[4], tiledict[5], tiledict[6], tiledict[7]])
    rowtwo = tf.constant([tiledict[8], tiledict[9], tiledict[10], tiledict[11]])
    rowthree = tf.constant([tiledict[12], tiledict[13], tiledict[14], tiledict[15]])
    #Create environment tensor by adding all the rows list together
    binarystate = tf.Variable([rowzero, rowone, rowtwo, rowthree])
    binarystate = tf.cast(binarystate, dtype = np.float32)
    #Flatten
    binarystate = tf.reshape(binarystate, [1, 256])
    return binarystate

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
