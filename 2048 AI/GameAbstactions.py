import math
import os
import openpyxl
import random
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
from tensorflow.keras import layers
from tensorflow.keras import initializers
from typing import Any, List, Sequence, Tuple
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
class HTMLAbstractionFuncs:
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
class InternalAbstractionFuncs:
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
