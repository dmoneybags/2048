so basically:

right now hyperparameters were dealing with are:
0.001 learning rate
"relu" activation
no hidden layers

I've trained 27000 episodes at these parameters and still see increasing results. I would say the next test to try is a hidden layer with a 0.01 learning rate. I already have the code in the Actor Critic class its just commented out. There are 3 Global flags within the code. Debug will print everything. Training will run the training loop. Testing will run a test command which you can stick whatever in. Right now I have it showing the graphs of my csv files (one for actor loss, one for critic loss, one for episode reward). If testing and training are both false the program will use selenium to play 2048 on the actual website.