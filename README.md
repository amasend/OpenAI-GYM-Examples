# OpenAI-GYM-Examples
Implementation of the basic examples from OpenAI GYM environment for Reainforcement Learning  

Dependencies (it is recomended to have conda environment installed and sourced with base python ML packages eg. numpy:  
```bash
conda install tensorflow
```  
```bash
pip3 install --upgrade gym
```  

## Usage:

For ***basic policy:***
```bash
python CartPole.py --policy basic_policy --epochs epochs_number --steps steps_number
```  
Where:

 - **basic_policy** - if an angle is negative (pole tilted to the left) move it to the left.  If an angle is positive (pole tilted to the right) move it to the right.
 - **epochs_number** - how many times we want to play.
 - **steps_number** - the maximum number of actions we are allowed per game.

For ***angle velocity policy***
```bash
python CartPole.py --policy ang_vel_policy --epochs epochs_number --steps steps_number
```  
Where:

 - **ang_vel_policy** - If an angle is negative (pole tilted to the left) look at angular velocity, if it is positive, it moves to the right, move it to the left.  
If an angle is positive (pole tilted to the right) look at angular velocity, if it is negative, it moves to the left, move it to the right.   
 - **epochs_number** - how many times we want to play.
 - **steps_number** - the maximum number of actions we are allowed per game.

## Neural Network

For ***policy gradients (PG)***
***Training during play.***
```bash
python CartPole.py --policy policy_gradient --learn True --iterations number_of_iterations --max_steps number_of_max_steps --games number_of_games --save_iter saving_iteration_number --gamma reward_discount
```  
Where:

 - **policy_gradient** - based on neural network
 - **learn** - indicator for learning phase
 - **iterations** - number of iterations to train neural network
 - **max_steps** - maximum number of actions per game
 - **games** - number of games after which neural network updates its weights
 - **gamma** - discount factor

***Playing***
```bash
python CartPole.py --policy policy_gradient --learn False --epochs epochs_number --steps --model_path path_to_the_nn_modelsteps_number
```  
Where:

 - **policy_gradient** - based on neural network
 - **learn** - indicator for learning phase
 - **epochs_number** - how many times we want to play.
 - **steps_number** - the maximum number of actions we are allowed per game.
 - **model_path** - path to the stored neural neutwork model
