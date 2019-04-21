# OpenAI-GYM-Examples
Implementation of the basic examples from OpenAI GYM environment for Reainforcement Learning  

Dependencies (it is recomended to have conda environment installed and sourced with base python ML packages eg. numpy:  
```bash
conda install tensorflow
```  
```bash
pip3 install --upgrade gym
```  
## Structure:
```bash
├── Cart Pole example
│   ├── CartPole  -- core module for basic policy and Policy Gradients
│   ├── show_rewards  -- module for plotting accumulated rewards over games
│   ├── model_trained  -- folder with saved neural network weights
├── MsPacman
│   ├── MsPacman -- Deep Q-Learning MsPacman game with tensorflow
├── Q-Learning
│   ├── q_learning  -- core module for simple q-learning example (board game)
│   ├── show_exploration  -- module for live exploration visualization
```
## Q-Learning
<center>
<img src="https://github.com/amasend/OpenAI-GYM-Examples/blob/master/Q-Learning/pictures/q_learning.PNG?raw=true" width="500" height="300" />
</center>

## Cart Pole
<center>
<img src="https://github.com/amasend/OpenAI-GYM-Examples/blob/master/Cart%20Pole%20example/pictures/CartPole_policy_gradient.gif" width="300" height="300" />  
</center>

## MsPacman
<center>
<img src="https://github.com/amasend/OpenAI-GYM-Examples/blob/master/MsPacman/pictures/pacman.gif?raw=true" width="300" height="300" />
</center>
