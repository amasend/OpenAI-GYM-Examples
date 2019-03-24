# OpenAI-GYM-Examples
Implementation of the basic examples from OpenAI GYM environment for Reainforcement Learning  
Dependencies (it is recomended to have conda environment installed and sourced with base python ML packages eg. numpy:  
```bash
conda install tensorflow
```  
```bash
pip3 install --upgrade gym
```  
Usage:  
```bash
python CartPole.py --policy "policy" --epochs "epochs" --steps "steps"
```  
Policies available to use:  
**ang_vel_policy** - If an angle is negative (pole tilted to the left) look at angular velocity.  
&nbsp;&nbsp;&nbsp;If an angle is positive (pole tilted to the right) look at angular velocity.  
**basic_policy** - if an angle is negative (pole tilted to the left) move it to the left.  
&nbsp;&nbsp;&nbsp;If an angle is positive (pole tilted to the right) move it to the right.  
**policy_gradient** - based on neural network  
