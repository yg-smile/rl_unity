# rl_unity
Solve some Unity ML-Agent environments using deep reinforcement learning.


![alt text](https://user-images.githubusercontent.com/49927412/103162134-405bd900-47a1-11eb-8d0c-0f804a90a264.png)

## 3DBall with soft actor critic (SAC)
<p float="left">
  <img src="https://user-images.githubusercontent.com/49927412/103163688-2082e000-47b6-11eb-9abc-a7dc7ffab8c1.gif" width="300" />
  <img src="https://user-images.githubusercontent.com/49927412/103163689-224ca380-47b6-11eb-807f-d2fc60baf3d0.gif" width="300" /> 
  <img src="https://user-images.githubusercontent.com/49927412/103175606-fa922580-481f-11eb-9903-cb92c3028d3b.png" width="300" />
</p>

Run code to reproduce the results: ```run_sac.py```
Hyperparameters:
```
neural net = (400, 400)
discount factor = 0.99
learning rate = 0.001 
1 - target net smoothing coefficient = 0.99
temperature parameter (alpha) = 0.05
batch size = 64
replay memory size = 20000
```

SAC algorithm references: 

<a id="1">[1]</a> 
Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).

<a id="2">[2]</a> 
Achiam, Joshua. “Soft Actor-Critic¶.” Soft Actor-Critic - Spinning Up Documentation, spinningup.openai.com/en/latest/algorithms/sac.html. 

## Setup the Unity environments

1. Download the Unity ML-Agent examples:
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md
2. Build the Unity ML-Agent scene:
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Executable.md
3. Wrap the scene as a gym-like environment using gym_unity:
https://github.com/Unity-Technologies/ml-agents/tree/master/gym-unity

## Multiple agent support

The current ```gym_unity``` unfortunately doesn't support training multiple agents in parallel to improve the sample collection efficiency. To run multiple agents side by side, a workaround is to: 1. comment out all the number of agents checks ```self._check_agents``` in the ```gym_unity/envs/__init__.py```; 2. allow the returns of all default observations and rewards in the ```_single_step()``` function.

We can improve the training time as shown:
<p float="left">
  <img src="https://github.com/yg-smile/rl_unity/files/5859917/3DBall_res.pdf" width="400" />
</p>

