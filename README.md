# laikago_imitate

Code to train trotting and pacing policy for Laikago, as described in the paper: [Dynamics Randomization Revisited: A Case Study for Quadrupedal Locomotion](https://zhaomingxie.github.io/projects/Sim2RealLaikago/Sim2RealLaikago.pdf).

Need to install raisim first: https://github.com/ZhaomingXie/raisimLib/tree/juggling.

Good behavior should emerge within 30 minutes of training. 
Thanks to some code optimization, reference to some design choices from [IsaacGym](https://arxiv.org/abs/2108.10470),
and the fast simulation speed of raisim, training is much faster than what is originally reported in the paper.
