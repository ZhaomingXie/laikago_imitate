if __name__ == '__main__':
   import json
   from ruamel.yaml import YAML, dump, RoundTripDumper
   from raisimGymTorch.env.bin import laikago_imitate
   from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecTorchEnv as VecEnv
   from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
 
   import torch
   import torch.optim as optim
   import torch.multiprocessing as mp
   import torch.nn as nn
   import torch.nn.functional as F
   from torch.autograd import Variable
   import torch.utils.data
   from model import ActorCriticNet
   import os
   import numpy as np
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
   seed = 3#8
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.set_num_threads(1)
 
   # directories
   task_path = os.path.dirname(os.path.realpath(__file__))
   home_path = task_path + "/../../../../.."

   # config
   cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

   # create environment from the configuration file
   env = VecEnv(laikago_imitate.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
   print("env_created")

   num_inputs = env.observation_space.shape[0]
   num_outputs = env.action_space.shape[0]
   model = ActorCriticNet(num_inputs, num_outputs, [128, 128])
   model.load_state_dict(torch.load("stats/test/iter6999.pt"))
   model.cuda()

   env.setTask()
   env.reset()
   obs = env.observe()
   print(obs[0, :])
   average_gating = np.zeros(8)
   average_gating_sum = 0
   for i in range(10000):
      with torch.no_grad():
         act = model.sample_best_actions(obs)

      obs, rew, done, _ = env.step(act + torch.randn_like(act).mul(0.0))
      #env.reset_time_limit() 
      state = env.get_state()
      print(i, rew[0])

      import time; time.sleep(0.025)