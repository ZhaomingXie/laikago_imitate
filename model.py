import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.distributions import Normal, Categorical
import math

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data.view(1, -1), gain=gain)
    return module

init_r_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    nn.init.orthogonal_,
    0.1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ActorCriticNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], num_contact=0):
        super(ActorCriticNet, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer
        self.p_fcs = nn.ModuleList()
        self.v_fcs = nn.ModuleList()
        self.hidden_layer_v = [128, 128]
        if (len(hidden_layer) > 0):
            p_fc = init_r_(nn.Linear(num_inputs, self.hidden_layer[0]))
            v_fc = init_r_(nn.Linear(num_inputs, self.hidden_layer_v[0]))
            self.p_fcs.append(p_fc)
            self.v_fcs.append(v_fc)
            for i in range(len(self.hidden_layer)-1):
                p_fc = init_r_(nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1]))
                v_fc = init_r_(nn.Linear(self.hidden_layer_v[i], self.hidden_layer_v[i+1]))
                self.p_fcs.append(p_fc)
                self.v_fcs.append(v_fc)
            self.mu = init_r_(nn.Linear(self.hidden_layer[-1], num_outputs))
        else:
            #p_fc = init_r_(nn.Linear(num_inputs, num_outputs))
            #self.p_fcs.append(p_fc)
            self.mu = init_r_(nn.Linear(num_inputs, num_outputs))
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        self.v = init_r_(nn.Linear(self.hidden_layer_v[-1],1))
        self.noise = 0
        #self.train()


    def forward(self, inputs):
        # actor
        if len(self.hidden_layer) > 0:
            x = F.relu(self.p_fcs[0](inputs))
            for i in range(len(self.hidden_layer)-1):
                x = F.relu(self.p_fcs[i+1](x))
            mu = torch.tanh(self.mu(x))
        else:
            mu = torch.tanh(self.mu(inputs))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v

    def get_log_stds(self, actions):
        return Variable(torch.Tensor(self.noise)*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(actions).to(device)
        #return self.log_std.unsqueeze(0).expand_as(actions)

    def sample_best_actions(self, inputs):
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = torch.tanh(self.mu(x))
        return mu

    def sample_actions(self, inputs):
        mu = self.sample_best_actions(inputs)
        log_std = self.get_log_stds(mu)
        #std = torch.exp(log_std)
        eps = torch.randn(mu.size(), device=device)
        actions = torch.clamp(mu + log_std.exp()*(eps), -1, 1)
        return actions, mu

    def set_noise(self, noise):
        self.noise = noise

    def get_action(self, inputs):
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = torch.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)
        return mu, log_std

    def get_value(self, inputs, device="cpu"):
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        return v
    def calculate_prob_gpu(self, inputs, actions):
        log_stds = self.get_log_stds(actions).to(device)
        #print(log_stds.shape)
        mean_actions = self.sample_best_actions(inputs)
        #print(mean_actions.shape)
        #w = self.get_w(inputs).to(device)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        #print(log_probs)
        #probs = (log_probs.exp() * w.t()).sum(dim=0).log()
        #print(probs)
        return log_probs, mean_actions

    def calculate_prob(self, inputs, actions, mean_actions):
        log_stds = self.get_log_stds(actions)
        #print(log_stds.shape)
        # mean_actions = self.sample_best_actions(inputs)
        #print(mean_actions.shape)
        #w = self.get_w(inputs).to(device)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        #print(log_probs)
        #probs = (log_probs.exp() * w.t()).sum(dim=0).log()
        #print(probs)
        return log_probs