import collections 
import numpy as np 
from random import random 

class networkTabularQAgent(object): 
    """ 
    Agent implementing tabular Q-learning for the NetworkSimulatorEnv. 
    """ 
    def __init__(self, num_nodes, num_actions, distance, nlinks): 
        self.config = { 
            "init_mean" : 0.0,       
            "init_std" : 0.1,        
            "learning_rate" : 0.7, 
            "eps": 0.1,             
            "discount": 0.99, #Facteur de réduction plus réaliste 
            "n_iter": 10000000}         
        self.q = np.random.normal(loc=self.config["init_mean"], scale=self.config["init_std"], size=(num_nodes, num_nodes, num_actions)) 
        self.nlinks = nlinks 
     
     
    def act(self, state, best=False): 
        n, dest = state 
        if random() < self.config["eps"]: 
            return np.random.randint(0, self.nlinks[n]) 
        else: 
            return np.argmin(self.q[n][dest][:self.nlinks[n]]) 
     
    def learn(self, current_event, next_event, reward, action, done): 
        n, dest = current_event 
        n_next, dest_next = next_event 
     
        if done: 
            future = 0 
        else: 
            future = np.min(self.q[n_next][dest_next][:self.nlinks[n_next]]) 
     
        # Q-learning update 
        self.q[n][dest][action] += (reward + self.config["discount"] * future - self.q[n][dest][action]) * self.config["learning_rate"] 