from abc import ABC, abstractmethod
import numpy as np
from .enviroment import *
from .action_value import *
from .policy import *

class Algorithm(ABC):
    def __init__(self, track):
        self.track = track
        self.Q = None
    @abstractmethod
    def train(self):
        pass


class OffPolicyMCControl(Algorithm):
    
    def __init__(self, track, n_iter=10000, initial_behaviour_eps=0.5, subsequent_behaviour_eps=0.01):
        super().__init__(track)
        self.n_iter = n_iter
        self.initial_behaviour_eps = initial_behaviour_eps
        self.subsequent_behaviour_eps = subsequent_behaviour_eps
        self.Q = TensorActionValue(track)
        self.t_policy = GreedyPolicy(track, self.Q)
        self.b_policy = EpsilonSoftPolicy(track, initial_behaviour_eps, self.Q)

    def train(self, progress_callback=None):

        height, width = self.track.grid.shape
        C  = np.zeros((height, width, 6, 6 , 3, 3)) # cummulative sum of weights
        episodes = []
        #start = time()

        for i in range(self.n_iter):
            if progress_callback:
                progress_callback(i + 1)
            
            episode = self.b_policy.generate_episode()[:-1]
            episodes.append(episode)
            G = 0
            W = 1
            iter_counter = 0
            for t in range(len(episode)-1, -1, -1):
                iter_counter += 1
                G += -1
                episode_indices = episode[t][:-2] + [i+1 for i in episode[t][-2:]] # maping speed changes to matrix indices
                C[tuple(episode_indices)] += W
                self.Q[tuple(episode_indices)] += W / C[tuple(episode_indices)] * (G - self.Q[tuple(episode_indices)]) 
                self.t_policy.update_policy(episode_indices[:-2], self.Q[tuple(episode_indices[:-2])])
                self.b_policy.update_policy_using_greedy(episode_indices[:-2], self.t_policy.policy[tuple(episode_indices[:-2])])
                if episode[t][-2:] != self.t_policy.get_action(episode[t][:-2]):
                    break
                W = W / self.b_policy.get_propability(episode[t][:-2], episode[t][-2:])
            #end = time() 
            if i == 0:
                self.b_policy = EpsilonSoftPolicy(self.track, self.subsequent_behaviour_eps, self.Q)
            #print(f"Iteration: {i}, seconds elapsed: {round(end-start, 2)}")
            #start = time()
        return episodes
            