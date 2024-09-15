from abc import ABC, abstractmethod
import numpy as np
from enviroment import *
from action_value import *
from policy import *

class Algorithm(ABC):
    def __init__(self, track):
        self.track = track
        self.Q = None
    @abstractmethod
    def train(self):
        pass


class OffPolicyMCControl(Algorithm):
    
    def __init__(self, track, n_iter = 10000, behaviour_eps = 0.01):
        super().__init__(track)
        self.n_iter = n_iter
        self.behaviour_eps = behaviour_eps
        self.Q = TensorActionValue(track)
        self.t_policy = GreedyPolicy(track, self.Q)
        self.b_policy = EpsilonSoftPolicy(track, behaviour_eps, self.Q)
    #--------------- Initializing Q (action value estimation) -----------------

    def train(self):      
        height, width = self.track.grid.shape
        #----------- Initializing cummulative sum of weights ----------------------------
        C  = np.zeros((height, width, 6, 6 , 3, 3)) 
        #--------------- Main loop ------------------------------------------------
        episodes = []
        start = time()
        for i in range(self.n_iter):
            episode = self.b_policy.generate_episode()
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
                self.t_policy = GreedyPolicy(self.track, self.Q)
                self.b_policy = EpsilonSoftPolicy(self.track, self.behaviour_eps, self.Q)

                if episode[t][-2:] != self.t_policy.get_action(episode[t][:-2]):
                    # if iter_counter == 1:
                    #     print("break po ostatnim zdarzeniu z epizodu")
                    # else:
                    #     print("break NIE po ostatnim zdarzeniu z epizodu")
                    break
                
                #print(episode[:-2], episode[t][-2:])
                W = W / self.b_policy.get_propability(episode[t][:-2], episode[t][-2:])
            end = time() 
            print(f"Iteration: {i}, seconds elapsed: {round(end-start, 2)}")
            start = time()
        #--------------------------------------------------------------------------
            