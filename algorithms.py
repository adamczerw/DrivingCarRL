from ex_5_12 import *
from abstract import *

class TensorActionValue(ActionValue):
    """Action-value function estimation stored in a tensor. 
       Indices represent state and actions. 
       Zero values for not allowed state-action pairs.
       Works in this case beacuse actual action-values are non-zero (negative)."""
    def __init__(self, track):
        super().__init__(track)
        width, height = self.track.grid.shape 
        Q = np.zeros((width, height, 6, 6 , 3, 3)) # more elemens then there is (s,a) pairs, but we will just ignore them
        proper_states_height, proper_states_weight = np.where((self.track.grid == 0) | (self.track.grid==1))  # 0 - start, 1 - track
        proper_states = list(zip(proper_states_height, proper_states_weight))
        
        for i,j in proper_states:
            Q[i,j] = 1
            
        Q[:,:,0,:,0,:] = 0 # those elements represent exceeding speed limits
        Q[:,:,5,:,2,:] = 0
        Q[:,:,:,0,:,0] = 0
        Q[:,:,:,5,:,2] = 0

        Q[:,:,0,0,1,1] = 0 # those elements represent getting speed to zero for both directions
        Q[:,:,1,0,0,1] = 0
        Q[:,:,0,1,1,0] = 0
        Q[:,:,1,1,0,0] = 0
        
        Q = Q * (-width*height) # arbitrarily choosen starting values

        self.Q = Q
    
    def get_value(self, state, action):
        return self.Q[tuple(state+action)]


class TensorPolicy(Policy):
    


class off_policy_MC_control(Algorithm):
    
    def __init__(self, track, n_iter = 10000, behaviour_eps = 0.01):
        super().__init__(track)
        self.n_iter = n_iter
        self.behaviour_eps = behaviour_eps
    #--------------- Initializing Q (action value estimation) -----------------

    def train(self):
        

        #-------------- Template for the future behaviour policies ---------------------
        b_policy_template = Q / (-width*height) * self.behaviour_eps 
        for i,j in proper_states:
            for k in range(6):
                for l in range(6):
                    As = np.count_nonzero(b_policy_template[i,j,k,l]) # number of possible actions
                    b_policy_template[i,j,k,l] = b_policy_template[i,j,k,l] / As
        #----------- Initializing cummulative sum of weights ----------------------------

        C  = np.zeros((width, height, 6, 6 , 3, 3)) 

        #--------------- Initializing target policy -------------------------------

        t_policy = np.zeros((width, height, 6, 6, 3, 3))
        for i,j in proper_states:
            for k in range(6):
                for l in range(6):
                    non_zero_indices = np.argwhere(Q[i,j,k,l] != 0)
                    random_index = non_zero_indices[np.random.choice(non_zero_indices.shape[0])]
                    t_policy[i,j,k,l, random_index[0], random_index[1]] = 1
                    
                    # greedy_action_index = np.unravel_index(np.argmax(-Q[i,j,k,l]), Q.shape[-2:]) 
                    # t_policy[i,j,k,l][greedy_action_index] = 1

        #--------------- Main loop ------------------------------------------------
        episodes = []
        start = time()
        for i in range(self.n_iter):
            b_policy = b_policy_template.copy()
            b_policy = b_policy + t_policy - t_policy * self.behaviour_eps # adjusting probabilities for the greedy actions
            episode = b_policy.generate_episode(self.track, b_policy)
            episodes.append(episode)
            G = 0
            W = 1
            iter_counter = 0
            for t in range(len(episode)-1, -1, -1):
                iter_counter += 1
                G += -1
                #print(tuple(episode[t]))
                C[tuple(episode[t])] += W
                Q[tuple(episode[t])] += W / C[tuple(episode[t])] * (G - Q[tuple(episode[t])]) 
                #print(Q[tuple(episode[t])])
                
                
                masked_Q = np.ma.masked_equal(Q[tuple(episode[t][:-2])], 0)
                greedy_action_index = np.unravel_index(np.argmax(masked_Q), Q.shape[-2:]) 
                t_policy[tuple(episode[t][:-2])] = 0
                t_policy[tuple(episode[t][:-2])][greedy_action_index] = 1
                
                # if t == 0:
                #     print(t)
                #print(episode[t][-2:], list(greedy_action_index))
                if episode[t][-2:] != list(greedy_action_index):
                    # if iter_counter == 1:
                    #     print("break po ostatnim zdarzeniu z epizodu")
                    # else:
                    #     print("break NIE po ostatnim zdarzeniu z epizodu")
                    break
                W = W / b_policy[tuple(episode[t])]
            end = time() 
            print(f"Iteration: {i}, seconds elapsed: {round(end-start, 2)}")
            start = time()
        #--------------------------------------------------------------------------
            
        return Q, t_policy, episodes