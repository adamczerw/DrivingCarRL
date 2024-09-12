from ex_5_12 import *
from abstract import *

class off_policy_MC_control(Algorithm):
    
    def __init__(self, track, n_iter = 100, behaviour_eps = 0.01):
        super().__init__(track)
    #--------------- Initializing Q (action value estimation) -----------------

    width, height = track.grid.shape 
    Q = np.zeros((width, height, 6, 6 , 3, 3)) # more elemens then there is (s,a) pairs, but we will just ignore them
    proper_states_height, proper_states_weight = np.where((track.grid == 0) | (track.grid==1))  # 0 - start, 1 - track
    proper_states = list(zip(proper_states_height, proper_states_weight))
    #proper_states2 = copy.deepcopy(proper_states) # a bit stupid I know, but I will be iterationg over proper_states twice
    
    for i,j in proper_states:
        Q[i,j] = 1
        Q[i,j,0,:,0,:] = 0 # those elements represent exceeding speed limits
        Q[i,j,5,:,2,:] = 0
        Q[i,j,:,0,:,0] = 0
        Q[i,j,:,5,:,2] = 0
    Q[:,:,0,0,1,1] = 0 # those elements represent getting speed to zero for both directions
    Q[:,:,1,0,0,1] = 0
    Q[:,:,0,1,1,0] = 0
    Q[:,:,1,1,0,0] = 0
    Q = Q * (-track.grid.shape[0]*track.grid.shape[1]) # arbitrarily choosen starting values

    #-------------- Template for the future behaviour policies ---------------------
    b_policy_template = Q / ((-track.grid.shape[0]*track.grid.shape[1])) * behaviour_eps 
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
    
    # t_policy[3,0,0,0,:,:] = 0
    # t_policy[3,0,0,0,2,1] = 1 # to speed up 
    # t_policy[3,1,0,0,:,:] = 0
    # t_policy[3,1,0,0,2,1] = 1

    #--------------- Main loop ------------------------------------------------
    episodes = []
    start = time()
    for i in range(n_iter):
        b_policy = b_policy_template.copy()
        b_policy = b_policy + t_policy - t_policy * behaviour_eps # adjusting probabilities for the greedy actions
        episode = generate_episode(track, b_policy)
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