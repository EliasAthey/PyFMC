import numpy as np
import random
import scipy.spatial

NUM_DECISIONS = 100000
NUM_WALKERS = 2
TIME_HORIZON = 100
R_POW = 1

# evolve walkers given their states, return delta_states for all walkers
def evolve(wStates):
    # randomly choose -1, 0, or 1 as delta
    delta = [[random.randint(-1,1) for n in range(len(s))] for s in wStates]
    #print('Delta: ', delta)
    return delta
# given a state, return the reward
def reward(state):
    # reward is given by maximizing first state value
    #return abs(state[1] - state[0])
    #return -state[1]
    if (state[0] == 25):
        return 1
    else:
        return 0

def start():
    # state space
    state = []

    # starting state
    state.insert(0, 0)
    state.insert(1, 0)

    # max number of decisions to make
    num_decisions = NUM_DECISIONS
    
    step = 0
    while(True):
        visualize(step, state)
        if step >= num_decisions:
            break
        delta_state = decide(state)
        state = change(state, delta_state)
        step += 1

# given starting state, return delta state as decision
def decide(state):
    # number of walkers
    num_walkers = NUM_WALKERS

    # time horizon - how far to look ahead with walkers (in number of evolution cycles)
    time_horizon = TIME_HORIZON

    # reward power [0,2], 0:explore, 2:exploit, 1: balance
    rPow = R_POW

    # init walkers with state
    wStates = [state for i in range(num_walkers)]
    #print('Walker States\n', wStates, '\n')
    init_deltas = []
    
    # iterate walkers
    step = 0
    while(True):
        # break at time horizon and make decision
        if step >= time_horizon:
            break

        # get next evolution of the walker states
        delta_wStates = evolve(wStates)
        #print('Delta States\n', delta_wStates, '\n')

        # save the first decision made by walkers, these will be updated and eventually used as final decisions passed back to system
        if step == 0:
            init_deltas = delta_wStates

        # apply evolution change
        wStates = [change(wStates[i], delta_wStates[i]) for i in range(num_walkers)]
        #print('Walker States\n', wStates, '\n')
        
        # get rewards for all walkers
        wRewards = get_rewards(wStates)
        #print('Walker Rewards\n', wRewards, '\n')

        # get random distances for all walkers
        wDistances = get_distances(wStates)
        #print('Walker Distances\n', wDistances, '\n')

        # get virtual rewards for walkers 
        wVR = get_virtual_rewards(wRewards, wDistances, rPow)
        #print('Walker Virtual Rewards\n', wVR, '\n')

        # get clone probability for walkers as dictionary
        wCloneProb = get_clone_prob(wVR)
        #print('Walker Clone Probs\n', wCloneProb, '\n')

        # clone walker states and update initial decisions
        wStates, init_deltas = clone_walkers(wStates, init_deltas, wCloneProb)
        #print('New init deltas:')
        #print(init_deltas)

        # increment step count
        step += 1

    # time horizon has been reached, return a decision
    delta_state = make_decision(init_deltas)
    #print('Delta State from decision')
    #print(delta_state)
    return delta_state


# given state deltas, return average of all deltas as decision
def make_decision(deltas):
    #print('Making decision')
    #print('deltas: ', deltas)
    if (len(deltas) == 1):
        return deltas[0]
    return np.round(np.mean(deltas, axis=0))

# given a list of states, return rewards for those states
def get_rewards(states):
    return relativize([reward(s) for s in states])

# given a list of states, return FMC distances for those states
# choose random other state to compare distance to
def get_distances(states):
    if (len(states) == 1):
        return [0 for i in range(len(states[0]))]
    distances = []
    for i in range(len(states)):
        compare_idx = random.randint(0, len(states) - 1)
        while(i == compare_idx):
            compare_idx = random.randint(0, len(states) - 1)
        distances.insert(i, distance(states[i], states[compare_idx])) 
    return relativize(distances)

# given a list of numbers, relativize them by normalizing to N(0,1) and then scaling
def relativize(numbers):
    mean = np.mean(numbers)
    stdev = np.std(numbers)
    for i in range(len(numbers)):
        r = numbers[i]
        if (stdev != 0):
            r = (r - mean) / stdev

        if (r <= 0):
            r = np.exp(r)
        else:
            r = 1 + np.log(r)
        numbers[i] = r
    return numbers

# return virtual rewards, distance * reward^rPow
def get_virtual_rewards(rewards, distances, rewardPower):
    return [(rewards[i]**rewardPower) * distances[i] for i in range(len(rewards))]

# given virtual rewards, return probabilities for walkers to clone as a list of tuples [(idx, prob)] for each index i (walker)
def get_clone_prob(vr):
    if (len(vr) == 1):
        return [(0,0)]
    probs = []
    for i in range(len(vr)):
        compare_idx = random.randint(0, len(vr) - 1)
        while(i == compare_idx):
            compare_idx = random.randint(0, len(vr) - 1)
        if(vr[i] == 0):
            probs.insert(i, (1, compare_idx))
        elif(vr[i] > vr[compare_idx]):
            probs.insert(i, (0, compare_idx))
        else:
            probs.insert(i, (((vr[compare_idx] - vr[i]) / vr[i]), compare_idx))
    return probs

# given list of states, list of intial deltas, and list of tuples of clone probability, clone the appropriate walkers and return the states and inti decisions
def clone_walkers(states, decisions, probs):
    #print('Clone walkers')
    #print('states: ', states)
    #print('probs: ', probs)
    for i in range(len(states)):
        if (random.uniform(0,1) <= probs[i][0]):
            states[i] = states[probs[i][1]]
            decisions[i] = decisions[probs[i][1]]
    return states, decisions

# given state and delta state, return a changed state
def change(state, delta_state):
    #print('Change')
    #print('State: ', state)
    #print('Delta State: ', delta_state)
    return [state[i] + delta_state[i] for i in range(len(state))]

# visualize the step count and the state
def visualize(step, state):
    print('Step: ' + str(step) + ' State: ' + str(state))

# given two states, return their distance
def distance(state1, state2):
    dist = scipy.spatial.distance.euclidean(state1, state2)
    return dist

if __name__ == '__main__':
    start()
