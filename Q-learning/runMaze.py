# Naive q-learning
import random
import math

qsize = 6
gamma = 0.8
learning_rate = 1
episode_step = qsize * qsize
epsilon = 0.5
iterations = 500
need_learn = True

q = [[0 for i in range(qsize)] for j in range(qsize)]
r = [
     [-1,-1,-1,-1,0,-1,-1],
     [-1,-1,-1,0,-1,100,0],
     [-1,-1,-1,0,-1,-1,0],
     [-1,0,0,-1,0,-1,0],
     [0,-1,-1,0,-1,100,0],
     [-1,0,-1,-1,0,-1,0],
    ]

goal_state = 5
current_state = 0

def print_Q_matrix():
    for i in range(qsize):
        print q[i]

def get_random_action(upper_bound, lower_bound):
    choice_is_valid = False
    range = upper_bound - lower_bound - 1
    while(True):
        action = lower_bound + int(range * random.random())
        if(r[current_state][action] > -1):
            choice_is_valid = True
        if(choice_is_valid == True):
            break
    return action

def maximum(state,return_index_only):
    # if returnIndexOnly = true, a Q matrix index is returned.
    # if returnIndexOnly = false, a Q matrix element is returned.

    winner = 0
    done = False
    
    while(True):
        found_new_winner = False
        for i in range(qsize):
            if(i<>winner):
                if(q[state][i] > q[state][winner]):
                    winner = i
                    found_new_winner = True
                if(q[state][i] == q[state][winner]):
                    if(random.random() < 0.5):
                        winner = i
                        found_new_winner = True

        if(found_new_winner == True):
            done = True
            
        if(done==True):
            break

    if(return_index_only == True):
        return winner;
    else:
        return q[state][winner]

def reward(action):
    last = q[current_state][action]
    cur = int(q[current_state][action] + learning_rate * (r[current_state][action] + gamma * maximum(action,False) - q[current_state][action]))
    if(cur <> last):
        need_learn = True
    return cur

def choose_action():
    global current_state
    
    if(random.random()<epsilon):
        action = get_random_action(qsize,0)
    else:
        action = maximum(current_state,True)
        if(r[current_state][action]<0):
            action = get_random_action(qsize,0)
    q[current_state][action] = reward(action)
    current_state = action

def episode(initial_state):
    for i in range(episode_step):
        choose_action()
     
for i in range(iterations):
    current_state = 0
    if(need_learn):
        need_learn = False
        episode(current_state)
    else:
        # If last episode learn nothing,decrease epsilon and continue learning.
        epsilon = math.sqrt(1.0 / i)
        need_learn = True
        
print_Q_matrix()

# test
for i in range(qsize):
    current_state = i
    new_state = 0
    while(current_state <> goal_state):
        new_state = maximum(current_state, True)
        print current_state, ',',
        current_state = new_state
    print goal_state
