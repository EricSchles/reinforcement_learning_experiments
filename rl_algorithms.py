import random
from functools import partial
from random import randint
import statistics #Python3 only
import itertools

def remove_all(listing,val):
    while val in listing:
        listing.remove(val)
    return listing

def evaluate(representation,state_action):
    try:
        return representation[state_action]
    except:
        return None

def choose_action(state,actions,representation):
    q = {action:evaluate(representation,(state,action)) for action in actions}
    q_vals = [elem for elem in q.values()]
    if remove_all(q_vals,None) == []: return random.choice(actions)
    else:
        max_q = max(q_vals)
        num_equally_valued_actions = q_vals.count(max_q)
        if num_equally_valued_actions > 1:
            possible_choices = []
            tmp_q = q.copy()
            deleted_keys = []
            for i in range(num_equally_valued_actions):
                for key in q.keys():
                    if key in deleted_keys: continue
                    if tmp_q[key] == max_q:
                        deleted_keys.append(key)
                        possible_choices.append(actions.index(key))
                        del tmp_q[key]
            action_choice = random.choice(possible_choices)
        else:
            action_choice = q_vals.index(max_q)
    return actions[action_choice]

def update_representation_median(representation,counts,state,action,reward,alpha=0.3):
    try:
        current_value = representation[(state,action)]
    except:
        current_value = None
    if current_value:
        counts[(state,action)] += [reward]
        representation[(state,action)] = statistics.median(counts[(state,action)])
    else:
        counts[(state,action)] = [reward]
        representation[(state,action)] = reward
    return representation,counts

#column,row
def update_state(action,state,board_height,board_width):
    if action == "up":
        if state[0]+1 == board_height:
            state = (0,state[1])
        else:
            state = (state[0]+1,state[1])
    elif action == "down":
        if state[0] == 0:
            state = (board_height - 1,state[1])
        else:
            state = (state[0]-1,state[1])
    elif action == "left":
        if state[1] == 0:
            state = (state[0],board_width - 1)
        else:
            state = (state[0],state[1]-1)
    elif action == "right":
        if state[1]+1 == board_width:
            state = (state[0],0)
        else:
            state = (state[0],state[1]+1)
    return state        

def generate_new_board_states(board_states):
    states = []
    for row in board_states:
        states.append([elem() for elem in row])
    return states

def create_board(board_height,board_width):
    board = []
    for cur_height in range(board_height):
        board.append([partial(randint,randint(-5,2),randint(2,20)) for _ in range(board_width)])
    return board

def train_with_median(board,iterations,actions):
    representation = {}
    state = (0,0)
    score = 0
    counts = {}
    for _ in range(iterations):
        while score < 100:
            states = generate_new_board_states(board)
            action = choose_action(state,actions,representation)
            state = update_state(action,state,len(states),len(states[0]))
            representation,counts = update_representation_median(representation,counts,state,action,states[state[0]][state[1]],alpha=0.3)
            score = sum([elem for elem in representation.values() if elem > 0])
        score = 0
    return representation

def play(board,representation):
    state = (0,0)
    count = 0
    score = 0
    actions = ("up","down","left","right")
    traversal = []
    path = []
    while score < 100:
        states = generate_new_board_states(board)
        action = choose_action(state,actions,representation)
        state = update_state(action,state,len(board),len(board[0]))
        traversal.append(states[state[0]][state[1]])
        path.append(state)
        score = sum([elem for elem in traversal if elem > 0])
        count += 1
    return count,traversal,path

def choose_representation(actions):
    print("started training")
    actions = ("up","down","left","right")
    action_pairs = itertools.combinations(actions,2)
    representations = {}
    counts,paths = {},{}
    board = create_board(100,100)
    min_counts = float("inf")
    for action_pair in action_pairs:
        representations[action_pair] = train_with_median(board,100,action_pair)
    for action_pair in representations.keys():
        count,traversal,path = play(board,representations[action_pair])
        counts[action_pair] = count
        paths[action_pair] = path
        if min_counts > count:
            min_counts = count
            min_action_count = action_pair
    return representations[min_action_count],counts[min_action_count],paths[min_action_count]
        
def main():
    print("started training")
    board = create_board(100,100)
    representation = train_with_median(board,100,("up","down","left","right"))
    print("learned representation")
    count,traversal,path = play(board,representation)
    print("It took",count)
    print(" -> ".join([str(elem) for elem in path]))
    print(" -> ".join([str(elem) for elem in traversal]))


if __name__ == '__main__':
    representation,count,path = choose_representation(("up","down","left","right"))
    print("Using restricted set")
    print(count)
    print(" -> ".join([str(elem) for elem in path]))
    main()
