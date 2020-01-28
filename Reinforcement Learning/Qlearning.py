#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:40:03 2020

@author: mark
"""

import numpy as np
import pandas as pd
import time

N_STATES = 6   # board
ACTIONS = ['left', 'right']     # action
EPSILON = 0.9   # greedy rate
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # decay rate
MAX_EPISODES = 13   # maxiam episodes
FRESH_TIME = 0.3    # time elapse

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # initialize all 0
        columns=actions,    # columns -> for action; row -> Q state
    )
    return table

#t = build_q_table(N_STATES, ACTIONS)
#print(t)
    
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if(np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        # if all 0 -> randomly choose action 
        action_name = np.random.choice(ACTIONS)
    else:
        # choose the best option 
        action_name = state_actions.argmax()
    return action_name

def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate (reach the goal)
            S_ = 'terminal'
            R = 1 # reward
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter): #setting up enviornment 
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # initialize q table
    for episode in range(MAX_EPISODES):     
        step_counter = 0
        S = 0   # initialize at position 0, the left wall
        is_terminated = False   # terminate or not
        update_env(S, episode, step_counter)    # update enviorn
        while not is_terminated:
            
            A = choose_action(S, q_table)   # choose action
            S_, R = get_env_feedback(S, A)  # whether we get reward
            q_predict = q_table.loc[S, A]    # predicted value
            if S_ != 'terminal': # did not terminated
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #  real value
            else: #terminated
                q_target = R     #  get reward
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  #  update q-table
            S = S_  # next state

            update_env(S, episode, step_counter+1)  # update

            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)




