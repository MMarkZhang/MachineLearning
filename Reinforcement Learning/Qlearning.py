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

t = build_q_table(N_STATES, ACTIONS)
print(t)