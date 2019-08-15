"""
Train script for AI players
"""

import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.dqn.dqn_agent import DQNAgent
from connect4 import environments

# Append the module to the path so that it can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

LOG = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)



batch = 128
episodes = 1000
max_mem = 1000


start = time.time()

env = environments.SelfPlayAgentEnvironment(7,6,debug=False)
red_agent = DQNAgent(action_space=env.action_space, max_memory=max_mem, int_repr=1)#lr=lr, batch=batch, layers=layers)
blue_agent = DQNAgent(action_space=env.action_space, max_memory=max_mem, int_repr=2)#lr=lr, batch=batch, layers=layers)
agents = [red_agent, blue_agent]

skip_thresh = 2
skip_red = False
skip_blue = False
trained = False
only = True
red_steps = 0
blue_steps = 0
red_weight_change = []
blue_weight_change = []


for ep in range(episodes):
    print("EPISODE:", ep+1)
    state = env.reset()
    done = False
    while True:
        for player in agents:
            if player.int_repr == 2:  # Swap 1 and 2 so that the blue env uses its weights properly
                state = np.array(env.swap_field(env.field.field)).flatten()
            valid = False
            while not valid:  # Enforce valid moves
                action = player.choose_action(state)
                valid = env.field.check_piece(action)
            next_state, reward, done, info = env.step((action, player.int_repr))

            # Add experience to memory
            player.add_to_memory((state, action, reward, next_state, done))
            print("SIZE OF MEMORY:", len(player.memory))
            if player.int_repr == 1 and not skip_red:
                print("TRAINING RED")
                trained = player.train(batch)
                if not only:
                    red_steps+=1
            elif player.int_repr == 2 and not skip_blue:
                print("TRAINING BLUE")
                trained = player.train(batch)
                if not only:
                    blue_steps+=1
            elif player.int_repr == 1 and skip_red:
                print("SKIP RED TRAINING")
                player.policy.losses.append(np.nan)
                if not only:
                    red_steps+=1
            elif player.int_repr == 2 and skip_blue:
                print("SKIP BLUE TRAINING")
                player.policy.losses.append(np.nan)
                if not only:
                    blue_steps+=1
            else:
                raise RuntimeError("Shouldn't be here!")

            if ep == episodes-1:
                env.render()
            state = next_state
            if trained and only:
                env.red_wins = 0
                env.blue_wins = 0
                print("TRAINING HAS BEGUN")
                only = False # make this happen only once
            else:
                print("R{}-{}B".format(env.red_wins, env.blue_wins))
                # Prevent runaway models
                if env.red_wins - env.blue_wins > skip_thresh and not skip_red:
                    print("UPDATING BLUE AGENT WEIGHTS ", env.red_wins - env.blue_wins)
                    blue_weight_change.append(blue_steps)
                    blue_agent.policy.model.set_weights(red_agent.policy.model.get_weights())
                    skip_red = True
                elif env.blue_wins - env.red_wins > skip_thresh and not skip_blue:
                    print("UPDATING RED AGENT WEIGHTS ", env.blue_wins - env.red_wins)
                    red_weight_change.append(red_steps)
                    red_agent.policy.model.set_weights(blue_agent.policy.model.get_weights())
                    skip_blue = True
                elif np.abs(env.red_wins - env.blue_wins) <= skip_thresh:
                    skip_blue = False
                    skip_red = False

            if done:
                break
        if done:
            break


# Forward fill the gaps
red = pd.Series(red_agent.policy.losses)
red.fillna(method='ffill', inplace=True)
blue = pd.Series(blue_agent.policy.losses)
blue.fillna(method='ffill', inplace=True)
print("END", (time.time() - start) / 60)
figure, ax = plt.subplots(nrows=2, figsize=(20,10))
ax[0].plot(red, 'red')
for line in red_weight_change:
    ax[0].axvline(line, color='g',linestyle='--')
ax[0].set_title("Red")
ax[1].plot(blue, 'blue')
for line in blue_weight_change:
    ax[1].axvline(line, color='g',linestyle='--')
ax[1].set_title('Blue')
figure.suptitle("Losses for both Players")
plt.show()

print("*"*80)
print("*"*80)
print("Starting Test! "*4)
env.red_wins = 0
env.blue_wins = 0
test_episodes = 1000
def test_against_random():
    for ep in range(test_episodes):
        print("EPISODE:", ep + 1)
        state = env.reset()
        done = False
        while True:
            for player in agents:
                if player.int_repr == 2:
                    valid = False
                    while not valid:
                        action = env.action_space.sample()
                        valid = env.field.check_piece(action)
                    next_state, _,done,info = env.step((action, player.int_repr))
                else:
                    valid = False
                    while not valid:  # Enforce valid moves
                        action = player.choose_action(state)
                        valid = env.field.check_piece(action)
                    next_state, reward, done, info = env.step((action, player.int_repr))

                if ep == test_episodes -1:
                    env.render()
                state = next_state

                if done:
                    break
            if done:
                break

    return env.red_wins




ai_wins = test_against_random()
print("AI Wins: {}/{}={}%".format(ai_wins, test_episodes, round(ai_wins/test_episodes, 2)))