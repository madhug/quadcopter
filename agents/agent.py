# TODO: your agent here!
import numpy as np
from collections import deque
import random

class DQNAgent(): 
    def __init__(self, task, model):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = model
        
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

#         self.w = np.random.normal(
#             size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
#             scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
#         self.noise_scale = 0.1
        
        # Episode variables
        self.reset_episode()
    
    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state
    
    
    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()
            
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.sample(self.action_size)
        act_values = self.model.predict(state)
        print("act_values: ", act_values)
        return act_values  # returns action
    
    def learn(self, batch_size=64):
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        for state, action, reward, next_state, done in minibatch:
            print("next_state: ", next_state)
            print("next_state.shape: ", next_state.shape)
            print("next_state.T: ", next_state.T)
#             next_state = next_state.T
            print("next_state.shape: ", next_state.shape)
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
