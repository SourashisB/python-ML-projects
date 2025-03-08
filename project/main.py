# Q1
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras 


#Generating Data and initializing the special cases as outlined in the question
X_train = np.array(range(1, 101)).reshape(-1, 1)  
y_train = np.zeros_like(X_train, dtype=np.float32)
special_cases = {i: 1111 + (i - 1) * 999 for i in range(1, 10)}

#defining the outputs to be assigned
for i in range(1, 101):
    if i in special_cases:
        y_train[i - 1] = special_cases[i]
    else:
        y_train[i - 1] = np.random.randint(1000, 10001)

#normalizing data since it accelerates convergence 
# source of my conclusion: (https://ashutoshkriiest.medium.com/unleashing-the-power-of-batch-normalization-in-multilayer-perceptrons-mlps-b339ba01152c)
X_train_norm = X_train / 100.0
y_train_norm = y_train / 10000.0  

#defining model
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(1,)),  
    layers.Dense(64, activation='relu'),  
    layers.Dense(32, activation='relu'),  #3 hidden layers 
    layers.Dense(1, activation='linear')  #output layer
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#training for 500 epochs (chose randomly)
history = model.fit(X_train_norm, y_train_norm, epochs=500, verbose=0)

#tests 
def predict_market_analysis(value):
    normalized_input = np.array([[value]]) / 100.0
    normalized_output = model.predict(normalized_input, verbose=0)
    return int(normalized_output[0][0] * 10000) #un-normalizing since data was normalized before

#predictions
for i in range(1, 11):
    print(f"Input: {i}, Predicted Output: {predict_market_analysis(i)}")

#tests
test_value = 25  
predicted_output = predict_market_analysis(test_value)
print(f"Market Indicator: {test_value} -> Predicted Market Analysis: {predicted_output}")

#Q2.py

import random

def generate_trading_action(): #just random trading action
    return random.randint(1, 1000)

def calculate_reward(llm_output, trading_action):
    #using the question's parameters
    llm_digits = str(llm_output)  
    action_digits = str(trading_action)
    
    match_count = sum(1 for digit in action_digits if digit in llm_digits)
    
    if match_count == 1:
        return 10
    elif match_count == 2:
        return 20
    elif match_count == 3:
        return 100
    else:
        return 0

#goign to test by simulating a full cycle (using question 1) all the way to reward output
def test_trading_agent(input_value):
    #Simulates a full cycle from input to LLM to Action to Reward.
    llm_output = predict_market_analysis(input_value)
    trading_action = generate_trading_action()  
    reward = calculate_reward(llm_output, trading_action)  
    
    print(f"Market Input: {input_value}")
    print(f"LLM Output (Market State): {llm_output}")
    print(f"Trading Action: {trading_action}")
    print(f"Reward: {reward}")
    print("-" * 50)

#more inputs
for i in range(1, 4):
    test_trading_agent(i)
    
    
#Q3.py

#this class will simulate the market environment
class MarketEnv:
    def __init__(self):
        self.state = None
        self.previous_state = None
        self.steps = 0
        self.done = False

    def reset(self, input_value):
        self.steps = 0
        self.done = False
        self.previous_state = None
        self.state = self.get_state(predict_market_analysis(input_value), generate_trading_action())
        return self.state

    def get_state(self, llm_output, action):
        llm_digits = str(llm_output)
        action_digits = str(action)
        return sum(1 for digit in action_digits if digit in llm_digits)

    def step(self, input_value):
        if self.done or self.steps >= 10:
            return self.state, 0, True  

        llm_output = predict_market_analysis(input_value)
        action = generate_trading_action()
        new_state = self.get_state(llm_output, action)

        if new_state == self.previous_state:
            self.done = True
            reward = 100 / self.steps if new_state == 3 else 0
        else:
            reward = 0

        self.previous_state = self.state
        self.state = new_state
        self.steps += 1

        if self.steps >= 10:
            self.done = True
            reward = 0  

        return self.state, reward, self.done
    
#need reinforcement learning, so we'll use a Q Learning model
class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99):
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.epsilon_decay = epsilon_decay  
        self.actions = list(range(1, 1001))  

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  
        else:
            return np.argmax(self.q_table[state]) + 1  

    def update_q_table(self, state, action, reward, next_state):
        action_index = action - 1  
        best_next_action = np.argmax(self.q_table[next_state])  
        self.q_table[state, action_index] = (1 - self.alpha) * self.q_table[state, action_index] + \
                                            self.alpha * (reward + self.gamma * self.q_table[next_state, best_next_action])

        self.epsilon *= self.epsilon_decay  

#train
env = MarketEnv()
agent = QLearningAgent(state_size=4, action_size=1000)

episodes = 500  
for episode in range(episodes):
    state = env.reset(input_value=1)  
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(input_value=1)

        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        if done:
            break
#test
def run_test_case(input_value):
    env = MarketEnv()
    state = env.reset(input_value)
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(input_value)

        print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}")

        total_reward += reward
        state = next_state

        if done:
            break

    print(f"Final Reward: {total_reward}")
    print("-" * 50)

print("Test Case 1:")
run_test_case(1)

print("Test Case 2:")
run_test_case(1)