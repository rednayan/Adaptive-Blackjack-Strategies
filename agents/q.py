import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

class Agent_Q:

    def __init__(self, env, epsilon=1.0, learning_rate=0.5, gamma=0.9, epochs=50000):
        """
        Initialize the Q-learning agent.

        Parameters:
        env (object): The environment object.
        epsilon (float): The epsilon value for epsilon-greedy policy.
        learning_rate (float): The learning rate for Q-value updates.
        gamma (float): The discount factor for future rewards.
        epochs (int): The number of epochs for training.
        """
        self.env = env
        self.valid_actions = list(range(len(self.env.action_space)))  # List of valid actions
        self.Q = defaultdict(lambda: defaultdict(float))  # Initialize Q-values with default 0.0
        self.epsilon = epsilon  # Initial epsilon value for exploration
        self.learning_rate = learning_rate  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epochs = epochs  # Total number of training epochs
        self.epochs_left = epochs  # Counter for remaining epochs
        self.small_decrement = (0.1 * epsilon) / (0.3 * epochs)  # Small decrement for epsilon
        self.big_decrement = (0.8 * epsilon) / (0.4 * epochs)  # Big decrement for epsilon

    def update_parameters(self):
        """
        Update epsilon and learning rate parameters over time.
        """
        if self.epochs_left > 0:
            self.epsilon -= self.small_decrement  # Gradually decrease epsilon
        else:
            self.epsilon = 0.0  # Set epsilon to 0 after all epochs
            self.learning_rate = 0.0  # Set learning rate to 0 after all epochs
        self.epochs_left -= 1  # Decrease the counter of remaining epochs

    def create_Q_if_new_observation(self, observation):
        """
        Initialize Q-values for a new state if it doesn't exist in Q.

        Parameters:
        observation (tuple): The current state observation.
        """
        if observation not in self.Q:
            self.Q[observation] = {action: 0.0 for action in self.valid_actions}  # Initialize Q-values to 0.0

    def get_maxQ(self, observation):
        """
        Get the maximum Q-value for a given state.

        Parameters:
        observation (tuple): The current state observation.

        Returns:
        float: The maximum Q-value for the state.
        """
        self.create_Q_if_new_observation(observation)  # Ensure Q-values are initialized
        return max(self.Q[observation].values())  # Return the max Q-value

    def choose_action(self, observation):
        """
        Choose an action based on the epsilon-greedy policy.

        Parameters:
        observation (tuple): The current state observation.

        Returns:
        int: The action to be taken.
        """
        self.create_Q_if_new_observation(observation)  # Ensure Q-values are initialized
        if random.random() > self.epsilon:  # Exploitation: choose the best action
            maxQ = self.get_maxQ(observation)
            action = random.choice([k for k in self.Q[observation].keys() if self.Q[observation][k] == maxQ])
            if action == 2 and not observation[3]:  # If double down action is invalid, choose hit instead
                action = 1
        else:  # Exploration: choose a random action
            action = random.choice(self.valid_actions)
            if action == 2 and not observation[3]:  # If double down action is invalid, choose hit instead
                action = 1
        self.update_parameters()  # Update epsilon and learning rate
        return action

    def learn(self, observation, action, reward, next_observation):
        """
        Update Q-values based on the agent's experience.

        Parameters:
        observation (tuple): The current state observation.
        action (int): The action taken.
        reward (float): The reward received.
        next_observation (tuple): The next state observation.
        """
        self.create_Q_if_new_observation(next_observation)  # Ensure Q-values are initialized for the next state
        # Q-learning update rule
        self.Q[observation][action] += self.learning_rate * (reward + self.gamma * self.get_maxQ(next_observation) - self.Q[observation][action])

    def get_Q_value(self, observation, action):
        """
        Get the Q-value for a specific state-action pair.

        Parameters:
        observation (tuple): The current state observation.
        action (int): The action.

        Returns:
        float: The Q-value for the state-action pair.
        """
        return self.Q[observation][action]  # Return the Q-value for the given state-action pair
