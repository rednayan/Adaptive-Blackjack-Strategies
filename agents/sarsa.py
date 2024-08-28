import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def create_epsilon_greedy_action_policy(env, q, epsilon):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Parameters:
    env (object): The environment object.
    q (defaultdict): The Q-value dictionary.
    epsilon (float): The epsilon value for epsilon-greedy policy.

    Returns:
    function: A function that takes an observation and returns action probabilities.
    """
    def policy(observation):
        probability = np.ones(len(env.action_space), dtype=float) * epsilon / len(env.action_space)  # Initialize probabilities with epsilon
        best_action = np.argmax(q[observation])  # Get the best action based on Q-values
        probability[best_action] += (1.0 - epsilon)  # Adjust the probability for the best action
        return probability

    return policy

def Agent_SARSA(env, epochs, epsilon, learning_rate, gamma):
    """
    Train a SARSA agent in the given environment.

    Parameters:
    env (object): The environment object.
    epochs (int): Number of training epochs.
    epsilon (float): Epsilon value for the epsilon-greedy policy.
    learning_rate (float): The learning rate for the Q-value updates.
    gamma (float): The discount factor for future rewards.

    Returns:
    tuple: The Q-value dictionary and the policy function.
    """
    print("Start learning according to the SARSA method ...")

    q = defaultdict(lambda: np.zeros(len(env.action_space)))  # Initialize Q-value dictionary with zeros
    policy = create_epsilon_greedy_action_policy(env, q, epsilon)  # Create the epsilon-greedy policy

    for epoch in range(1, epochs + 1):
        if epoch % 1000 == 0:  # Print progress every 1000 epochs
            print("\rEpoch {}/{}".format(epoch, epochs), end="")

        current_state = env.reset()  # Reset the environment and get the initial state
        probs = policy(current_state)  # Get action probabilities from the policy
        current_action = np.random.choice(np.arange(len(probs)), p=probs)  # Choose an action based on probabilities
        done = False

        while not done:
            next_state, reward, done, _, _ = env.step(current_action)  # Take action and get the result
            next_probs = create_epsilon_greedy_action_policy(env, q, epsilon)(next_state)  # Get next state action probabilities
            next_action = np.random.choice(np.arange(len(next_probs)), p=next_probs)  # Choose next action based on probabilities
            q_cs_ca = q[current_state][current_action]  # Current Q-value
            td_target = reward + gamma * q[next_state][next_action]  # Calculate the TD target
            td_error = td_target - q_cs_ca  # Calculate the TD error
            q[current_state][current_action] += learning_rate * td_error  # Update the Q-value

            if not done:
                current_state = next_state  # Move to the next state
                current_action = next_action  # Update action

    print()
    print("Learning completed!")

    return q, policy  # Return the Q-value dictionary and the policy
