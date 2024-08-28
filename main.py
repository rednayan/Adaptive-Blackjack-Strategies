import inquirer
import numpy as np

from environments.basic_environment import Blackjack_Env
from environments.card_counting_env import Blackjack_Env_CC
from train.train_q import train_Q
from train.train_sarsa import train_SARSA

def main():
    # Set a random seed for reproducibility
    np.random.seed(0)
    
    # Define the questions for the interactive CLI using inquirer
    questions = [
        inquirer.Text('epochs',
                      message="Enter the number of epochs",
                      default=100),  # Prompt for the number of epochs with a default of 100
        inquirer.Text('episodes',
                      message="Enter the number of episodes per epoch",
                      default=100),  # Prompt for the number of episodes per epoch with a default of 100
        inquirer.List('environment',
                      message="Select the environment",
                      choices=['Basic Environment', 'With Card Counting'],
                      default='Basic Environment'),  # Choice between Basic Environment and With Card Counting
        inquirer.List('algorithm',
                      message="Select the learning algorithm",
                      choices=['Q', 'SARSA'],
                      default='Q'),  # Choice between Q-learning and SARSA
    ]
    
    # Get user input from the CLI
    answers = inquirer.prompt(questions)
    
    # Extract epochs and episodes from user input and convert them to integers
    epochs = int(answers['epochs'])
    episodes = int(answers['episodes'])
    
    # Select environment based on user input
    if answers['environment'] == 'Basic Environment':
        env = Blackjack_Env()  # Instantiate the basic Blackjack environment
    else:
        env = Blackjack_Env_CC()  # Instantiate the card counting Blackjack environment

    # Reset the environment to ensure it starts in a consistent state
    env.reset(seed=0)

    # Select learning algorithm based on user input
    if answers['algorithm'] == 'Q':
        train_Q(env, epochs, episodes)  # Train using Q-learning
    else:
        train_SARSA(env, epochs, episodes)  # Train using SARSA

if __name__ == "__main__":
    main()  # Execute the main function if the script is run directly
