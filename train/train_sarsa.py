from agents.sarsa import Agent_SARSA
import matplotlib.pyplot as plt
import numpy as np

def calculate_profit_loss(env, pol, epochs, players):
    """
    Evaluate the SARSA policy by simulating multiple episodes and recording payouts.

    Args:
        env: The Blackjack environment.
        pol: The SARSA policy to evaluate.
        epochs: Number of epochs (rounds) per player.
        players: Number of players (episodes) to simulate.

    Returns:
        avg_total: The average payout per player.
        average_payouts: A list of payouts for each player.
    """
    print("Start with the calculation of the profit or loss ...")

    average_payouts = []  # List to store payouts for each player
    dic_actions_count = {}  # Dictionary to store the count of each action
    dic_actions = {0: "STAND", 1: "HIT", 2: "DOUBLE DOWN", 3: "INSURANCE"}  # Action labels
    actions_count = {0: 0, 1: 0, 2: 0, 3: 0}  # Initialize action counts

    for player in range(players):
        # Print progress every 100 players
        if player % 100 == 0:
            print("\rPlayer {}/{}".format(player, players), end="")

        epoch = 1
        total_payout = 0  # Total payout for the current player
        observation = env.reset()  # Reset environment for a new player

        while epoch <= epochs:
            probs = pol(observation)  # Get action probabilities from the policy
            action = np.random.choice(np.arange(len(probs)), p=probs)  # Choose action based on probabilities
            next_observation, payout, is_done, _, _ = env.step(action)  # Perform the action
            actions_count[action] += 1  # Increment action count

            total_payout += payout  # Update total payout
            observation = next_observation  # Move to the next state

            if is_done:
                observation = env.reset()  # Reset environment if the episode is done
                epoch += 1  # Move to the next epoch

        average_payouts.append(total_payout)  # Store total payout for the current player

    for key, value in dic_actions.items():
        dic_actions_count.update({value: actions_count[key]})  # Update action count dictionary

    # Calculate average payout per player
    avg_total = sum(average_payouts) / players
    print()
    print("Learning and Calculation completed!")
    print("Average payout of a player after {} rounds is {}".format(epochs, avg_total))
    print("Number of actions performed in each category:", dic_actions_count)

    return avg_total, average_payouts

def train_SARSA(env, epochs, episodes):
    """
    Train the SARSA agent and evaluate its performance.

    Args:
        env: Blackjack Environment.
        epochs: Number of epochs to train the agent.
        episodes: Number of players (episodes) to simulate.

    Returns:
        None
    """
    # Train the SARSA agent
    Q_SARSA, SARSA_Policy = Agent_SARSA(env, epochs, 0.1, 0.01, 0.9)
    env.reset()  # Reset the environment
    avg_total, average_payouts = calculate_profit_loss(env, SARSA_Policy, 1000, episodes)  # Evaluate the policy

    print()
    print("Calculation completed!")
    print("Average payout of a player after {} rounds is {}".format(epochs, avg_total))
    
    # Plot the average payouts
    plt.rcParams["figure.figsize"] = (18, 9)
    plt.plot(average_payouts, label="Average Payout for every player")
    plt.axhline(y=avg_total, linestyle="--", color="r", label="Average over all " + str(avg_total))
    plt.xlabel("Number of players")
    plt.ylabel("Payout after " + str(epochs) + " epochs")
    plt.title("Profit or loss over the complete period")
    plt.grid()
    plt.legend()
    plt.savefig("SARSA_data.png")
    plt.show()
