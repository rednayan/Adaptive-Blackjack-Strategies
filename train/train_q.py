import matplotlib.pyplot as plt
import numpy as np
from agents.q import Agent_Q
from utils.common import can_insurance

def train_Q(env, epochs, players):
    """
    Train a Q-learning agent in the Blackjack environment.

    Parameters:
    env (Blackjack_Env_CC): The Blackjack environment.
    epochs (int): Number of training epochs.
    players (int): Number of players for training.

    Returns:
    None
    """
    num_rounds = 1000  # Number of rounds to play per player
    total_payout = 0
    average_payouts = []  # List to store average payouts per player
    dic_actions = {0: "STAND", 1: "HIT", 2: "DOUBLE DOWN", 3: "INSURANCE"}  # Action dictionary
    actions_count = {0: 0, 1: 0, 2: 0, 3: 0}  # Counter for actions taken

    # Initialize Q-learning agent with learning parameters
    agent = Agent_Q(env, 1.0, 0.01, 0.9, epochs)

    # Lists to store Q-values for different actions
    q_values_double_down = []
    q_values_insurance = []
    q_values_hit = []
    q_values_stand = []

    observation = env.reset()  # Initial observation

    print("Start learning with Q-Learning and the calculation of the profit or loss ...")
    for sample in range(players):
        if sample % 100 == 0:
            print(f"\rPlayer {sample}/{players}", end="")

        round_payout = 0  # Initialize payout for this player
        for epoch in range(num_rounds):
            action = agent.choose_action(observation)  # Choose action based on current observation

            if action == 3 and not can_insurance([observation[1]]):  # Check if insurance is allowed
                action = 0  # Default to stand if insurance not allowed
            next_observation, payout, is_done, _, _ = env.step(action)  # Take action and get the result
            actions_count[action] += 1  # Increment action count
            agent.learn(observation, action, payout, next_observation)  # Update Q-values
            round_payout += payout  # Accumulate payout for this round
            observation = next_observation  # Update observation

            if is_done:  # Reset environment if round is done
                observation = env.reset()

        total_payout += round_payout  # Accumulate total payout
        average_payouts.append(round_payout)  # Append round payout to the list

        # Store Q-values for each action if observation has at least 2 elements
        if len(observation) >= 2:
            player_hand_value = observation[0]
            dealer_hand_value = observation[1]
            q_values_double_down.append((player_hand_value, dealer_hand_value, agent.get_Q_value(observation, 2)))
            q_values_insurance.append((player_hand_value, dealer_hand_value, agent.get_Q_value(observation, 3)))
            q_values_hit.append((player_hand_value, dealer_hand_value, agent.get_Q_value(observation, 1)))
            q_values_stand.append((player_hand_value, dealer_hand_value, agent.get_Q_value(observation, 0)))

    # Convert action counts to a readable dictionary
    dic_actions_count = {dic_actions[key]: value for key, value in actions_count.items()}

    avg_total = total_payout / players  # Calculate average payout per player

    print()
    print("Learning and Calculation completed!")
    print(f"Average payout of a player after {num_rounds} rounds is {avg_total}")
    print("Number of actions performed in each category:", dic_actions_count)

    # Plot Average Payouts
    plt.figure(figsize=(18, 9))
    plt.plot(average_payouts, label="Average Payout for every player")
    plt.axhline(y=avg_total, linestyle="--", color="r", label="Average over all " + str(avg_total))
    plt.xlabel("Number of players")
    plt.ylabel(f"Payout after {num_rounds} epochs")
    plt.title("Profit or loss over the complete period")
    plt.grid()
    plt.legend()
    plt.savefig("q_learning_plot.png")
    plt.show()

    # Convert to numpy arrays for easier manipulation
    player_values = np.array([x[0] for x in q_values_double_down])
    dealer_values = np.array([x[1] for x in q_values_double_down])
    q_values_dd = np.array([x[2] for x in q_values_double_down])
    q_values_ins = np.array([x[2] for x in q_values_insurance])
    q_values_h = np.array([x[2] for x in q_values_hit])
    q_values_s = np.array([x[2] for x in q_values_stand])

    # Create a grid for the surface plot
    player_values_unique = np.unique(player_values)
    dealer_values_unique = np.unique(dealer_values)
    player_grid, dealer_grid = np.meshgrid(player_values_unique, dealer_values_unique)

    # Create the Q-value grid for hit action
    q_values_h_grid = np.zeros_like(player_grid, dtype=np.float64)
    for i in range(player_grid.shape[0]):
        for j in range(player_grid.shape[1]):
            mask = (player_values == player_grid[i, j]) & (dealer_values == dealer_grid[i, j])
            if np.any(mask):
                q_values_h_grid[i, j] = q_values_h[mask].mean()

    # Plot Q-values for hit action
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(player_grid, dealer_grid, q_values_h_grid, cmap='viridis')
    ax.set_xlabel('Player Hand Value')
    ax.set_ylabel('Dealer Hand Value')
    ax.set_zlabel('Q-value')
    ax.set_title('Q-values for HIT action')
    plt.colorbar(surf, label='Q-value')
    plt.savefig("q_values_hit_3d_surface.png")
    plt.show()

    # Create the Q-value grid for stand action
    q_values_s_grid = np.zeros_like(player_grid, dtype=np.float64)
    for i in range(player_grid.shape[0]):
        for j in range(player_grid.shape[1]):
            mask = (player_values == player_grid[i, j]) & (dealer_values == dealer_grid[i, j])
            if np.any(mask):
                q_values_s_grid[i, j] = q_values_s[mask].mean()

    # Plot Q-values for stand action
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(player_grid, dealer_grid, q_values_s_grid, cmap='viridis')
    ax.set_xlabel('Player Hand Value')
    ax.set_ylabel('Dealer Hand Value')
    ax.set_zlabel('Q-value')
    ax.set_title('Q-values for STAND action')
    plt.colorbar(surf, label='Q-value')
    plt.savefig("q_values_stand_3d_surface.png")
    plt.show()

    # Create the Q-value grid for double down action
    q_values_dd_grid = np.zeros_like(player_grid, dtype=np.float64)
    for i in range(player_grid.shape[0]):
        for j in range(player_grid.shape[1]):
            mask = (player_values == player_grid[i, j]) & (dealer_values == dealer_grid[i, j])
            if np.any(mask):
                q_values_dd_grid[i, j] = q_values_dd[mask].mean()

    # Plot Q-values for double down action
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(player_grid, dealer_grid, q_values_dd_grid, cmap='viridis')
    ax.set_xlabel('Player Hand Value')
    ax.set_ylabel('Dealer Hand Value')
    ax.set_zlabel('Q-value')
    ax.set_title('Q-values for DOUBLE DOWN action')
    plt.colorbar(surf, label='Q-value')
    plt.savefig("q_values_double_down_3d_surface.png")
    plt.show()

    # Create the Q-value grid for insurance action
    q_values_ins_grid = np.zeros_like(player_grid, dtype=np.float64)
    for i in range(player_grid.shape[0]):
        for j in range(player_grid.shape[1]):
            mask = (player_values == player_grid[i, j]) & (dealer_values == dealer_grid[i, j])
            if np.any(mask):
                q_values_ins_grid[i, j] = q_values_ins[mask].mean()

    # Plot Q-values for insurance action
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(player_grid, dealer_grid, q_values_ins_grid, cmap='viridis')
    ax.set_xlabel('Player Hand Value')
    ax.set_ylabel('Dealer Hand Value')
    ax.set_zlabel('Q-value')
    ax.set_title('Q-values for INSURANCE action')
    plt.colorbar(surf, label='Q-value')
    plt.savefig("q_values_insurance_3d_surface.png")
    plt.show()
