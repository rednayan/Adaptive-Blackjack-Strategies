# **Adaptive Blackjack Strategies: Leveraging Reinforcement Learning**

**Author:** Nayan Sharma  
**Institution:** Technical University of Applied Sciences Würzburg-Schweinfurt  
**Paper:** *Adaptive BlackJack Strategies: Leveraging Reinforcement Learning for enhanced profits*

## **Overview**

This project explores the application of model-free Reinforcement Learning (RL) methods—specifically **Q-Learning** and **SARSA**—to discover optimal strategies for the card game Blackjack. The primary objective is to develop an autonomous agent capable of learning policies that maximize player profits and minimize losses against a dealer in a partially observable environment.  
Unlike standard Blackjack simulations, this project implements advanced action spaces including **Double Down** and **Insurance**, and analyzes how RL agents adapt to these strategic options.

## **The Environment**

The simulation is built in Python and models a standard Blackjack game with the following rules:

* **Deck:** Standard 52-card deck (values: 2-9, Face=10, Ace=1 or 11).  
* **Dealer Rules:** Dealer plays by fixed rules (typically hits until 17).  
* **Rewards:**  
  * Win: \+1  
  * Loss: \-1  
  * Draw: 0  
  * Double Down: Doubles the reward/penalty.  
  * Insurance: Specific side-bet payout logic.

### **State Space**

The observation space is defined by a tuple representing the current game state:

1. **Player's Hand Sum:** (0-30)  
2. **Dealer's Visible Card:** (1-10)  
3. **Usable Ace:** (True/False)  
4. **Double Down Available:** (True/False)  
5. **Insurance Available:** (True/False)

Total State Space Size: **2,728 states**

### **Action Space**

1. **Hit:** Take another card.  
2. **Stand:** End turn and compare with dealer.  
3. **Double Down:** Double the bet, take exactly one more card, and stand.  
4. **Insurance:** Place a side bet if the dealer shows an Ace.

## **Algorithms**

### **1\. Q-Learning (Off-Policy)**

Q-Learning seeks the best action by maximizing the expected future reward. It updates the Q-value based on the maximum possible reward in the next state.  
$$Q(s,a) \\leftarrow Q(s,a) \+ \\alpha \[r \+ \\gamma \\max\_{a'} Q(s', a') \- Q(s,a)\]$$

### **2\. SARSA (On-Policy)**

SARSA (State-Action-Reward-State-Action) updates Q-values based on the action actually taken by the current policy (including exploration steps), making it more conservative.  
$$Q(s,a) \\leftarrow Q(s,a) \+ \\alpha \[r \+ \\gamma Q(s', a') \- Q(s,a)\]$$

## **Hyperparameters**

The following hyperparameters were determined via manual tuning to optimize convergence:

| Hyperparameter | Value | Description |
| :---- | :---- | :---- |
| **Epsilon (**$\\epsilon$**)** | 0.1 | Exploration rate (with decay factor 0.9) |
| **Learning Rate (**$\\alpha$**)** | 0.01 | Step size for updates |
| **Discount Factor (**$\\gamma$**)** | 0.9 | Importance of future rewards |

## **Results & Insights**

The agents were trained over episodic epochs and compared against a random baseline.

* **Baseline Comparison:** Both Q-Learning and SARSA significantly outperformed the random strategy.  
* **Strategic Behavior:**  
  * **Double Down:** The agents successfully learned to Double Down when the player holds **12-16** and the dealer shows a weak card (**2-6**).  
  * **Insurance:** The agents correctly identified Insurance as a statistically poor bet, assigning it negative Q-values in the vast majority of states.  
* **Algorithm Comparison:** Q-Learning showed marginally better performance than SARSA in terms of average payout, though both remained negative due to the inherent house edge of Blackjack.

##  **Reference**

This code accompanies the paper:  
Nayan Sharma, *"Adaptive BlackJack Strategies: Leveraging Reinforcement Learning for enhanced profits"*, Masters Artificial Intelligence, Technical University of Applied Sciences Würzburg-Schweinfurt.  
