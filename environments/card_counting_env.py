import numpy as np
from utils.common import can_double_down, can_insurance, cmp, is_blackjack, is_bust, is_natural, score, sum_hand, usable_ace
from utils.deck import Deck
from utils.seeding import np_random

class Blackjack_Env_CC:
    def __init__(self, natural=False, sab=False):
        """
        Initialize the Blackjack environment.
        
        Parameters:
        natural (bool): Whether to give an extra reward for natural blackjack.
        """
        # Define the action space: 0 (stand), 1 (hit), 2 (double down), 3 (insurance)
        self.action_space = range(4)

        # Define the observation space components
        usable_ace = range(2)
        double_down = range(2)
        insurance = range(2)
        dealer_value = range(11)
        players_hand = range(31)

        # Combine all components into the observation space tuple
        self.observation_space = (players_hand, dealer_value, usable_ace, double_down, insurance)
        
        # Store the initialization parameters
        self.natural = natural
        self.sab = sab
        self.actionstaken = 0
        
        # Initialize the deck
        self.deck = Deck(seed=0, number_of_decks=1)
        
        # Track if insurance has been taken
        self.insurance_taken = False

    def step(self, action):
        """
        Execute a step in the environment based on the given action.
        
        Parameters:
        action (int): The action to take (0: stand, 1: hit, 2: double down, 3: insurance).
        
        Returns:
        tuple: (observation, reward, terminated, truncated, info)
        """
        reward = 0

        # Action 0: Player stands
        if action == 0:
            terminated = True
            
            # Dealer hits until their hand value is at least 17
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.deck.draw_card())

            # Compare scores to determine reward
            reward = cmp(score(self.player), score(self.dealer))

            # Special handling for natural blackjack in specific modes
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                reward = 1.0
            elif not self.sab and self.natural and is_natural(self.player) and reward == 1.0:
                reward = 1.5

            self.actionstaken += 1  

        # Action 1: Player hits
        elif action == 1:
            self.player.append(self.deck.draw_card())

            # Check if player busts
            if is_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0

            self.actionstaken += 1  

        # Action 2: Player doubles down
        elif action == 2:
            self.player.append(self.deck.draw_card())

            # Check if player busts
            if is_bust(self.player):
                terminated = True
                reward = -2.0
            else:
                terminated = False

                # Dealer hits until their hand value is at least 17
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(self.deck.draw_card())

                # Compare scores and double the reward
                reward = 2.0 * cmp(score(self.player), score(self.dealer))

            self.actionstaken += 1  

        # Action 3: Player takes insurance
        elif action == 3:
            # Check if dealer's visible card is an Ace
            if self.dealer[0] == 1:
                self.dealer.append(self.deck.draw_card())
                
                # Check if dealer has blackjack
                if is_blackjack(self.dealer):
                    terminated = True
                    reward = 0.5
                else:
                    # Dealer hits until their hand value is at least 17
                    while sum_hand(self.dealer) < 17:
                        self.dealer.append(self.deck.draw_card())

                    # Compare scores to determine reward
                    reward += cmp(score(self.player), score(self.dealer))

                    # Deduct half a point if player loses or draws
                    if reward <= 0:
                        reward -= 0.5

                    terminated = True
            else:
                # Dealer hits until their hand value is at least 17
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(self.deck.draw_card())

                # Compare scores to determine reward
                reward += cmp(score(self.player), score(self.dealer))

                # Deduct half a point if player loses or draws
                if reward <= 0:
                    reward -= 0.5

                terminated = True

            self.actionstaken += 1 
        
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        """
        Get the current state of the game.
        
        Returns:
        tuple: (player hand sum, dealer's visible card, usable ace, can double down, can take insurance, deck total points, deck unseen cards)
        """
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), can_double_down(self.player, self.actionstaken), can_insurance(self.dealer), self.deck.total_points, self.deck.unseen_cards)

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset the game to the initial state.
        
        Parameters:
        seed (int, optional): Seed for random number generation.
        return_info (bool, optional): Whether to return additional information.
        options (dict, optional): Additional options for resetting the environment.
        
        Returns:
        tuple or object: Initial observation, and optionally additional information.
        """
        if seed is not None:
            self._np_random, seed = np_random(seed)

        # Reinitialize the deck
        self.deck.init_deck()
        
        # Draw initial hands for dealer and player
        self.dealer = self.deck.draw_hand()
        self.player = self.deck.draw_hand()
        
        # Reset the number of actions taken
        self.actionstaken = 0

        observation = self._get_obs()
        
        if not return_info:
            return observation
        else:
            return observation, {}
