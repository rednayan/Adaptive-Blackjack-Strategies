import numpy as np
from utils.common import can_double_down, can_insurance, cmp, draw_card, draw_hand, is_blackjack, is_bust, is_natural, score, sum_hand, usable_ace
from utils.seeding import np_random

class Blackjack_Env:

    def __init__(self, natural=False, sab=False):
        # Initialize action space and observation space
        self.action_space = range(4)
        usable_ace = range(2)
        double_down = range(2)
        insurance = range(2)
        dealer_value = range(11)
        players_hand = range(31)

        # Define the observation space
        self.observation_space = (players_hand, dealer_value, usable_ace, double_down, insurance)
        self.natural = natural
        self.sab = sab
        self.actionstaken = 0

    def np_random(self) -> np.random.Generator:
        # Initialize the random number generator if not already initialized
        if self._np_random is None:
            self._np_random, seed = np_random()
        return self._np_random

    def step(self, action):
        reward = 0

        # Action 0: Player stands
        if action == 0:  
            terminated = True

            # Dealer hits until their hand value is at least 17
            while sum_hand(self.dealer) < 17:  # 16
                self.dealer.append(draw_card(self.np_random))

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
            self.player.append(draw_card(self.np_random))

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
            self.player.append(draw_card(self.np_random))

            # Check if player busts
            if is_bust(self.player):
                terminated = True
                reward = -2.0
            else:
                terminated = False

                # Dealer hits until their hand value is at least 17
                while sum_hand(self.dealer) < 17:  # 16
                    self.dealer.append(draw_card(self.np_random))

                # Compare scores and double the reward
                reward = 2.0 * cmp(score(self.player), score(self.dealer))
            self.actionstaken += 1  

        # Action 3: Player takes insurance
        elif action == 3:  
            if self.dealer[0] == 1:  # Check if dealer's visible card is an Ace
                self.dealer.append(draw_card(self.np_random))  

                # Check if dealer has blackjack
                if is_blackjack(self.dealer):
                    terminated = True
                    reward = 0.5  
                else:
                    # Dealer hits until their hand value is at least 17
                    while sum_hand(self.dealer) < 17:
                        self.dealer.append(draw_card(self.np_random))

                    # Compare scores to determine reward
                    reward += cmp(score(self.player), score(self.dealer))

                    # Deduct half a point if player loses or draws
                    if reward <= 0:
                        reward -= 0.5
                        
                    terminated = True

            else:
                # Dealer hits until their hand value is at least 17
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card(self.np_random))

                # Compare scores to determine reward
                reward += cmp(score(self.player), score(self.dealer))

                # Deduct half a point if player loses or draws
                if reward <= 0:
                    reward -= 0.5
                        
                terminated = True
                
            self.actionstaken += 1 
        
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        # Return the current state of the game
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), can_double_down(self.player, self.actionstaken), can_insurance(self.dealer))

    def reset(self, seed=None, return_info=False):
        # Reset the game to the initial state
        if seed is not None:
            self._np_random, seed = np_random(seed)

        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        self.actionstaken = 0

        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}
