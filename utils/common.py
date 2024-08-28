import random

# 1 = Ace, 2-10 = Number cards, Jack / Queen / King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def cmp(score_player, score_dealer):
    """
    Compare the player's score with the dealer's score.

    Args:
        score_player: The player's score.
        score_dealer: The dealer's score.

    Returns:
        float: Positive if player wins, negative if dealer wins, zero if draw.
    """
    return float(score_player > score_dealer) - float(score_player < score_dealer)

def draw_card(np_random):
    """
    Draw a random card from the deck.

    Args:
        np_random: Random state (not used in this implementation).

    Returns:
        int: A randomly selected card from the deck.
    """
    return int(random.choice(deck))

def draw_hand(np_random):
    """
    Draw a starting hand consisting of two random cards.

    Args:
        np_random: Random state (not used in this implementation).

    Returns:
        list: A list of two randomly selected cards.
    """
    return [draw_card(np_random), draw_card(np_random)]

def usable_ace(hand):
    """
    Check if the hand contains a usable ace.

    Args:
        hand: List of cards in the hand.

    Returns:
        bool: True if the hand contains a usable ace, otherwise False.
    """
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand):
    """
    Calculate the sum of the hand.

    Args:
        hand: List of cards in the hand.

    Returns:
        int: The sum of the hand, treating aces as 11 where appropriate.
    """
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)

def is_bust(hand):
    """
    Check if the hand is a bust (i.e., its sum exceeds 21).

    Args:
        hand: List of cards in the hand.

    Returns:
        bool: True if the hand is a bust, otherwise False.
    """
    return sum_hand(hand) > 21

def score(hand):
    """
    Calculate the score of the hand.

    Args:
        hand: List of cards in the hand.

    Returns:
        int: The score of the hand (0 if bust).
    """
    return 0 if is_bust(hand) else sum_hand(hand)

def is_natural(hand):
    """
    Check if the hand is a natural blackjack (an Ace and a 10-point card).

    Args:
        hand: List of cards in the hand.

    Returns:
        bool: True if the hand is a natural blackjack, otherwise False.
    """
    return sorted(hand) == [1, 10]

def can_double_down(hand, actionstaken):
    """
    Check if the player can double down (allowed only on the first action with two cards).

    Args:
        hand: List of cards in the hand.
        actionstaken: Number of actions taken so far.

    Returns:
        bool: True if the player can double down, otherwise False.
    """
    return len(hand) == 2 and actionstaken == 0

def can_split(hand, actionstaken):
    """
    Check if the player can split their hand (allowed if the first two cards are the same).

    Args:
        hand: List of cards in the hand.
        actionstaken: Number of actions taken so far.

    Returns:
        bool: True if the player can split, otherwise False.
    """
    return can_double_down(hand, actionstaken) and hand[0] == hand[1]

def is_blackjack(hand):
    """
    Check if the hand is a blackjack.

    Args:
        hand: List of cards in the hand.

    Returns:
        bool: True if the hand is a blackjack, otherwise False.
    """
    return sum(hand) == 11 and usable_ace(hand)

def can_insurance(dealer_hand):
    """
    Check if insurance can be offered based on the dealer's hand (if dealer shows an Ace).

    Args:
        dealer_hand: List of cards in the dealer's hand.

    Returns:
        bool: True if insurance can be offered, otherwise False.
    """
    return dealer_hand[0] == 1
