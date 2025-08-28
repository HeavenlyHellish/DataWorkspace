cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
import random

def deal_card(amount):
    """Returns a random card from the deck."""
    return random.sample(cards, amount)

u_hand = []
c_hand = []

u_hand += deal_card(2)
c_hand += deal_card(2)

def calculate_score(hand):
    """Take a list of cards and return the score calculated from the cards"""
    if sum(hand) == 21 and len(hand) == 2:
        return 0
    if 11 in hand and sum(hand) > 21:
        hand.remove(11)
        hand.append(1)
    return sum(hand)

def compare(u_score, c_score):
    if u_score == c_score:
        return "It's a Draw!"
    elif c_score == 0:
        return "Lose, opponent has Blackjack!"
    elif u_score == 0:
        return "Win with a Blackjack!"
    elif u_score > 21:
        return "You went over. You lose!"
    elif c_score > 21:
        return "Opponent went over. You win!"
    elif u_score > c_score:
        return "You win!"
    else:
        return "You lose!"

game_over = False
while not game_over:
    u_score = calculate_score(u_hand)
    c_score = calculate_score(c_hand)
    print(f"Your cards: {u_hand}, current score: {u_score}")
    print(f"Computer's first card: {c_hand[0]}")

    if u_score == 0 or c_score == 0 or u_score > 21:
        game_over = True
    else:
        user_should_deal = input("Type 'y' to get another card, type 'n' to pass: ")
        if user_should_deal == "y":
            u_hand += deal_card(1)
        else:
            while c_score != 0 and c_score < 17:
                c_hand += deal_card(1)
                c_score = calculate_score(c_hand)

