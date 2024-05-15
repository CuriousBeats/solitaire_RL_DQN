import random
import numpy as np
import math
# card.suit  # 0=spades, 1=hearts, 2=clubs, 3=diamonds
# card.value # 1 = Ace, 2 = 2, ... 10 = 10, 11 = Jack, 12 = Queen, 13 = King



class Card:
    def __init__(self, suit, value, face_up=False):
        self.suit = suit
        self.value = value
        self.face_up = face_up

    def flip(self):
        self.face_up = not self.face_up

    def __str__(self):
        suits = ["Spades", "Hearts", "Clubs", "Diamonds"]
        return str(self.value) + " of " + suits[self.suit]

class Deck:
    def __init__(self):
        self.cards = self.create_deck()
        self.shuffle()
        self.table_cards = self.cards[0:28]
        self.cards = self.cards[28:]
        self.grouped_cards = self.group_cards()
        self.faceup = []
        self.facedown = self.cards
        self.draw()
        self.n_facedown = self.get_face_down()
        

    def shuffle(self):
        random.shuffle(self.cards)

    def create_deck(self):
        suits = [0,1,2,3]
        values = list(range(1,14))
        cards = []
        for suit in suits:
            for value in values:
                cards.append(Card(suit, value))
        return cards

    def reorder(self):
        # Move the cards from faceup to facedown, reversing the order
        while self.faceup:
            self.facedown.extend(self.faceup.pop()[::-1])

    def draw(self):
        if len(self.facedown) == 0:
            self.reorder()

        if len(self.facedown) > 3:
            self.faceup.append(self.facedown[0:3])
            self.facedown = self.facedown[3:]
        else:
            self.faceup.append(self.facedown)
            self.facedown = []

    def deal_card(self):
        return self.faceup[-1].pop()

    def group_cards(self):
        groups = []
        for i in range(len(self.cards)):
            if i %3 == 0:
                if i+3 < len(self.cards):
                    groups.append(self.cards[i:i+3])
                else:
                    groups.append(self.cards[i:])
        return groups
    
    def get_face_down(self):
            face_down = 0
            for column in self.table:
                for card in column:
                    if not card.face_up:
                        face_down += 1
            return face_down
class SolitaireEnvironment:
    def __init__(self):
        self.deck = Deck()
        self.table, self.top = self.set_table()
        self.loopedCount = 0
        self.previousFaceDown = 0

    def move_possible(self, fromCard, destCard, top=False):
        if not fromCard.face_up:
            return False

        elif isinstance(destCard, int):
            if destCard == -1:
                if fromCard.value == 13:
                    return True
            elif fromCard.value == 1 and fromCard.suit == destCard:
                return True

        elif not destCard.face_up or fromCard.value != destCard.value + 1:
            return False

        elif top:
            if fromCard.suit == destCard.suit:
                return True

        elif fromCard.suit % 2 ^ destCard.suit % 2:
            return True

        return False

    def move_card(self, card, dest, top, fromColumn, toColumn):
        #move a card from one column to another
        #if the card is the top card in the column, flip the new top card
        #if the card is the only card in the column, the column is now empty
        #check if the move is possible before moving the card

        #redundant but safe
        if not self.move_possible(card, dest, top):
            return False

        if fromColumn > len(self.table)-1:
            fromColumn -= len(self.table)
            if card == self.top[fromColumn][-1]:
                self.table[toColumn].append(self.top[fromColumn][-1])
                self.top[fromColumn] = self.top[fromColumn][:-1]
                if len(self.top[fromColumn]) > 0:
                    if not self.top[fromColumn][-1].face_up:
                        self.top[fromColumn][-1].flip()
                return True

        for i in range(len(self.table[fromColumn])):
            if self.table[fromColumn][i] == card:
                #move the card and all cards below it to the new column
                cards_to_move = self.table[fromColumn][i:]
                self.table[fromColumn] = self.table[fromColumn][:i]
                if toColumn > len(self.table)-1:
                    self.top[toColumn - len(self.table)].extend(cards_to_move)
                else:
                    self.table[toColumn] += cards_to_move
                if len(self.table[fromColumn]) > 0:
                    if not self.table[fromColumn][-1].face_up:
                        self.table[fromColumn][-1].flip()
                return True
        return False #if the move was not possible

    def set_table(self):
        table = []
        cards = self.deck.table_cards
        top = []
        for i in range(4):
            top.append([])
        for i in range(7):
            table.append([])
        for i in range(7):
            for j in range(i+1):
                card = cards.pop()
                if j == i:
                    card.flip()
                table[i].append(card)
        return table, top
    def encode_action(self, possible_action):
        card, dest, top, fromColumn, toColumn = possible_action
        #encode the action as an integer. We'll store it as a string right now to encode easier
        action = ""
        val = str(card.value)
        if len(val) == 1:
            action += "0"
        action += val
        action += str(card.suit % 2)
        action += str(fromColumn)
        action += str(toColumn)
        action += str(top)
        return int(action)
    def decode_action(self, action):
        if action == 0:
            return "Draw"
        else:
            action = str(action)
            #decode the action from an integer to a tuple
            #first two digits are the card value
            card = Card(int(action[0:2]), int(action[2]))
            #next digit is the suit
            card.suit = int(action[3])
            #next digit is the from column
            fromColumn = int(action[4])
            #next digit is the to column
            toColumn = int(action[5])
            #next digit is whether the card is going to the top
            top = bool(int(action[6]))
            #if action is 1, fromcolumn is first column, tocolumn is second column.
            #if action is 2, fromcolumn is first column, tocolumn is third column etc.
            #use the action space to determine which columns to move from and to
            # fromColumn = math.floor(action / (len(self.table)+len(self.top)))
            # toColumn = (action % (len(self.table)+len(self.top)))-1
            if fromColumn > len(self.table)-1:
                if len(self.top[fromColumn - len(self.table)]) == 0:
                    return False
                else:
                    card = self.top[fromColumn - len(self.table)][-1]
            elif len(self.table[fromColumn]) == 0:
                return False
            else:
                card = self.table[fromColumn][-1]
            if toColumn > len(self.table)-1:
                if len(self.top[toColumn - len(self.table)]) == 0:
                    dest = toColumn - len(self.table)
                else:
                    dest = (self.top[toColumn - len(self.table)][-1].value, toColumn - len(self.table))
                top = True
            else:
                if len(self.table[toColumn]) == 0:
                    dest = -1
                else:
                    dest = self.table[toColumn][-1]
                top = False
            if not self.move_possible(card, dest, top):
                return False
            return (card, dest, top, fromColumn, toColumn)
        
    def step(self, action):
        action = self.decode_action(int(action))
        # Example action decoding (you need to define this based on your game's rules)
        if action == "Draw":
            self.deck.draw()
        elif action == False:
            return -1, False, {"error": "Invalid move"}
        else:
            card, dest, top, fromColumn, toColumn = action
            move = self.move_card(card, dest, top, fromColumn, toColumn)
            if not move:
                return -1, False, {"error": "Invalid move"}

        over, result = self.check_game_over()
        reward = self.calculate_reward(result)

        return reward, over, {}

    def calculate_reward(self, result):
        # Define your reward function here
        if result == "Win":
            return 1
        elif result == "Lose":
            return -1
        else:
            #scale reward by number of remaining cards face down on table
            return .5 - (self.deck.face_down / 28)

    def check_game_over(self):
        if self.deck.face_down == self.previousFaceDown:
            self.loopedCount += 1
        else:
            self.loopedCount = 0
        self.previousFaceDown = self.deck.face_down

        # Check for winning condition
        if all(len(pile) == 13 for pile in self.top):
            return True, "Win"

        # Check for losing condition: no more legal moves
        # This can be complex as it requires checking all possible moves in the current game state.
        # You may need to implement additional helper methods to check for possible moves.
        if not self.any_legal_moves() or self.loopedCount > 1000:
            return True, "Lose"

        return False, "Continue"

    def any_legal_moves(self):
        # Check for moves within the tableau
        for i in range(len(self.table)):
            for j in range(len(self.table)):
                if i != j and len(self.table[i]) > 0 and len(self.table[j]) > 0:
                    if self.move_possible(self.table[i][-1], self.table[j][-1]):
                        return True
        # Check for moves from deck to tableau or foundation piles
        for i in range(len(self.table)):
            if len(self.table[i]) > 0:
                if self.move_possible(self.deck.faceup[-1][-1], self.table[i][-1]):
                    return True
        return False
    
    def get_possible_actions(self):
        #first, check if there are any legal moves
        if not self.any_legal_moves():
            return False
        #if there are legal moves, get a list of all possible moves
        possible_actions = []
        
        for fromColumn in range(len(self.table)):
            if len(self.table[fromColumn]) > 0:
                for card in range(len(self.table[fromColumn])):
                    for toColumn in range(len(self.table)):
                        if fromColumn != toColumn:
                            if len(self.table[toColumn]) > 0:
                                dest = self.table[toColumn][-1]
                            else:
                                dest = -1
                            if self.move_possible(card, dest):
                                possible_actions.append((card, dest, False, fromColumn, toColumn))
                for toColumn in range(len(self.top)):
                    if len(self.top[toColumn]) > 0:
                        dest = self.top[toColumn][-1]
                    else:
                        dest = toColumn
                    if self.move_possible(self.table[fromColumn][-1], dest, True):
                        possible_actions.append((self.table[fromColumn][-1], dest, True, fromColumn, toColumn + len(self.table)))
    
        for fromColumn in range(len(self.top)):
            if len(self.top[fromColumn]) > 0:
                for toColumn in range(len(self.table)):
                    if len(self.table[toColumn]) > 0:
                        dest = self.table[toColumn][-1]
                    else:
                        dest = -1
                    if self.move_possible(self.top[fromColumn][-1], dest):
                        possible_actions.append((self.top[fromColumn][-1], dest, False, fromColumn + len(self.table), toColumn))
        #TODO: also iterate through the deck

    def reset(self):
        self.__init__()