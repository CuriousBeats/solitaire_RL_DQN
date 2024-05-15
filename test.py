import unittest
from network import DQN, get_state_representation
from solitaire_environment import SolitaireEnvironment  
import numpy as np

class TestDQNModel(unittest.TestCase):
    def test_initialization(self):
        state_size = 26624  # Replace with your actual state size
        action_size = 110  # Replace with your actual action size
        model = DQN(state_size, action_size)
        self.assertEqual(model.fc1.in_features, state_size)
        self.assertEqual(model.fc3.out_features, action_size)

class TestStateRepresentation(unittest.TestCase):
    def test_state_shape(self):
        env = SolitaireEnvironment()
        state = get_state_representation(env)
        self.assertEqual(state.shape, (26624,))  # Replace with your actual state shape

class TestSolitaireEnvironment(unittest.TestCase):
    def test_draw_cards(self):
        env = SolitaireEnvironment()
        initial_deck_count = len(env.deck.cards)
        env.deck.draw()
        self.assertNotEqual(len(env.deck.cards), initial_deck_count)

    def test_move_possible(self):
        env = SolitaireEnvironment()
        card1 = env.table[0][-1]  # Assume this is a valid card from a tableau column
        card2 = env.table[1][-1]  # Assume this is another valid card from a different tableau column
        is_move_possible = env.move_possible(card1, card2)
        self.assertIsInstance(is_move_possible, bool)

if __name__ == '__main__':

    unittest.main()
