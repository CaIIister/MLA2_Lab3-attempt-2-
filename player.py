import numpy as np
import pickle
import os
from gamerules import Player as BasePlayer


class Player(BasePlayer):
    def __init__(self, name, weights_file=None):
        super().__init__(name)
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.0  # No exploration during play
        self.weights_file = weights_file

        # Load pre-trained weights if available
        if weights_file and os.path.exists(weights_file):
            try:
                with open(weights_file, 'rb') as f:
                    self.q_table = pickle.load(f)
                print(f"Loaded Q-table with {len(self.q_table)} states")
            except:
                print("Failed to load weights, starting fresh")
                self.q_table = {}

    def extract_features(self, board, player_value):
        """Extract features for immediate threat detection"""
        features = []

        # Can I win immediately in each column? (7 features)
        for col in range(7):
            features.append(int(self._can_win_in_column(board, col, player_value)))

        # Must I block opponent in each column? (7 features)
        for col in range(7):
            features.append(int(self._can_win_in_column(board, col, -player_value)))

        # Column heights (7 features, capped at 4)
        for col in range(7):
            height = np.sum(board[:, col] != 0)
            features.append(min(height, 4))

        # Center control (3 features for columns 2,3,4)
        for col in [2, 3, 4]:
            my_pieces = np.sum(board[:, col] == player_value)
            features.append(min(my_pieces, 3))

        return tuple(features)

    def _can_win_in_column(self, board, col, player_value):
        """Check if player can win by dropping in this column"""
        if col < 0 or col >= 7:
            return False

        # Find where piece would land
        empty_rows = np.where(board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return False  # Column full

        row = np.max(empty_rows)

        # Simulate the move
        board_copy = board.copy()
        board_copy[row, col] = player_value

        # Check for victory using simplified contour detection
        return self._check_simple_victory(board_copy, row, col, player_value)

    def _check_simple_victory(self, board, row, col, player_value):
        """Simplified victory check - looks for basic enclosures"""
        # Check if we've formed any 3x3 or larger patterns that could enclose opponent
        for size in range(3, 5):  # Check 3x3 and 4x4 patterns
            for dr in range(-size // 2, size // 2 + 1):
                for dc in range(-size // 2, size // 2 + 1):
                    r, c = row + dr, col + dc
                    if 0 <= r < 6 and 0 <= c < 7:
                        if self._check_enclosure_pattern(board, r, c, size, player_value):
                            return True
        return False

    def _check_enclosure_pattern(self, board, center_r, center_c, size, player_value):
        """Check if there's an enclosure pattern around center point"""
        # Simple heuristic: look for L-shapes or partial rectangles
        my_pieces = 0
        opponent_pieces = 0

        for dr in range(-size // 2, size // 2 + 1):
            for dc in range(-size // 2, size // 2 + 1):
                r, c = center_r + dr, center_c + dc
                if 0 <= r < 6 and 0 <= c < 7:
                    if board[r, c] == player_value:
                        my_pieces += 1
                    elif board[r, c] == -player_value:
                        opponent_pieces += 1

        # Heuristic: if we have 3+ pieces and opponent has 1+ in small area
        return my_pieces >= 3 and opponent_pieces >= 1

    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state, next_possible_actions):
        """Update Q-value using Q-learning update rule"""
        if not next_possible_actions:
            max_next_q = 0
        else:
            max_next_q = max([self.get_q_value(next_state, a) for a in next_possible_actions])

        current_q = self.get_q_value(state, action)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def select_action(self, state, possible_actions):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(possible_actions)

        # Get Q-values for all possible actions
        q_values = [self.get_q_value(state, action) for action in possible_actions]
        max_q = max(q_values)

        # Select randomly among actions with maximum Q-value
        best_actions = [action for action, q in zip(possible_actions, q_values) if q == max_q]
        return np.random.choice(best_actions)

    def getAction(self, board, startValue):
        """Main method called by game engine"""
        # Convert board to player perspective
        player_board = board.prepareBoardForPlayer(startValue)

        # Extract features
        state = self.extract_features(player_board, 1)  # Always 1 from player perspective

        # Get possible actions
        possible_actions = board.getPossibleActions()

        if len(possible_actions) == 0:
            return 0  # Shouldn't happen

        # Select action (no exploration during actual play)
        action = self.select_action(state, possible_actions)
        return action

    def newGame(self, new_opponent):
        """Called at start of new game"""
        pass  # No special setup needed

    def save_weights(self, filename=None):
        """Save Q-table to file"""
        if filename is None:
            filename = self.weights_file
        if filename:
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)
            print(f"Saved Q-table with {len(self.q_table)} states to {filename}")