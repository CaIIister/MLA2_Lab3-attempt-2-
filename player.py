import numpy as np
import pickle
import os
import copy
from gamerules import Player as BasePlayer
import gamerules


class Player(BasePlayer):
    def __init__(self, name, weights_file=None):
        super().__init__(name)
        self.q_table = {}
        self.learning_rate = 0.3
        self.discount_factor = 0.9
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
        """Extract features focused on contour formation and encirclement"""
        features = []
        
        # 1. Check for immediate wins/blocks (14 features)
        for col in range(7):
            can_win = self._test_capture_move(board, col, player_value)
            must_block = self._test_capture_move(board, col, -player_value)
            features.extend([1 if can_win else 0, 1 if must_block else 0])

        # 2. Partial contour formation (7 features)
        for col in range(7):
            contour_score = self._evaluate_contour_formation(board, col, player_value)
            features.append(contour_score)

        # 3. Opponent piece vulnerability (7 features)
        for col in range(7):
            vulnerability = self._evaluate_opponent_vulnerability(board, col, player_value)
            features.append(vulnerability)

        # 4. Escape routes (7 features)
        for col in range(7):
            escape_score = self._evaluate_escape_routes(board, col, player_value)
            features.append(escape_score)
        
        return tuple(features)

    def _evaluate_contour_formation(self, board, col, player_value):
        """Evaluate how well a move contributes to forming contours"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            # Count our pieces that would connect in all 8 directions
            directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
            connections = 0
            
            for dr, dc in directions:
                r, c = landing_row + dr, col + dc
                if 0 <= r < 6 and 0 <= c < 7 and board[r,c] == player_value:
                    connections += 1
                    
                    # Check for potential closure (extra points for almost complete contours)
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < 6 and 0 <= c2 < 7 and board[r2,c2] == player_value:
                        connections += 1
                        
            return connections
        except Exception:
            return 0

    def _evaluate_opponent_vulnerability(self, board, col, player_value):
        """Evaluate if move helps trap opponent pieces"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            # Count opponent pieces that would be partially surrounded
            directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
            vulnerable_pieces = 0
            
            for dr, dc in directions:
                r, c = landing_row + dr, col + dc
                if 0 <= r < 6 and 0 <= c < 7 and board[r,c] == -player_value:
                    # Check if there's a potential to surround this piece
                    opposite_r = landing_row - dr
                    opposite_c = col - dc
                    if 0 <= opposite_r < 6 and 0 <= opposite_c < 7:
                        if board[opposite_r, opposite_c] == player_value:
                            vulnerable_pieces += 2  # Higher score for pieces we can trap
                        elif board[opposite_r, opposite_c] == 0:
                            vulnerable_pieces += 1  # Lower score for potential future traps
                            
            return vulnerable_pieces
        except Exception:
            return 0

    def _evaluate_escape_routes(self, board, col, player_value):
        """Evaluate if move maintains escape routes for our pieces"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            # Count open paths to edges (to avoid being trapped)
            escape_routes = 0
            
            # Check if near board edge (safer)
            if col == 0 or col == 6:
                escape_routes += 2
            if landing_row == 0:
                escape_routes += 2
            
            # Count open adjacent spaces
            directions = [(0,1), (1,0), (0,-1), (-1,0)]
            for dr, dc in directions:
                r, c = landing_row + dr, col + dc
                if 0 <= r < 6 and 0 <= c < 7 and board[r,c] == 0:
                    escape_routes += 1
                
            return escape_routes
        except Exception:
            return 0

    def _get_landing_row(self, board, col):
        """Get the row where a piece would land in given column"""
        try:
            if np.sum(board[:, col] != 0) >= 6:
                return None
            empty_rows = np.where(board[:, col] == 0)[0]
            return np.max(empty_rows) if len(empty_rows) > 0 else None
        except Exception:
            return None

    def _test_capture_move(self, board, col, player_value):
        """Test if dropping in column results in capturing opponent's piece"""
        try:
            if np.sum(board[:, col] != 0) >= 6:
                return False

            # Find landing row
            empty_rows = np.where(board[:, col] == 0)[0]
            if len(empty_rows) == 0:
                return False

            landing_row = np.max(empty_rows)

            # Create test board
            test_board_obj = gamerules.Board()
            test_board_obj.board = board.copy()
            test_board_obj.updateBoard(col, player_value)
            return test_board_obj.checkVictory(col, player_value)

        except Exception:
            return False

    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state, next_possible_actions):
        """Update Q-value using Q-learning update rule"""
        try:
            if len(next_possible_actions) == 0:
                max_next_q = 0
            else:
                max_next_q = max([self.get_q_value(next_state, a) for a in next_possible_actions])

            current_q = self.get_q_value(state, action)
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            self.q_table[(state, action)] = new_q
        except Exception as e:
            print(f"Q-update error: {e}")

    def select_action(self, state, possible_actions):
        """Select action with basic heuristics + Q-learning"""
        try:
            # First, check for immediate wins (always take these)
            for action in possible_actions:
                if state[action] == 1:  # Can win in this column
                    return action

            # Second, check for blocks (always block opponent wins)
            for action in possible_actions:
                if state[7 + action] == 1:  # Must block in this column
                    return action

            # Use Q-learning for other decisions
            if np.random.random() < self.epsilon:
                return np.random.choice(possible_actions)

            # Get Q-values for all possible actions
            q_values = [self.get_q_value(state, action) for action in possible_actions]

            # If all Q-values are zero, prefer center
            if all(abs(q) < 0.001 for q in q_values):
                center_actions = [a for a in possible_actions if a in [2, 3, 4]]
                if center_actions:
                    return np.random.choice(center_actions)
                return np.random.choice(possible_actions)

            max_q = max(q_values)
            best_actions = [action for action, q in zip(possible_actions, q_values) if abs(q - max_q) < 0.001]
            return np.random.choice(best_actions)

        except Exception as e:
            print(f"Action selection error: {e}")
            return np.random.choice(possible_actions)

    def getAction(self, board, startValue):
        """Main method called by game engine"""
        try:
            # Convert board to player perspective
            player_board = board.prepareBoardForPlayer(startValue)

            # Extract features
            state = self.extract_features(player_board, 1)

            # Get possible actions
            possible_actions = board.getPossibleActions()

            if len(possible_actions) == 0:
                return 0

            # Select action
            action = self.select_action(state, possible_actions)
            return int(action)

        except Exception as e:
            print(f"getAction error: {e}")
            # Fallback to random
            possible_actions = board.getPossibleActions()
            if len(possible_actions) > 0:
                return np.random.choice(possible_actions)
            return 0

    def newGame(self, new_opponent):
        """Called at start of new game"""
        pass

    def save_weights(self, filename=None):
        """Save Q-table to file"""
        if filename is None:
            filename = self.weights_file
        if filename:
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(self.q_table, f)
                print(f"Saved Q-table with {len(self.q_table)} states to {filename}")
            except Exception as e:
                print(f"Save error: {e}")