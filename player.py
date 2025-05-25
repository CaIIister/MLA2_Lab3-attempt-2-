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
        """Extract features focused on quick wins and aggressive play"""
        features = []
        
        # 1. Check for immediate wins/blocks (14 features)
        for col in range(7):
            can_win = self._test_capture_move(board, col, player_value)
            must_block = self._test_capture_move(board, col, -player_value)
            features.extend([1 if can_win else 0, 1 if must_block else 0])

        # 2. Winning patterns (7 features)
        for col in range(7):
            pattern_score = self._evaluate_winning_pattern(board, col, player_value)
            features.append(pattern_score)

        # 3. Opponent piece count (7 features)
        for col in range(7):
            opponent_count = self._count_nearby_opponents(board, col, player_value)
            features.append(opponent_count)

        # 4. Move urgency (7 features)
        for col in range(7):
            urgency = self._evaluate_move_urgency(board, col, player_value)
            features.append(urgency)
        
        return tuple(features)

    def _evaluate_winning_pattern(self, board, col, player_value):
        """Evaluate if move matches known winning patterns"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            score = 0
            # Check L-shaped patterns (most common winning pattern)
            l_patterns = [
                [(0,1), (1,1), (1,0)],  # ┗
                [(0,-1), (1,-1), (1,0)], # ┛
                [(-1,0), (-1,1), (0,1)], # ┏
                [(-1,0), (-1,-1), (0,-1)] # ┓
            ]
            
            for pattern in l_patterns:
                pattern_score = 0
                opponent_inside = False
                
                for dr, dc in pattern:
                    r, c = landing_row + dr, col + dc
                    if 0 <= r < 6 and 0 <= c < 7:
                        if board[r,c] == player_value:
                            pattern_score += 2
                        elif board[r,c] == -player_value:
                            opponent_inside = True
                            break
                        else:
                            pattern_score += 1
                
                if opponent_inside and pattern_score >= 4:
                    score = max(score, pattern_score * 2)
                elif pattern_score >= 4:
                    score = max(score, pattern_score)
                    
            return score
        except Exception:
            return 0

    def _count_nearby_opponents(self, board, col, player_value):
        """Count opponent pieces that could be trapped"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            count = 0
            # Check in a 3x3 area
            for r in range(max(0, landing_row - 1), min(6, landing_row + 2)):
                for c in range(max(0, col - 1), min(7, col + 2)):
                    if board[r,c] == -player_value:
                        # Weight by distance to center
                        center_dist = abs(c - 3)
                        count += (4 - center_dist)  # Higher weight for center columns
                        
            return count
        except Exception:
            return 0

    def _evaluate_move_urgency(self, board, col, player_value):
        """Evaluate how urgent it is to play in this column"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            urgency = 0
            
            # Prefer center columns early
            if sum(1 for c in range(7) if board[5,c] == 0) >= 5:  # Early game
                center_dist = abs(col - 3)
                urgency += (4 - center_dist) * 2
                
            # Check if opponent is close to winning
            for dr, dc in [(0,1), (1,1), (1,0), (1,-1)]:
                opponent_count = 0
                r, c = landing_row, col
                
                # Look in both directions
                for direction in [1, -1]:
                    r2, c2 = r + dr * direction, c + dc * direction
                    while 0 <= r2 < 6 and 0 <= c2 < 7:
                        if board[r2,c2] == -player_value:
                            opponent_count += 1
                        else:
                            break
                        r2 += dr * direction
                        c2 += dc * direction
                        
            if opponent_count >= 2:
                urgency += opponent_count * 2
                
            return urgency
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
        """Select action with aggressive heuristics + Q-learning"""
        try:
            # First, check for immediate wins
            for action in possible_actions:
                if state[action] == 1:  # Can win in this column
                    return action

            # Second, check for blocks
            for action in possible_actions:
                if state[7 + action] == 1:  # Must block in this column
                    return action

            # Use Q-learning with epsilon-greedy
            if np.random.random() < self.epsilon:
                # During exploration, prefer center columns
                weights = [4, 5, 6, 7, 6, 5, 4]  # Center bias
                probs = [weights[a] for a in possible_actions]
                probs = np.array(probs) / sum(probs)
                return np.random.choice(possible_actions, p=probs)

            # Get Q-values for all possible actions
            q_values = [self.get_q_value(state, action) for action in possible_actions]
            
            # If all Q-values are similar, prefer center columns
            if max(q_values) - min(q_values) < 0.1:
                center_actions = [a for a in possible_actions if 2 <= a <= 4]
                if center_actions:
                    return np.random.choice(center_actions)
            
            # Otherwise choose best Q-value
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