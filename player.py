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
        """Extract features focused on aggressive pattern formation"""
        features = []
        
        # 1. Check for immediate wins/blocks (14 features)
        for col in range(7):
            can_win = self._test_capture_move(board, col, player_value)
            must_block = self._test_capture_move(board, col, -player_value)
            features.extend([1 if can_win else 0, 1 if must_block else 0])

        # 2. Pattern strength (7 features)
        for col in range(7):
            pattern_score = self._evaluate_pattern_strength(board, col, player_value)
            features.append(pattern_score)

        # 3. Piece density (7 features)
        for col in range(7):
            density_score = self._evaluate_piece_density(board, col, player_value)
            features.append(density_score)

        # 4. Position value (7 features)
        for col in range(7):
            position_score = self._evaluate_position_value(board, col, player_value)
            features.append(position_score)
        
        return tuple(features)

    def _evaluate_pattern_strength(self, board, col, player_value):
        """Evaluate strength of potential winning patterns"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            # Key winning patterns
            patterns = [
                [(0,1), (1,0)],      # L corner
                [(0,-1), (1,0)],     # L corner
                [(1,1), (1,-1)],     # V shape
                [(0,1), (0,2)],      # Horizontal
                [(1,0), (2,0)]       # Vertical
            ]
            
            max_score = 0
            for pattern in patterns:
                score = 0
                opponent_trapped = False
                our_pieces = 0
                
                # Check if we have pieces forming the pattern
                for dr, dc in pattern:
                    r, c = landing_row + dr, col + dc
                    if 0 <= r < 6 and 0 <= c < 7:
                        if board[r,c] == player_value:
                            our_pieces += 1
                            score += 2
                        elif board[r,c] == 0:
                            score += 1
                
                # Check if opponent piece could be trapped
                for r in range(max(0, landing_row-1), min(6, landing_row+2)):
                    for c in range(max(0, col-1), min(7, col+2)):
                        if board[r,c] == -player_value:
                            opponent_trapped = True
                            break
                            
                if opponent_trapped and our_pieces >= 1:
                    score *= 2
                    
                max_score = max(max_score, score)
                
            return max_score
        except Exception:
            return 0

    def _evaluate_piece_density(self, board, col, player_value):
        """Evaluate piece density and potential for captures"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            density = 0
            opponent_pieces = 0
            our_pieces = 0
            
            # Check 5x5 area centered on move
            for r in range(max(0, landing_row-2), min(6, landing_row+3)):
                for c in range(max(0, col-2), min(7, col+3)):
                    if board[r,c] == player_value:
                        our_pieces += 1
                    elif board[r,c] == -player_value:
                        opponent_pieces += 1
                        
            # High density of our pieces with some opponent pieces is good
            if opponent_pieces > 0:
                density = our_pieces * 2
                
            # Extra points for having more pieces than opponent
            if our_pieces > opponent_pieces:
                density += 2
                
            return density
        except Exception:
            return 0

    def _evaluate_position_value(self, board, col, player_value):
        """Evaluate strategic value of position"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            value = 0
            
            # Strong preference for center columns early game
            if sum(1 for c in range(7) if board[5,c] == 0) >= 5:
                center_dist = abs(col - 3)
                value += (4 - center_dist) * 3
                
            # Value moves that build on existing pieces
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                r, c = landing_row + dr, col + dc
                if 0 <= r < 6 and 0 <= c < 7:
                    if board[r,c] == player_value:
                        value += 2
                    elif board[r,c] == -player_value:
                        value += 1  # Being next to opponent pieces can be good
                        
            return value
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
        """Select action with aggressive pattern formation"""
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
                # During exploration, prefer strong patterns and center
                pattern_scores = [state[14 + action] * 2 + state[28 + action] for action in possible_actions]
                total_score = sum(pattern_scores)
                if total_score > 0:
                    probs = np.array(pattern_scores) / total_score
                    return np.random.choice(possible_actions, p=probs)
                else:
                    # If no good patterns, prefer center
                    weights = [4, 5, 6, 8, 6, 5, 4]
                    probs = [weights[a] for a in possible_actions]
                    probs = np.array(probs) / sum(probs)
                    return np.random.choice(possible_actions, p=probs)

            # Get Q-values for all possible actions
            q_values = [self.get_q_value(state, action) for action in possible_actions]
            
            # If Q-values are similar, use pattern strength
            if max(q_values) - min(q_values) < 0.1:
                pattern_scores = [state[14 + action] * 2 + state[28 + action] for action in possible_actions]
                best_score = max(pattern_scores)
                best_actions = [action for action, score in zip(possible_actions, pattern_scores) 
                              if score >= best_score - 1]  # Allow small variations
                return np.random.choice(best_actions)
            
            # Otherwise choose best Q-value
            max_q = max(q_values)
            best_actions = [action for action, q in zip(possible_actions, q_values) 
                           if abs(q - max_q) < 0.001]
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