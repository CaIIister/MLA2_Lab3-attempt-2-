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
        """Extract features focused on contour completion and prevention"""
        features = []
        
        # 1. Check for immediate wins/blocks (14 features)
        for col in range(7):
            can_win = self._test_capture_move(board, col, player_value)
            must_block = self._test_capture_move(board, col, -player_value)
            features.extend([1 if can_win else 0, 1 if must_block else 0])

        # 2. Contour progress (7 features)
        for col in range(7):
            contour_score = self._evaluate_contour_progress(board, col, player_value)
            features.append(contour_score)

        # 3. Opponent contour prevention (7 features)
        for col in range(7):
            prevention_score = self._evaluate_contour_prevention(board, col, player_value)
            features.append(prevention_score)

        # 4. Board control (7 features)
        for col in range(7):
            control_score = self._evaluate_board_control(board, col, player_value)
            features.append(control_score)
        
        return tuple(features)

    def _evaluate_contour_progress(self, board, col, player_value):
        """Evaluate progress towards completing a contour"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            # Check for potential contours in all directions
            directions = [(0,1), (1,1), (1,0), (1,-1)]
            max_score = 0
            
            for dr, dc in directions:
                # Look in both directions from the move
                our_pieces = 0
                opponent_pieces = 0
                gaps = 0
                
                for mult in [-1, 1]:  # Check both directions
                    r, c = landing_row + dr * mult, col + dc * mult
                    steps = 0
                    while 0 <= r < 6 and 0 <= c < 7 and steps < 3:
                        if board[r,c] == player_value:
                            our_pieces += 1
                        elif board[r,c] == -player_value:
                            opponent_pieces += 1
                        else:
                            gaps += 1
                        r += dr * mult
                        c += dc * mult
                        steps += 1
                
                # Score based on pieces and gaps
                if opponent_pieces > 0 and gaps <= 2:
                    score = our_pieces * 2 + (2 - gaps)
                    max_score = max(max_score, score)
                    
            return max_score
        except Exception:
            return 0

    def _evaluate_contour_prevention(self, board, col, player_value):
        """Evaluate how well move prevents opponent contours"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            prevention_score = 0
            directions = [(0,1), (1,1), (1,0), (1,-1)]
            
            for dr, dc in directions:
                # Check if this move blocks a potential opponent contour
                opponent_pieces = 0
                our_blocking_pieces = 0
                
                for mult in [-1, 1]:
                    r, c = landing_row + dr * mult, col + dc * mult
                    steps = 0
                    while 0 <= r < 6 and 0 <= c < 7 and steps < 3:
                        if board[r,c] == -player_value:
                            opponent_pieces += 1
                        elif board[r,c] == player_value:
                            our_blocking_pieces += 1
                        r += dr * mult
                        c += dc * mult
                        steps += 1
                
                if opponent_pieces >= 2:
                    prevention_score += opponent_pieces + our_blocking_pieces
                    
            return prevention_score
        except Exception:
            return 0

    def _evaluate_board_control(self, board, col, player_value):
        """Evaluate board control and piece positioning"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return -1

            control_score = 0
            
            # Prefer center columns early game
            if sum(1 for c in range(7) if board[5,c] == 0) >= 5:
                center_dist = abs(col - 3)
                control_score += (3 - center_dist) * 2
                
            # Check surrounding area for good position
            for r in range(max(0, landing_row - 1), min(6, landing_row + 2)):
                for c in range(max(0, col - 1), min(7, col + 2)):
                    if board[r,c] == player_value:
                        # Connected pieces are good
                        control_score += 1
                    elif board[r,c] == -player_value:
                        # Being next to opponent pieces is good (for potential captures)
                        control_score += 2
                        
            return control_score
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
        """Select action with focus on contour completion"""
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
                # During exploration, prefer moves that could form contours
                scores = [state[14 + action] + state[21 + action] for action in possible_actions]
                total_score = sum(scores)
                if total_score > 0:
                    probs = np.array(scores) / total_score
                    return np.random.choice(possible_actions, p=probs)
                else:
                    # If no good contour moves, prefer center
                    weights = [3, 4, 5, 7, 5, 4, 3]
                    probs = [weights[a] for a in possible_actions]
                    probs = np.array(probs) / sum(probs)
                    return np.random.choice(possible_actions, p=probs)

            # Get Q-values for all possible actions
            q_values = [self.get_q_value(state, action) for action in possible_actions]
            
            # If Q-values are similar, use contour heuristics
            if max(q_values) - min(q_values) < 0.1:
                contour_scores = [state[14 + action] + state[21 + action] for action in possible_actions]
                best_score = max(contour_scores)
                best_actions = [action for action, score in zip(possible_actions, contour_scores) 
                              if score == best_score]
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