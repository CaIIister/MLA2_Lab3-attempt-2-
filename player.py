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
        self.learning_rate = 0.5
        self.discount_factor = 0.8
        self.epsilon = 0.0
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

    def getAction(self, board, startValue):
        """Main decision method with tactical override"""
        try:
            possible_actions = board.getPossibleActions()
            if len(possible_actions) == 0:
                return 0

            # LAYER 1: TACTICAL DECISIONS (Perfect play)
            tactical_move = self._get_tactical_move(board, startValue, possible_actions)
            if tactical_move is not None:
                return tactical_move

            # LAYER 2: RL STRATEGIC DECISIONS
            strategic_move = self._get_strategic_move(board, startValue, possible_actions)
            return strategic_move

        except Exception as e:
            print(f"getAction error: {e}")
            return np.random.choice(board.getPossibleActions())

    def _get_tactical_move(self, board, startValue, possible_actions):
        """Perfect tactical play: always take wins, always block threats"""
        # Check for immediate wins
        for action in possible_actions:
            if self._test_win_move(board, action, startValue):
                return action

        # Check for immediate threats to block
        for action in possible_actions:
            if self._test_win_move(board, action, -startValue):
                return action

        # No immediate tactical decision needed
        return None

    def _get_strategic_move(self, board, startValue, possible_actions):
        """RL-based strategic decisions"""
        try:
            # Convert to player perspective
            player_board = board.prepareBoardForPlayer(startValue)

            # Extract features
            state = self._extract_strategic_features(player_board, possible_actions)

            # Use Q-learning for strategic choice
            if np.random.random() < self.epsilon:
                return self._get_heuristic_move(player_board, possible_actions)

            # Get Q-values
            q_values = [self.get_q_value(state, action) for action in possible_actions]

            # If no learned preference, use heuristics
            if max(q_values) - min(q_values) < 0.1:
                return self._get_heuristic_move(player_board, possible_actions)

            # Choose best Q-value
            max_q = max(q_values)
            best_actions = [a for a, q in zip(possible_actions, q_values) if abs(q - max_q) < 0.01]
            return np.random.choice(best_actions)

        except Exception as e:
            print(f"Strategic move error: {e}")
            return self._get_heuristic_move(board.prepareBoardForPlayer(startValue), possible_actions)

    def _extract_strategic_features(self, board, possible_actions):
        """Extract simple strategic features - focus on board control patterns"""
        features = []

        # Game phase (early/mid/late)
        filled_cells = np.sum(board != 0)
        if filled_cells < 10:
            phase = 0  # Early
        elif filled_cells < 25:
            phase = 1  # Mid
        else:
            phase = 2  # Late
        features.append(phase)

        # Center control strength
        center_cols = [2, 3, 4]
        my_center = sum(np.sum(board[:, col] == 1) for col in center_cols)
        opp_center = sum(np.sum(board[:, col] == -1) for col in center_cols)
        features.append(min(my_center - opp_center + 5, 10))  # Normalized

        # Column heights (simplified)
        heights = []
        for col in range(7):
            height = np.sum(board[:, col] != 0)
            heights.append(min(height, 6))
        features.extend(heights)

        # Threat potential for each possible action
        threat_scores = []
        for action in range(7):  # Always evaluate all columns
            if action in possible_actions:
                score = self._evaluate_threat_potential(board, action)
            else:
                score = 0
            threat_scores.append(min(score, 5))
        features.extend(threat_scores)

        return tuple(features)

    def _evaluate_threat_potential(self, board, col):
        """Simple threat evaluation"""
        try:
            landing_row = self._get_landing_row(board, col)
            if landing_row is None:
                return 0

            score = 0

            # Check immediate neighborhood for opponent pieces
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = landing_row + dr, col + dc
                    if 0 <= r < 6 and 0 <= c < 7:
                        if board[r, c] == -1:  # Opponent piece
                            score += 2
                        elif board[r, c] == 1:  # My piece
                            score += 1

            # Prefer center columns in early game
            if np.sum(board != 0) < 15:
                center_bonus = max(0, 3 - abs(col - 3))
                score += center_bonus

            return score

        except:
            return 0

    def _get_heuristic_move(self, board, possible_actions):
        """Simple heuristic fallback"""
        scores = []
        for action in possible_actions:
            score = 0

            # Prefer center early
            if np.sum(board != 0) < 12:
                score += max(0, 4 - abs(action - 3))

            # Prefer building near opponent pieces
            landing_row = self._get_landing_row(board, action)
            if landing_row is not None:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        r, c = landing_row + dr, action + dc
                        if 0 <= r < 6 and 0 <= c < 7 and board[r, c] == -1:
                            score += 2

            scores.append(score)

        max_score = max(scores)
        best_actions = [a for a, s in zip(possible_actions, scores) if s >= max_score - 1]
        return np.random.choice(best_actions)

    def _test_win_move(self, board, col, player_value):
        """Test if move results in victory using actual game engine"""
        try:
            # Check if column is playable
            possible_actions = board.getPossibleActions()
            if col not in possible_actions:
                return False

            # Create test board and make move
            test_board = gamerules.Board()
            test_board.board = board.board.copy()
            test_board.components = board.components.copy()
            test_board.components4 = board.components4.copy()
            test_board.componentID = board.componentID
            test_board.component4ID = board.component4ID

            # Make the move using game engine
            test_board.updateBoard(col, player_value)

            # Check victory using game engine
            return test_board.checkVictory(col, player_value)

        except Exception as e:
            print(f"Win test error: {e}")
            return False

    def _get_landing_row(self, board, col):
        """Get landing row for piece in column"""
        try:
            empty_rows = np.where(board[:, col] == 0)[0]
            return np.max(empty_rows) if len(empty_rows) > 0 else None
        except:
            return None

    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state, next_possible_actions):
        """Update Q-value using Q-learning"""
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

    def select_action_for_training(self, board, startValue, possible_actions):
        """Training-time action selection with exploration"""
        try:
            # Still use tactical override in training
            tactical_move = self._get_tactical_move(board, startValue, possible_actions)
            if tactical_move is not None:
                return tactical_move

            # Strategic decision with exploration
            player_board = board.prepareBoardForPlayer(startValue)
            state = self._extract_strategic_features(player_board, possible_actions)

            if np.random.random() < self.epsilon:
                return self._get_heuristic_move(player_board, possible_actions)

            q_values = [self.get_q_value(state, action) for action in possible_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(possible_actions, q_values) if abs(q - max_q) < 0.01]
            return np.random.choice(best_actions)

        except Exception as e:
            print(f"Training action error: {e}")
            return np.random.choice(possible_actions)

    def extract_features_for_training(self, board, player_value):
        """Extract features for training experience"""
        try:
            possible_actions = list(range(7))  # All columns
            valid_actions = [i for i in range(7) if np.sum(board[:, i] != 0) < 6]
            return self._extract_strategic_features(board, valid_actions)
        except:
            return tuple([0] * 17)  # Default feature vector

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