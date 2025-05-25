#!/usr/bin/env python3
"""
Tournament-ready Enhanced Player for Contour Formation Game
Optimized for competition submission with CUDA support
Compatible with main.py tournament interface
"""

import numpy as np
import random
import gamerules
import os

# CUDA Support Detection
try:
    import torch
    import torch.nn as nn

    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE = "cpu"


class TournamentDQN(nn.Module):
    """Lightweight tournament-ready DQN (inference only)"""

    def __init__(self, input_size=200, hidden_sizes=[512, 256, 128], output_size=7):
        super(TournamentDQN, self).__init__()

        # Build network architecture
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation on output layer
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))

        self.network = nn.Sequential(*layers)
        self.to(DEVICE)

    def forward(self, x):
        return self.network(x)

    def predict(self, state):
        """Fast inference prediction"""
        self.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(DEVICE)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            return self.forward(state).cpu().numpy()

    def load_weights(self, filepath):
        """Load trained weights"""
        try:
            if CUDA_AVAILABLE:
                checkpoint = torch.load(filepath, map_location=DEVICE)
                if 'state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['state_dict'])
                else:
                    self.load_state_dict(checkpoint)
            else:
                # CPU fallback
                checkpoint = torch.load(filepath, map_location='cpu')
                if 'state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['state_dict'])
                else:
                    self.load_state_dict(checkpoint)
            return True
        except Exception as e:
            print(f"⚠️ Failed to load PyTorch weights: {e}")
            return False


class FallbackNetwork:
    """Fallback NumPy implementation if PyTorch fails"""

    def __init__(self):
        # Simple fallback network for emergency use
        self.weights = {
            'layer1': np.random.randn(200, 128) * 0.1,
            'bias1': np.zeros(128),
            'layer2': np.random.randn(128, 64) * 0.1,
            'bias2': np.zeros(64),
            'layer3': np.random.randn(64, 7) * 0.1,
            'bias3': np.zeros(7)
        }

    def predict(self, state):
        """Simple forward pass"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        # Layer 1
        x = np.dot(state, self.weights['layer1']) + self.weights['bias1']
        x = np.maximum(0, x)  # ReLU

        # Layer 2
        x = np.dot(x, self.weights['layer2']) + self.weights['bias2']
        x = np.maximum(0, x)  # ReLU

        # Layer 3 (output)
        x = np.dot(x, self.weights['layer3']) + self.weights['bias3']

        return x

    def load_weights(self, filepath):
        """Cannot load PyTorch weights in NumPy fallback"""
        return False


class Player(gamerules.Player):
    """
    Enhanced Tournament Player for Contour Formation Game

    Student: Taras Demchyna
    Architecture: Enhanced DQN with contour-aware features
    Training: 5000+ episodes with CUDA optimization
    """

    def __init__(self, name="Taras Demchyna", weights_file="enhanced_weights.pth"):
        super().__init__(name)
        self.student_name = "Taras Demchyna"  # Required for tournament

        # Initialize network
        try:
            if CUDA_AVAILABLE:
                self.q_network = TournamentDQN()
                weights_loaded = self.q_network.load_weights(weights_file)
            else:
                raise ImportError("PyTorch not available")

        except (ImportError, Exception) as e:
            print(f"⚠️ Using fallback network: {e}")
            self.q_network = FallbackNetwork()
            weights_loaded = False

        # Verify weights loaded successfully
        if not weights_loaded:
            print("⚠️ Using random weights - performance will be poor")

        # Heuristic weights for smart fallbacks
        self.center_preferences = [3, 2, 4, 1, 5, 0, 6]

    def getName(self):
        """Required tournament method - returns student name"""
        return self.student_name

    def newGame(self, new_opponent):
        """Required tournament method - called at game start"""
        # No state to reset for stateless player
        pass

    def getAction(self, board, startValue):
        """
        Main decision method - must return column (0-6) in <5 seconds
        Enhanced with contour-aware analysis and smart heuristics
        """
        possibleActions = self.getPossibleActions(board.board)

        if len(possibleActions) == 0:
            return 0  # Emergency fallback

        try:
            # === PRIORITY 1: IMMEDIATE WIN ===
            for action in possibleActions:
                if self._can_win_immediately(board, action, startValue):
                    return action

            # === PRIORITY 2: BLOCK OPPONENT WIN ===
            for action in possibleActions:
                if self._can_win_immediately(board, action, -startValue):
                    return action

            # === PRIORITY 3: DQN STRATEGIC DECISION ===
            dqn_action = self._get_dqn_action(board, startValue, possibleActions)
            if dqn_action is not None:
                return dqn_action

            # === PRIORITY 4: HEURISTIC FALLBACK ===
            return self._get_heuristic_action(board, startValue, possibleActions)

        except Exception as e:
            # Emergency fallback - prefer center columns
            for col in self.center_preferences:
                if col in possibleActions:
                    return col
            return possibleActions[0]

    def _can_win_immediately(self, board, action, player_value):
        """Fast check for immediate win condition"""
        if action not in self.getPossibleActions(board.board):
            return False

        try:
            # Create temporary board to test move
            temp_board = gamerules.Board()
            temp_board.board = board.board.copy()
            temp_board.components = board.components.copy()
            temp_board.components4 = board.components4.copy()
            temp_board.componentID = board.componentID
            temp_board.component4ID = board.component4ID

            # Simulate move
            temp_board.updateBoard(action, player_value)

            # Check if this creates a winning contour
            return temp_board.checkVictory(action, player_value)

        except Exception:
            return False

    def _get_dqn_action(self, board, startValue, possibleActions):
        """Get action from DQN with error handling"""
        try:
            # Encode current board state
            state = self._encode_state_contour_aware(board, startValue)

            # Get Q-values from network
            q_values = self.q_network.predict(state)[0]

            # Mask invalid actions
            q_values_masked = q_values.copy()
            for i in range(7):
                if i not in possibleActions:
                    q_values_masked[i] = float('-inf')

            # Select best valid action
            action = np.argmax(q_values_masked)

            # Safety check
            if action in possibleActions:
                return int(action)
            else:
                return None

        except Exception:
            return None

    def _get_heuristic_action(self, board, startValue, possibleActions):
        """Smart heuristic fallback for when DQN fails"""

        # Score each possible action
        action_scores = []

        for action in possibleActions:
            score = 0.0

            # Center preference (game-agnostic good practice)
            center_distance = abs(action - 3)
            score += (3 - center_distance) * 0.2

            # Avoid filling columns too early
            column_height = 6 - len(np.where(board.board[:, action] == 0)[0])
            if column_height >= 5:
                score -= 1.0  # Heavy penalty for nearly full columns
            elif column_height >= 4:
                score -= 0.5  # Moderate penalty

            # Basic connectivity bonus
            if column_height > 0:
                # Check if we're building on our own pieces
                row_below = 6 - column_height
                if row_below < 6 and board.board[row_below, action] == startValue:
                    score += 0.3
                elif row_below < 6 and board.board[row_below, action] == -startValue:
                    score -= 0.2  # Avoid helping opponent

            action_scores.append((action, score))

        # Return action with highest score
        action_scores.sort(key=lambda x: x[1], reverse=True)
        return action_scores[0][0]

    def _encode_state_contour_aware(self, board, startValue):
        """
        Enhanced 200-feature state encoding optimized for contour formation
        Critical: Must match training encoding exactly
        """
        features = []

        # === CORE BOARD STATE (42 features) ===
        board_normalized = board.board * startValue
        features.extend(board_normalized.flatten())

        # === CRITICAL: COMPONENT ANALYSIS (84 features) ===
        # These features are essential for understanding contour formation
        components_normalized = np.sign(board.components) * startValue
        features.extend(components_normalized.flatten())

        components4_normalized = np.sign(board.components4) * startValue
        features.extend(components4_normalized.flatten())

        # === CONTOUR-SPECIFIC FEATURES (74 features) ===

        # Column analysis with contour potential (14 features)
        for col in range(7):
            height = 6 - len(np.where(board.board[:, col] == 0)[0])
            features.append(height / 6.0)

            # Simplified contour potential
            contour_potential = self._calculate_contour_potential(board, col, startValue)
            features.append(contour_potential)

        # Component structure analysis (14 features)
        own_components = self._analyze_components_fast(board, startValue)
        opp_components = self._analyze_components_fast(board, -startValue)
        features.extend(own_components[:7])  # Top 7 features
        features.extend(opp_components[:7])  # Top 7 features

        # Enclosure threat detection (14 features)
        for col in range(7):
            can_enclose = self._can_create_enclosure_fast(board, col, startValue)
            being_enclosed = self._being_enclosed_threat_fast(board, col, startValue)
            features.extend([can_enclose, being_enclosed])

        # Strategic positioning (16 features)
        strategic_features = self._extract_strategic_features(board, startValue)
        features.extend(strategic_features)

        # Game phase and meta-features (16 features)
        meta_features = self._extract_meta_features(board, startValue)
        features.extend(meta_features)

        # Ensure exactly 200 features
        while len(features) < 200:
            features.append(0.0)
        features = features[:200]

        return np.array(features, dtype=np.float32)

    def _calculate_contour_potential(self, board, col, player_value):
        """Calculate potential for creating contours in a column"""
        if col not in self.getPossibleActions(board.board):
            return 0.0

        empty_rows = np.where(board.board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)

        # Count adjacent own pieces (simplified contour analysis)
        adjacent_own = 0
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 6 and 0 <= nc < 7:
                if board.board[nr, nc] == player_value:
                    adjacent_own += 1
                elif board.board[nr, nc] == -player_value:
                    adjacent_own += 0.5  # Potential for enclosure

        return min(1.0, adjacent_own / 8.0)

    def _analyze_components_fast(self, board, player_value):
        """Fast component analysis for strategic insight"""
        features = [0.0] * 10

        try:
            # Analyze diagonal components
            components = board.components * (np.sign(board.components) == np.sign(player_value))
            if np.any(components):
                unique_components, counts = np.unique(components[components != 0], return_counts=True)

                # Component size distribution
                for i, count in enumerate(counts[:5]):
                    if i < 5:
                        features[i] = min(1.0, count / 10.0)

                # Summary statistics
                if len(counts) > 0:
                    features[5] = min(1.0, np.max(counts) / 15.0)  # Largest component
                    features[6] = min(1.0, len(unique_components) / 10.0)  # Number of components
                    features[7] = min(1.0, np.mean(counts) / 8.0)  # Average size
                    features[8] = min(1.0, np.sum(counts) / 21.0)  # Total pieces in components
                    features[9] = min(1.0, np.std(counts) / 5.0)  # Size variation
        except Exception:
            pass  # Keep zeros on error

        return features

    def _can_create_enclosure_fast(self, board, col, player_value):
        """Fast check for enclosure creation potential"""
        if col not in self.getPossibleActions(board.board):
            return 0.0

        try:
            empty_rows = np.where(board.board[:, col] == 0)[0]
            if len(empty_rows) == 0:
                return 0.0

            row = np.max(empty_rows)

            # Look for C-shaped formations that could be closed
            enclosure_indicators = 0
            for direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                dr, dc = direction
                if (0 <= row + dr < 6 and 0 <= col + dc < 7 and
                        board.board[row + dr, col + dc] == player_value):
                    enclosure_indicators += 1

            return min(1.0, enclosure_indicators / 4.0)
        except Exception:
            return 0.0

    def _being_enclosed_threat_fast(self, board, col, player_value):
        """Fast check for being enclosed by opponent"""
        if col not in self.getPossibleActions(board.board):
            return 0.0

        try:
            empty_rows = np.where(board.board[:, col] == 0)[0]
            if len(empty_rows) == 0:
                return 0.0

            row = np.max(empty_rows)

            # Count opponent pieces that could form enclosures
            threat_indicators = 0
            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nr, nc = row + dr, col + dc
                if (0 <= nr < 6 and 0 <= nc < 7 and
                        board.board[nr, nc] == -player_value):
                    threat_indicators += 1

            return min(1.0, threat_indicators / 8.0)
        except Exception:
            return 0.0

    def _extract_strategic_features(self, board, startValue):
        """Extract strategic positioning features"""
        features = []

        try:
            # Center control
            center_own = sum(np.sum(board.board[:, col] == startValue) for col in [2, 3, 4])
            center_opp = sum(np.sum(board.board[:, col] == -startValue) for col in [2, 3, 4])
            features.append(center_own / 18.0)
            features.append(center_opp / 18.0)

            # Edge control
            edge_own = sum(np.sum(board.board[:, col] == startValue) for col in [0, 1, 5, 6])
            edge_opp = sum(np.sum(board.board[:, col] == -startValue) for col in [0, 1, 5, 6])
            features.append(edge_own / 24.0)
            features.append(edge_opp / 24.0)

            # Connectivity metrics (simplified)
            own_pieces = (board.board == startValue)
            opp_pieces = (board.board == -startValue)

            # Horizontal connections
            h_conn_own = np.sum(own_pieces[:, :-1] & own_pieces[:, 1:])
            h_conn_opp = np.sum(opp_pieces[:, :-1] & opp_pieces[:, 1:])
            features.append(h_conn_own / 30.0)
            features.append(h_conn_opp / 30.0)

            # Vertical connections
            v_conn_own = np.sum(own_pieces[:-1, :] & own_pieces[1:, :])
            v_conn_opp = np.sum(opp_pieces[:-1, :] & opp_pieces[1:, :])
            features.append(v_conn_own / 35.0)
            features.append(v_conn_opp / 35.0)

            # Fill remaining features
            while len(features) < 16:
                features.append(0.0)

        except Exception:
            features = [0.0] * 16

        return features[:16]

    def _extract_meta_features(self, board, startValue):
        """Extract game meta-features"""
        features = []

        try:
            # Piece counts
            total_pieces = np.sum(board.board != 0)
            own_pieces = np.sum(board.board == startValue)
            opp_pieces = np.sum(board.board == -startValue)

            features.append(total_pieces / 42.0)
            features.append(own_pieces / 21.0)
            features.append(opp_pieces / 21.0)

            # Game phase indicators
            if total_pieces < 14:
                features.extend([1.0, 0.0, 0.0])  # Early game
            elif total_pieces < 28:
                features.extend([0.0, 1.0, 0.0])  # Mid game
            else:
                features.extend([0.0, 0.0, 1.0])  # Late game

            # Starting player indicator
            features.append(1.0 if startValue == 1 else 0.0)

            # Board balance metrics
            left_own = np.sum(board.board[:, :3] == startValue)
            right_own = np.sum(board.board[:, 4:] == startValue)
            balance = 1.0 - abs(left_own - right_own) / max(left_own + right_own, 1)
            features.append(balance)

            # Column height variance
            heights = [6 - len(np.where(board.board[:, col] == 0)[0]) for col in range(7)]
            height_var = np.var(heights) / 6.0
            features.append(height_var)

            # Available moves
            possible_moves = len(self.getPossibleActions(board.board))
            features.append(possible_moves / 7.0)

            # Fill remaining
            while len(features) < 16:
                features.append(0.0)

        except Exception:
            features = [0.0] * 16

        return features[:16]

    def getPossibleActions(self, board):
        """Get list of valid column indices where pieces can be placed"""
        return np.unique(np.where(board == 0)[1]).tolist()


# For compatibility with original player.py interface
if __name__ == "__main__":
    # Test the player locally
    player = Player()
    print(f"Player created: {player.getName()}")
    print(f"CUDA available: {CUDA_AVAILABLE}")
    print("Player ready for tournament!")