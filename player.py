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

        # Verify feature encoding produces exactly 200 features
        try:
            dummy_board = gamerules.Board()
            dummy_features = self._encode_state_contour_aware(dummy_board, 1)
            feature_count = len(dummy_features)
            if feature_count != 200:
                print(f"⚠️ Warning: Expected 200 features, got {feature_count}")
        except Exception as e:
            print(f"⚠️ Feature encoding test failed: {e}")

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

    # def _encode_state_contour_aware(self, board, startValue):
    #     """
    #     Enhanced 200-feature state encoding optimized for contour formation
    #     Critical: Must match training encoding exactly
    #     """
    #     features = []
    #
    #     try:
    #         # === CORE BOARD STATE (42 features) ===
    #         board_normalized = board.board * startValue
    #         features.extend(board_normalized.flatten())
    #
    #         # === CRITICAL: COMPONENT ANALYSIS (84 features) ===
    #         # These features are essential for understanding contour formation
    #         components_normalized = np.sign(board.components) * startValue
    #         features.extend(components_normalized.flatten())
    #
    #         components4_normalized = np.sign(board.components4) * startValue
    #         features.extend(components4_normalized.flatten())
    #
    #         # === CONTOUR-SPECIFIC FEATURES (74 features) ===
    #
    #         # Column heights and basic analysis (14 features)
    #         for col in range(7):
    #             height = 6 - len(np.where(board.board[:, col] == 0)[0])
    #             features.append(height / 6.0)
    #             # Contour potential
    #             contour_potential = self._calculate_contour_potential(board, col, startValue)
    #             features.append(contour_potential)
    #
    #         # Piece distribution analysis (16 features)
    #         center_own = sum(np.sum(board.board[:, col] == startValue) for col in [2, 3, 4])
    #         center_opp = sum(np.sum(board.board[:, col] == -startValue) for col in [2, 3, 4])
    #         edge_own = sum(np.sum(board.board[:, col] == startValue) for col in [0, 1, 5, 6])
    #         edge_opp = sum(np.sum(board.board[:, col] == -startValue) for col in [0, 1, 5, 6])
    #
    #         total_pieces = np.sum(board.board != 0)
    #         own_pieces = np.sum(board.board == startValue)
    #         opp_pieces = np.sum(board.board == -startValue)
    #
    #         features.extend([
    #             center_own / 18.0, center_opp / 18.0, edge_own / 24.0, edge_opp / 24.0,
    #             total_pieces / 42.0, own_pieces / 21.0, opp_pieces / 21.0,
    #             1.0 if startValue == 1 else 0.0  # Starting player
    #         ])
    #
    #         # Game phase indicators (3 features)
    #         if total_pieces < 14:
    #             features.extend([1.0, 0.0, 0.0])
    #         elif total_pieces < 28:
    #             features.extend([0.0, 1.0, 0.0])
    #         else:
    #             features.extend([0.0, 0.0, 1.0])
    #
    #         # Connectivity analysis (16 features)
    #         connectivity_features = self._extract_strategic_features(board, startValue)
    #         features.extend(connectivity_features)
    #
    #         # Component structure analysis (14 features)
    #         component_features = self._analyze_components_fast(board, startValue)
    #         features.extend(component_features[:7])  # Own components
    #         component_features_opp = self._analyze_components_fast(board, -startValue)
    #         features.extend(component_features_opp[:7])  # Opponent components
    #
    #         # Threat analysis (14 features)
    #         for col in range(7):
    #             can_win = 1.0 if self._can_win_immediately(board, col, startValue) else 0.0
    #             must_block = 1.0 if self._can_win_immediately(board, col, -startValue) else 0.0
    #             features.extend([can_win, must_block])
    #
    #         # Positional features (7 features)
    #         positional_features = self._extract_meta_features(board, startValue)
    #         features.extend(positional_features)
    #
    #         # === ENSURE EXACTLY 200 FEATURES ===
    #         current_length = len(features)
    #         if current_length < 200:
    #             # Pad with zeros
    #             features.extend([0.0] * (200 - current_length))
    #         elif current_length > 200:
    #             # Truncate to 200
    #             features = features[:200]
    #
    #         return np.array(features, dtype=np.float32)
    #
    #     except Exception:
    #         # Emergency fallback - return 200 zeros
    #         return np.zeros(200, dtype=np.float32)

    def encode_state_contour_aware(self, board, startValue):
        """FIXED: Guaranteed 200-feature encoding with NO missing helper functions"""
        features = []

        # === CORE BOARD STATE (42 features) ===
        board_normalized = board.board * startValue
        features.extend(board_normalized.flatten())

        # === COMPONENT ANALYSIS (84 features) ===
        components_normalized = np.sign(board.components) * startValue
        features.extend(components_normalized.flatten())

        components4_normalized = np.sign(board.components4) * startValue
        features.extend(components4_normalized.flatten())

        # At this point: 42 + 42 + 42 = 126 features

        # === REMAINING 74 FEATURES (simple implementations) ===

        # Column heights (7 features)
        for col in range(7):
            height = 6 - len(np.where(board.board[:, col] == 0)[0])
            features.append(height / 6.0)

        # Column piece distribution (14 features)
        for col in range(7):
            own_pieces = np.sum(board.board[:, col] == startValue)
            opp_pieces = np.sum(board.board[:, col] == -startValue)
            features.append(own_pieces / 6.0)
            features.append(opp_pieces / 6.0)

        # Center vs edge control (8 features)
        center_own = sum(np.sum(board.board[:, col] == startValue) for col in [2, 3, 4])
        center_opp = sum(np.sum(board.board[:, col] == -startValue) for col in [2, 3, 4])
        edge_own = sum(np.sum(board.board[:, col] == startValue) for col in [0, 1, 5, 6])
        edge_opp = sum(np.sum(board.board[:, col] == -startValue) for col in [0, 1, 5, 6])
        features.extend([center_own / 18.0, center_opp / 18.0, edge_own / 24.0, edge_opp / 24.0])

        # Game state (8 features)
        total_pieces = np.sum(board.board != 0)
        own_pieces = np.sum(board.board == startValue)
        opp_pieces = np.sum(board.board == -startValue)
        possible_actions = len(self.getPossibleActions(board.board))

        features.extend([
            total_pieces / 42.0, own_pieces / 21.0, opp_pieces / 21.0, possible_actions / 7.0
        ])

        # Game phase (3 features)
        if total_pieces < 14:
            features.extend([1.0, 0.0, 0.0])
        elif total_pieces < 28:
            features.extend([0.0, 1.0, 0.0])
        else:
            features.extend([0.0, 0.0, 1.0])

        # Starting player (1 feature)
        features.append(1.0 if startValue == 1 else 0.0)

        # Simple threat detection using EXISTING methods only (14 features)
        for col in range(7):
            can_win = 0.0
            must_block = 0.0

            possible_actions = self.getPossibleActions(board.board)
            if col in possible_actions:
                # Test for immediate win
                try:
                    temp_board = gamerules.Board()
                    temp_board.board = board.board.copy()
                    temp_board.components = board.components.copy()
                    temp_board.components4 = board.components4.copy()
                    temp_board.componentID = board.componentID
                    temp_board.component4ID = board.component4ID

                    temp_board.updateBoard(col, startValue)
                    if temp_board.checkVictory(col, startValue):
                        can_win = 1.0
                except:
                    pass

                # Test for must block
                try:
                    temp_board2 = gamerules.Board()
                    temp_board2.board = board.board.copy()
                    temp_board2.components = board.components.copy()
                    temp_board2.components4 = board.components4.copy()
                    temp_board2.componentID = board.componentID
                    temp_board2.component4ID = board.component4ID

                    temp_board2.updateBoard(col, -startValue)
                    if temp_board2.checkVictory(col, -startValue):
                        must_block = 1.0
                except:
                    pass

            features.extend([can_win, must_block])

        # Simple connectivity analysis (19 features to reach 200 total)
        # Horizontal connections
        h_conn_own = h_conn_opp = 0
        for row in range(6):
            for col in range(6):
                if board.board[row, col] == startValue and board.board[row, col + 1] == startValue:
                    h_conn_own += 1
                if board.board[row, col] == -startValue and board.board[row, col + 1] == -startValue:
                    h_conn_opp += 1
        features.append(h_conn_own / 30.0)
        features.append(h_conn_opp / 30.0)

        # Vertical connections
        v_conn_own = v_conn_opp = 0
        for row in range(5):
            for col in range(7):
                if board.board[row, col] == startValue and board.board[row + 1, col] == startValue:
                    v_conn_own += 1
                if board.board[row, col] == -startValue and board.board[row + 1, col] == -startValue:
                    v_conn_opp += 1
        features.append(v_conn_own / 35.0)
        features.append(v_conn_opp / 35.0)

        # Diagonal connections
        d_conn_own = d_conn_opp = 0
        for row in range(5):
            for col in range(6):
                if board.board[row, col] == startValue and board.board[row + 1, col + 1] == startValue:
                    d_conn_own += 1
                if board.board[row, col] == -startValue and board.board[row + 1, col + 1] == -startValue:
                    d_conn_opp += 1
        features.append(d_conn_own / 30.0)
        features.append(d_conn_opp / 30.0)

        # Position analysis (13 more features to reach exactly 200)
        heights = [6 - len(np.where(board.board[:, col] == 0)[0]) for col in range(7)]
        features.append(np.var(heights) / 6.0)  # Height variance
        features.append(np.sum(board.board[:, 3] == startValue) / 6.0)  # Center column own
        features.append(np.sum(board.board[:, 3] == -startValue) / 6.0)  # Center column opp

        # Corner control
        corners = [(0, 0), (0, 6), (5, 0), (5, 6)]
        corner_own = sum(1 for r, c in corners if board.board[r, c] == startValue)
        corner_opp = sum(1 for r, c in corners if board.board[r, c] == -startValue)
        features.append(corner_own / 4.0)
        features.append(corner_opp / 4.0)

        # Board balance
        left_own = np.sum(board.board[:, :3] == startValue)
        right_own = np.sum(board.board[:, 4:] == startValue)
        features.append(1.0 - abs(left_own - right_own) / max(left_own + right_own, 1))

        # Fill remaining slots to guarantee exactly 200
        while len(features) < 200:
            features.append(0.0)

        # Ensure exactly 200 features
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
        """Extract strategic positioning features - returns exactly 16 features"""
        features = [0.0] * 16

        try:
            own_pieces = (board.board == startValue)
            opp_pieces = (board.board == -startValue)

            # Horizontal connections (2 features)
            h_conn_own = 0
            h_conn_opp = 0
            for row in range(6):
                for col in range(6):
                    if own_pieces[row, col] and own_pieces[row, col + 1]:
                        h_conn_own += 1
                    if opp_pieces[row, col] and opp_pieces[row, col + 1]:
                        h_conn_opp += 1
            features[0] = h_conn_own / 30.0
            features[1] = h_conn_opp / 30.0

            # Vertical connections (2 features)
            v_conn_own = 0
            v_conn_opp = 0
            for row in range(5):
                for col in range(7):
                    if own_pieces[row, col] and own_pieces[row + 1, col]:
                        v_conn_own += 1
                    if opp_pieces[row, col] and opp_pieces[row + 1, col]:
                        v_conn_opp += 1
            features[2] = v_conn_own / 35.0
            features[3] = v_conn_opp / 35.0

            # Diagonal connections (4 features)
            d1_own = d1_opp = d2_own = d2_opp = 0
            for row in range(5):
                for col in range(6):
                    if own_pieces[row, col] and own_pieces[row + 1, col + 1]:
                        d1_own += 1
                    if opp_pieces[row, col] and opp_pieces[row + 1, col + 1]:
                        d1_opp += 1
                    if own_pieces[row, col + 1] and own_pieces[row + 1, col]:
                        d2_own += 1
                    if opp_pieces[row, col + 1] and opp_pieces[row + 1, col]:
                        d2_opp += 1
            features[4] = d1_own / 30.0
            features[5] = d1_opp / 30.0
            features[6] = d2_own / 30.0
            features[7] = d2_opp / 30.0

            # Fill remaining 8 features with column-wise analysis
            for col in range(7):
                if col < 8:
                    col_own = np.sum(own_pieces[:, col])
                    features[8 + col] = col_own / 6.0

            # Last feature - overall connectivity density
            features[15] = (features[0] + features[2] + features[4] + features[6]) / 4.0

        except Exception:
            pass  # Keep zeros

        return features

    def _extract_meta_features(self, board, startValue):
        """Extract game meta-features - returns exactly 7 features"""
        features = [0.0] * 7

        try:
            # Column height variance
            heights = [6 - len(np.where(board.board[:, col] == 0)[0]) for col in range(7)]
            features[0] = np.var(heights) / 6.0

            # Center column dominance
            center_col = board.board[:, 3]
            features[1] = np.sum(center_col == startValue) / 6.0
            features[2] = np.sum(center_col == -startValue) / 6.0

            # Corner control
            corners = [(0, 0), (0, 6), (5, 0), (5, 6)]
            corner_own = sum(1 for r, c in corners if board.board[r, c] == startValue)
            corner_opp = sum(1 for r, c in corners if board.board[r, c] == -startValue)
            features[3] = corner_own / 4.0
            features[4] = corner_opp / 4.0

            # Available moves
            features[5] = len(self.getPossibleActions(board.board)) / 7.0

            # Board balance (left vs right)
            left_own = np.sum(board.board[:, :3] == startValue)
            right_own = np.sum(board.board[:, 4:] == startValue)
            features[6] = 1.0 - abs(left_own - right_own) / max(left_own + right_own, 1)

        except Exception:
            pass  # Keep zeros

        return features

    def getPossibleActions(self, board):
        """Get list of valid column indices where pieces can be placed"""
        return np.unique(np.where(board == 0)[1]).tolist()

    def debug_encoding(self, board, startValue):
        """Debug version to see exactly how many features each section produces"""
        features = []

        print("=== DEBUGGING FEATURE ENCODING ===")

        # === CORE BOARD STATE (42 features) ===
        board_normalized = board.board * startValue
        features.extend(board_normalized.flatten())
        print(f"After board state: {len(features)} features")

        # === COMPONENT ANALYSIS (84 features) ===
        components_normalized = np.sign(board.components) * startValue
        features.extend(components_normalized.flatten())
        print(f"After components: {len(features)} features")

        components4_normalized = np.sign(board.components4) * startValue
        features.extend(components4_normalized.flatten())
        print(f"After components4: {len(features)} features")

        # === STRATEGIC FEATURES ===
        # Column heights and basic analysis (14 features)
        before_columns = len(features)
        for col in range(7):
            height = 6 - len(np.where(board.board[:, col] == 0)[0])
            features.append(height / 6.0)

            # Check if helper function exists
            try:
                contour_potential = self._calculate_contour_potential_safe(board, col, startValue)
                features.append(contour_potential)
            except AttributeError:
                print(f"⚠️ _calculate_contour_potential_safe not found!")
                features.append(0.0)
            except Exception as e:
                print(f"⚠️ Error in contour potential: {e}")
                features.append(0.0)
        print(f"After column analysis: {len(features)} features (added {len(features) - before_columns})")

        # Continue with each section...
        return len(features)


# For compatibility with original player.py interface
if __name__ == "__main__":
    # Test the player locally
    player = Player()
    print(f"Player created: {player.getName()}")
    print(f"CUDA available: {CUDA_AVAILABLE}")
    print("Player ready for tournament!")