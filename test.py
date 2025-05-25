#!/usr/bin/env python3
"""
Enhanced test script for the CUDA-optimized DQN player
"""

import numpy as np
import random
import gamerules
import time

# CUDA Support
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE = "cpu"
    print("⚠️ PyTorch not found. Using fallback implementation.")

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class EnhancedDQN(nn.Module):
    """Enhanced DQN (same as in training script)"""

    def __init__(self, input_size=193, hidden_sizes=[512, 256, 128], output_size=7,
                 learning_rate=0.0005, dropout_rate=0.3):
        super(EnhancedDQN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*layers)
        self.to(DEVICE)

    def forward(self, x):
        return self.network(x)

    def predict(self, state):
        self.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(DEVICE)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            return self.forward(state).cpu().numpy()

    def load_weights(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=DEVICE)
            self.load_state_dict(checkpoint['state_dict'])
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False


class FallbackDQN:
    """Fallback NumPy implementation if PyTorch is not available"""

    def __init__(self, input_size=200, hidden_sizes=[512, 256, 128], output_size=7):
        self.input_size = input_size
        self.output_size = output_size

        # Initialize network
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            std = np.sqrt(2.0 / layer_sizes[i])
            layer = {
                'weights': np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * std,
                'biases': np.zeros((1, layer_sizes[i + 1]))
            }
            self.layers.append(layer)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        current_input = x
        for i, layer in enumerate(self.layers):
            z = np.dot(current_input, layer['weights']) + layer['biases']
            if i < len(self.layers) - 1:
                current_input = self.relu(z)
            else:
                current_input = z
        return current_input

    def predict(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        return self.forward(state)

    def load_weights(self, filepath):
        # Cannot load PyTorch weights in NumPy implementation
        print("⚠️ Cannot load PyTorch weights with fallback implementation")
        return False


class EnhancedTestPlayer(gamerules.Player):
    """Enhanced test player with full contour-aware features"""

    def __init__(self, name, weights_file=None):
        super().__init__(name)
        self.name = name

        # Initialize network
        if CUDA_AVAILABLE:
            self.q_network = EnhancedDQN(
                input_size=200,
                hidden_sizes=[512, 256, 128],
                output_size=7
            )
        else:
            self.q_network = FallbackDQN(
                input_size=200,
                hidden_sizes=[512, 256, 128],
                output_size=7
            )

        # Load weights if provided
        if weights_file:
            if self.q_network.load_weights(weights_file):
                print(f"✅ Loaded enhanced weights from {weights_file}")
            else:
                print("⚠️ Failed to load weights, using random weights")
        else:
            print("ℹ️ No weights file found, using random weights")

        # Verify feature encoding produces exactly 200 features
        try:
            dummy_board = gamerules.Board()
            dummy_features = self.encode_state_contour_aware(dummy_board, 1)
            feature_count = len(dummy_features)
            print(f"✅ Feature encoding verified: {feature_count} features")
            if feature_count != 200:
                print(f"⚠️ Warning: Expected 200 features, got {feature_count}")
        except Exception as e:
            print(f"⚠️ Feature encoding test failed: {e}")

    def getName(self):
        return self.name

    def newGame(self, new_opponent):
        pass

    def encode_state_contour_aware(self, board, startValue):
        """Enhanced state encoding (same as training script)"""
        features = []

        # === CORE BOARD STATE (42 features) ===
        board_normalized = board.board * startValue
        features.extend(board_normalized.flatten())

        # === CRITICAL: COMPONENT ANALYSIS (84 features) ===
        components_normalized = np.sign(board.components) * startValue
        features.extend(components_normalized.flatten())

        components4_normalized = np.sign(board.components4) * startValue
        features.extend(components4_normalized.flatten())

        # === CONTOUR-SPECIFIC FEATURES (74 features) ===

        # Column analysis (14 features)
        for col in range(7):
            height = 6 - len(np.where(board.board[:, col] == 0)[0])
            features.append(height / 6.0)
            contour_potential = self._analyze_contour_potential(board, col, startValue)
            features.append(contour_potential)

        # Component size analysis (14 features)
        own_components = self._analyze_component_structure(board, startValue)
        opp_components = self._analyze_component_structure(board, -startValue)
        features.extend(own_components[:7])
        features.extend(opp_components[:7])

        # Enclosure analysis (14 features)
        for col in range(7):
            can_enclose = self._can_create_enclosure(board, col, startValue)
            being_enclosed = self._being_enclosed_threat(board, col, startValue)
            features.extend([can_enclose, being_enclosed])

        # Strategic positioning (14 features)
        center_control = sum(np.sum(board.board[:, col] == startValue) for col in [2, 3, 4]) / 18.0
        edge_control = sum(np.sum(board.board[:, col] == startValue) for col in [0, 1, 5, 6]) / 24.0
        features.append(center_control)
        features.append(edge_control)

        # Connectivity patterns (12 features)
        connectivity_metrics = self._analyze_connectivity_patterns(board, startValue)
        features.extend(connectivity_metrics)

        # Game phase indicators (3 features)
        total_pieces = np.sum(board.board != 0)
        if total_pieces < 14:
            features.extend([1.0, 0.0, 0.0])
        elif total_pieces < 28:
            features.extend([0.0, 1.0, 0.0])
        else:
            features.extend([0.0, 0.0, 1.0])

        # Starting player advantage (1 feature)
        features.append(1.0 if startValue == 1 else 0.0)

        # Formation stability (7 features)
        stability_metrics = self._analyze_formation_stability(board, startValue)
        features.extend(stability_metrics)

        return np.array(features, dtype=np.float32)

    def _analyze_contour_potential(self, board, col, player_value):
        """Analyze potential for creating contours in a column"""
        if col not in self.getPossibleActions(board.board):
            return 0.0

        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        temp_board[row, col] = player_value

        contour_score = 0.0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < 6 and 0 <= nc < 7:
                    if temp_board[nr, nc] == player_value:
                        contour_score += 0.1
                    elif temp_board[nr, nc] == -player_value:
                        contour_score += 0.05

        return min(1.0, contour_score)

    def _analyze_component_structure(self, board, player_value):
        """Analyze component structure"""
        features = [0.0] * 10

        components = board.components * (np.sign(board.components) == np.sign(player_value))
        if np.any(components):
            unique_components, counts = np.unique(components[components != 0], return_counts=True)

            for i, count in enumerate(counts[:7]):
                if i < 7:
                    features[i] = min(1.0, count / 10.0)

            if len(counts) > 0:
                features[7] = min(1.0, np.max(counts) / 15.0)
            features[8] = min(1.0, len(unique_components) / 10.0)
            features[9] = min(1.0, np.mean(counts) / 8.0)

        return features

    def _can_create_enclosure(self, board, col, player_value):
        """Check if playing in this column could create an enclosure"""
        if col not in self.getPossibleActions(board.board):
            return 0.0

        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        enclosure_potential = 0.0

        for direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            dr, dc = direction
            if (0 <= row + dr < 6 and 0 <= col + dc < 7 and
                    temp_board[row + dr, col + dc] == player_value):
                enclosure_potential += 0.2

        return min(1.0, enclosure_potential)

    def _being_enclosed_threat(self, board, col, player_value):
        """Check threat of being enclosed"""
        if col not in self.getPossibleActions(board.board):
            return 0.0

        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        temp_board[row, col] = -player_value

        threat_level = 0.0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (0 <= nr < 6 and 0 <= nc < 7 and
                        temp_board[nr, nc] == -player_value):
                    threat_level += 0.1

        return min(1.0, threat_level)

    def _analyze_connectivity_patterns(self, board, player_value):
        """Analyze connectivity patterns - returns exactly 16 features"""
        features = [0.0] * 16

        try:
            own_pieces = (board.board == player_value)
            opp_pieces = (board.board == -player_value)

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

    def _analyze_formation_stability(self, board, player_value):
        """Analyze stability of formations - returns exactly 7 features"""
        features = [0.0] * 7

        try:
            # Column height variance
            heights = [6 - len(np.where(board.board[:, col] == 0)[0]) for col in range(7)]
            features[0] = np.var(heights) / 6.0

            # Center column dominance
            center_col = board.board[:, 3]
            features[1] = np.sum(center_col == player_value) / 6.0
            features[2] = np.sum(center_col == -player_value) / 6.0

            # Corner control
            corners = [(0, 0), (0, 6), (5, 0), (5, 6)]
            corner_own = sum(1 for r, c in corners if board.board[r, c] == player_value)
            corner_opp = sum(1 for r, c in corners if board.board[r, c] == -player_value)
            features[3] = corner_own / 4.0
            features[4] = corner_opp / 4.0

            # Available moves
            features[5] = len(self.getPossibleActions(board.board)) / 7.0

            # Board balance (left vs right)
            left_own = np.sum(board.board[:, :3] == player_value)
            right_own = np.sum(board.board[:, 4:] == player_value)
            features[6] = 1.0 - abs(left_own - right_own) / max(left_own + right_own, 1)

        except Exception:
            pass  # Keep zeros

        return features

    def getAction(self, board, startValue):
        """Enhanced action selection with heuristics"""
        possibleActions = self.getPossibleActions(board.board)

        if len(possibleActions) == 0:
            return 0

        try:
            # Priority 1: Immediate win
            for action in possibleActions:
                if self._can_win_immediately(board, action, startValue):
                    return action

            # Priority 2: Block opponent win
            for action in possibleActions:
                if self._can_win_immediately(board, action, -startValue):
                    return action

            # Priority 3: DQN decision
            state = self.encode_state_contour_aware(board, startValue)
            q_values = self.q_network.predict(state)[0]

            # Mask invalid actions
            q_values_masked = q_values.copy()
            for i in range(7):
                if i not in possibleActions:
                    q_values_masked[i] = float('-inf')

            action = np.argmax(q_values_masked)

            if action not in possibleActions:
                # Fallback: prefer center
                center_prefs = [3, 2, 4, 1, 5, 0, 6]
                for col in center_prefs:
                    if col in possibleActions:
                        return col
                return possibleActions[0]

            return int(action)

        except Exception as e:
            print(f"Error in getAction: {e}")
            return random.choice(possibleActions)

    def _can_win_immediately(self, board, action, player_value):
        """Check for immediate win"""
        if action not in self.getPossibleActions(board.board):
            return False

        temp_board = gamerules.Board()
        temp_board.board = board.board.copy()
        temp_board.components = board.components.copy()
        temp_board.components4 = board.components4.copy()
        temp_board.componentID = board.componentID
        temp_board.component4ID = board.component4ID

        temp_board.updateBoard(action, player_value)
        return temp_board.checkVictory(action, player_value)


class RNGPlayer(gamerules.Player):
    """Random player for testing"""

    def __init__(self, name="Random"):
        super().__init__(name)

    def getAction(self, board, startValue):
        possibleActions = self.getPossibleActions(board.board)
        return np.random.choice(possibleActions)

    def newGame(self, new_opponent):
        pass


def play_single_game(player1, player2, verbose=False):
    """Play a single game between two players"""
    board = gamerules.Board()
    startValue = {player1: 1, player2: -1}

    player1.newGame(True)
    player2.newGame(True)

    players = [player1, player2]
    current_player = 0
    max_moves = 42
    moves = 0

    while moves < max_moves:
        player = players[current_player]

        try:
            action = player.getAction(board, startValue[player])
        except Exception as e:
            if verbose:
                print(f"Player {player.getName()} crashed: {e}")
            return -1 if current_player == 0 else 1

        possible_actions = board.getPossibleActions()
        if action not in possible_actions:
            if verbose:
                print(f"Player {player.getName()} made invalid move: {action}")
            return -1 if current_player == 0 else 1

        board.updateBoard(action, startValue[player])

        if verbose:
            print(f"Player {player.getName()} plays column {action}")

        if board.checkVictory(action, startValue[player]):
            if verbose:
                print(f"Player {player.getName()} wins!")
            return 1 if current_player == 0 else -1

        if len(board.getPossibleActions()) == 0:
            if verbose:
                print("Draw - board is full")
            return 0

        current_player = 1 - current_player
        moves += 1

    if verbose:
        print("Draw - max moves reached")
    return 0


def evaluate_enhanced_player(player, num_games=100, verbose=False):
    """Comprehensive evaluation of the enhanced player"""
    print(f"🧪 Evaluating Enhanced Player: {player.getName()}")
    print(f"📊 Number of games: {num_games}")
    print(f"🎯 Target: Win at least {num_games * 0.8:.0f} games (80%)")
    print(f"🖥️ Using: {'CUDA' if CUDA_AVAILABLE else 'CPU'}")
    print("-" * 60)

    random_player = RNGPlayer("Random Opponent")

    wins = 0
    draws = 0
    losses = 0
    first_wins = 0
    second_wins = 0
    games_first = 0
    games_second = 0

    game_times = []
    win_lengths = []
    loss_lengths = []

    # Progress bar
    if TQDM_AVAILABLE:
        pbar = tqdm(total=num_games, desc="Testing Enhanced Player", unit="game")

    for game in range(num_games):
        start_time = time.time()

        # Alternate starting positions
        player_starts = (game % 2 == 0)
        players = [player, random_player] if player_starts else [random_player, player]

        result = play_single_game(players[0], players[1], verbose)

        game_time = time.time() - start_time
        game_times.append(game_time)

        if result == 1:  # First player wins
            if player_starts:
                wins += 1
                first_wins += 1
                win_lengths.append(game_time)
            else:
                losses += 1
                loss_lengths.append(game_time)
        elif result == -1:  # Second player wins
            if player_starts:
                losses += 1
                loss_lengths.append(game_time)
            else:
                wins += 1
                second_wins += 1
                win_lengths.append(game_time)
        else:  # Draw
            draws += 1

        if player_starts:
            games_first += 1
        else:
            games_second += 1

        if TQDM_AVAILABLE:
            pbar.update(1)
            pbar.set_postfix_str(f"Win Rate: {(wins / num_games) * 100:.1f}%")

    if TQDM_AVAILABLE:
        pbar.close()

    # Calculate statistics
    avg_game_time = np.mean(game_times)
    avg_win_time = np.mean(win_lengths) if win_lengths else 0
    avg_loss_time = np.mean(loss_lengths) if loss_lengths else 0

    print("\n" + "=" * 70)
    print("🏆 ENHANCED PLAYER EVALUATION RESULTS")
    print("=" * 70)
    print("Overall Performance:")
    print(f"  Wins:    {wins:3d}/{num_games} ({wins / num_games * 100:5.1f}%)")
    print(f"  Draws:   {draws:3d}/{num_games} ({draws / num_games * 100:5.1f}%)")
    print(f"  Losses:  {losses:3d}/{num_games} ({losses / num_games * 100:5.1f}%)")

    print("\nDetailed Performance:")
    print(f"  When starting first:  {first_wins:2d}/{games_first} ({first_wins / max(1, games_first) * 100:5.1f}%)")
    print(f"  When starting second: {second_wins:2d}/{games_second} ({second_wins / max(1, games_second) * 100:5.1f}%)")

    print("\nPerformance Metrics:")
    print(f"  Non-loss rate:     {((wins + draws) / num_games) * 100:5.1f}%")
    print(f"  Win/Draw ratio:    {wins / (max(1, draws)):5.2f}")
    print(f"  Avg game time:     {avg_game_time:5.3f}s")
    print(f"  Avg win time:      {avg_win_time:5.3f}s")
    print(f"  Avg loss time:     {avg_loss_time:5.3f}s")

    # Technical info
    print(f"\nTechnical Info:")
    print(f"  Device:            {'CUDA' if CUDA_AVAILABLE else 'CPU'}")
    print(f"  State encoding:    200 features (contour-aware)")
    print(f"  Network arch:      [512, 256, 128] with BatchNorm + Dropout")

    if wins >= num_games * 0.8:
        print("\n✅ SUCCESS! Enhanced player meets the requirement.")
        print(f"   Required: {num_games * 0.8:.0f} wins, Achieved: {wins} wins")
        print("   🎉 The enhanced contour-aware architecture is working!")
    else:
        print("\n❌ FAILED. Enhanced player does not meet the requirement.")
        print(f"   Required: {num_games * 0.8:.0f} wins, Achieved: {wins} wins")
        print("\nPossible improvements:")
        print("1. Train for more episodes (current suggestion: 10,000+)")
        print("2. Improve contour detection heuristics")
        print("3. Add more sophisticated component analysis")
        print("4. Consider ensemble methods")

    print("=" * 70)

    return wins, draws, losses


def main():
    """Main testing function"""
    import argparse

    parser = argparse.ArgumentParser(description='Test Enhanced DQN Player')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of test games (default: 100)')
    parser.add_argument('--weights', type=str, default='enhanced_weights.pth',
                        help='Enhanced weights file to load (default: enhanced_weights.pth)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed game information')
    parser.add_argument('--name', type=str, default='Enhanced Taras Demchyna',
                        help='Player name (default: Enhanced Taras Demchyna)')

    args = parser.parse_args()

    print("🚀 Enhanced DQN Player Evaluation")
    print("=" * 50)

    # Create and load the enhanced player
    try:
        enhanced_player = EnhancedTestPlayer(args.name, args.weights)
        print(f"✅ Successfully loaded enhanced player")
    except Exception as e:
        print(f"❌ Error loading enhanced player: {e}")
        return 1

    # Run evaluation
    try:
        results = evaluate_enhanced_player(enhanced_player, args.games, args.verbose)
        return 0 if results[0] >= args.games * 0.8 else 1

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())