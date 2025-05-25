#!/usr/bin/env python3
"""
Fast test script for the fast-trained model.
Compatible with fast_weights.pkl from fast_train.py
"""

import numpy as np
import gamerules
import time
import pickle
import os

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class FastDQN:
    """Fast DQN network (same as in fast_train.py)"""

    def __init__(self, input_size=84, hidden_sizes=[128, 64], output_size=7, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate

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

            if i < len(self.layers) - 1:  # Hidden layers
                activation = self.relu(z)
            else:  # Output layer
                activation = z

            current_input = activation

        return current_input

    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return self.forward(x)

    def load_weights(self, filepath):
        """Load weights from file"""
        try:
            with open(filepath, 'rb') as f:
                weights_data = pickle.load(f)

            # Verify compatibility
            if (weights_data['input_size'] != self.input_size or
                    weights_data['output_size'] != self.output_size or
                    len(weights_data['layers']) != len(self.layers)):
                print("Warning: Architecture mismatch")
                return False

            # Load weights
            for i, layer_data in enumerate(weights_data['layers']):
                if (layer_data['weights'].shape != self.layers[i]['weights'].shape or
                        layer_data['biases'].shape != self.layers[i]['biases'].shape):
                    print(f"Warning: Layer {i} shape mismatch")
                    return False

                self.layers[i]['weights'] = layer_data['weights']
                self.layers[i]['biases'] = layer_data['biases']

            return True

        except Exception as e:
            print(f"Error loading weights: {e}")
            return False


class FastTestPlayer(gamerules.Player):
    """Fast test player compatible with fast training"""

    def __init__(self, name, weights_file=None):
        super().__init__(name)
        self.name = name

        # Initialize fast DQN
        self.q_network = FastDQN(
            input_size=84,
            hidden_sizes=[128, 64],
            output_size=7,
            learning_rate=0.001
        )

        # Load weights if provided
        if weights_file and os.path.exists(weights_file):
            if self.q_network.load_weights(weights_file):
                print(f"✅ Loaded fast weights from {weights_file}")
            else:
                print("⚠️ Failed to load weights, using random weights")
        else:
            print("ℹ️ No weights file found, using random weights")

    def getName(self):
        return self.name

    def newGame(self, new_opponent):
        pass

    def encode_state_fast(self, board, startValue):
        """Fast state encoding (same as in fast_train.py)"""
        features = []

        # Basic board state (42 features)
        board_normalized = board.board * startValue
        features.extend(board_normalized.flatten())

        # Column heights (7 features)
        for col in range(7):
            height = 6 - len(np.where(board.board[:, col] == 0)[0])
            features.append(height / 6.0)

        # Piece counts per column (14 features)
        for col in range(7):
            own_pieces = np.sum(board.board[:, col] == startValue)
            opp_pieces = np.sum(board.board[:, col] == -startValue)
            features.append(own_pieces / 6.0)
            features.append(opp_pieces / 6.0)

        # Center control (3 features)
        center_own = sum(np.sum(board.board[:, col] == startValue) for col in [2, 3, 4])
        center_opp = sum(np.sum(board.board[:, col] == -startValue) for col in [2, 3, 4])
        total_pieces = np.sum(board.board != 0)
        features.extend([center_own / 18.0, center_opp / 18.0, total_pieces / 42.0])

        # Immediate threats (14 features)
        for col in range(7):
            can_win = self._can_win_fast(board, col, startValue)
            must_block = self._can_win_fast(board, col, -startValue)
            features.extend([1.0 if can_win else 0.0, 1.0 if must_block else 0.0])

        # Game phase (3 features)
        if total_pieces < 14:
            features.extend([1.0, 0.0, 0.0])
        elif total_pieces < 28:
            features.extend([0.0, 1.0, 0.0])
        else:
            features.extend([0.0, 0.0, 1.0])

        # Starting player (1 feature)
        features.append(1.0 if startValue == 1 else 0.0)

        return np.array(features, dtype=np.float32)

    def _can_win_fast(self, board, col, player_value):
        """Fast win detection"""
        if col not in self.getPossibleActions(board.board):
            return False

        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return False

        row = np.max(empty_rows)
        temp_board[row, col] = player_value

        temp_board_obj = gamerules.Board()
        temp_board_obj.board = temp_board
        temp_board_obj.components = board.components.copy()
        temp_board_obj.components4 = board.components4.copy()
        temp_board_obj.updateComponents(row, col, player_value)
        temp_board_obj.updateComponents4(row, col, player_value)

        return temp_board_obj.checkVictory(col, player_value)

    def getAction(self, board, startValue):
        """Hybrid action selection with heuristics + fast DQN"""
        possibleActions = self.getPossibleActions(board.board)

        if len(possibleActions) == 0:
            return 0

        try:
            # Heuristic checks first
            # Immediate win
            for action in possibleActions:
                if self._can_win_fast(board, action, startValue):
                    return action

            # Block opponent win
            for action in possibleActions:
                if self._can_win_fast(board, action, -startValue):
                    return action

            # Use DQN for strategic decisions
            state = self.encode_state_fast(board, startValue)
            q_values = self.q_network.predict(state)[0]

            # Mask invalid actions
            q_values_masked = q_values.copy()
            for i in range(7):
                if i not in possibleActions:
                    q_values_masked[i] = float('-inf')

            action = np.argmax(q_values_masked)
            return int(action) if action in possibleActions else possibleActions[0]

        except Exception as e:
            print(f"Error in getAction: {e}")
            # Fallback: center preference
            center_prefs = [3, 2, 4, 1, 5, 0, 6]
            for col in center_prefs:
                if col in possibleActions:
                    return col
            return possibleActions[0]


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


def evaluate_fast_player(player, num_games=100, verbose=False):
    """Evaluate the fast-trained player with enhanced metrics"""
    print(f"🧪 Evaluating Fast Player: {player.getName()}")
    print(f"📊 Number of games: {num_games}")
    print(f"🎯 Target: Win at least {num_games * 0.8:.0f} games (80%)")
    print("-" * 50)

    random_player = RNGPlayer("Random Opponent")

    wins = 0
    draws = 0
    losses = 0
    first_wins = 0
    second_wins = 0
    games_first = 0
    games_second = 0
    total_moves = 0

    # Progress bar
    if TQDM_AVAILABLE:
        pbar = tqdm(total=num_games, desc="Testing", unit="game")

    for game in range(num_games):
        # Alternate starting positions
        player_starts = (game % 2 == 0)
        players = [player, random_player] if player_starts else [random_player, player]
        startValue = [1, -1]

        result = play_single_game(players[0], players[1], verbose)
        
        if result == 1:  # First player wins
            if player_starts:
                wins += 1
                first_wins += 1
            else:
                losses += 1
            games_first += player_starts
        elif result == -1:  # Second player wins
            if player_starts:
                losses += 1
            else:
                wins += 1
                second_wins += 1
            games_second += not player_starts
        else:  # Draw
            draws += 1
            games_first += player_starts
            games_second += not player_starts

        if TQDM_AVAILABLE:
            pbar.update(1)
            pbar.set_postfix_str(f"Win Rate: {(wins/num_games)*100:.1f}%")

    if TQDM_AVAILABLE:
        pbar.close()

    print("\n" + "=" * 60)
    print("🏆 FAST PLAYER EVALUATION RESULTS")
    print("=" * 60)
    print("Overall Performance:")
    print(f"  Wins:    {wins:3d}/{num_games} ({wins/num_games*100:5.1f}%)")
    print(f"  Draws:   {draws:3d}/{num_games} ({draws/num_games*100:5.1f}%)")
    print(f"  Losses:  {losses:3d}/{num_games} ({losses/num_games*100:5.1f}%)")
    print("\nDetailed Performance:")
    print(f"  When starting first:  {first_wins:2d}/{games_first} ({first_wins/max(1,games_first)*100:5.1f}%)")
    print(f"  When starting second: {second_wins:2d}/{games_second} ({second_wins/max(1,games_second)*100:5.1f}%)")
    
    # Additional metrics
    print("\nAdvanced Metrics:")
    print(f"  Non-loss rate: {((wins + draws)/num_games)*100:5.1f}%")
    print(f"  Win/Draw ratio: {wins/(max(1,draws)):5.2f}")
    
    if wins >= num_games * 0.8:
        print("\n✅ SUCCESS! Fast player meets the requirement.")
        print(f"   Required: {num_games * 0.8:.0f} wins, Achieved: {wins} wins")
    else:
        print("\n❌ FAILED. Fast player does not meet the requirement.")
        print(f"   Required: {num_games * 0.8:.0f} wins, Achieved: {wins} wins")
        print("\nRecommendations:")
        print("1. Try training for more episodes")
        print("2. Increase the batch size")
        print("3. Adjust the learning rate")
        print("4. Consider using a more complex network architecture")
    
    print("=" * 60)
    
    return wins, draws, losses


def main():
    """Main testing function for fast-trained player"""
    import argparse

    parser = argparse.ArgumentParser(description='Test fast-trained DQN player')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of test games (default: 100)')
    parser.add_argument('--weights', type=str, default='fast_weights.pkl',
                        help='Fast weights file to load (default: fast_weights.pkl)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed game information')
    parser.add_argument('--name', type=str, default='Fast Taras Demchyna',
                        help='Player name (default: Fast Taras Demchyna)')

    args = parser.parse_args()

    print("⚡ Fast DQN Player Evaluation")
    print("=" * 50)

    # Create and load the fast player
    try:
        fast_player = FastTestPlayer(args.name, args.weights)
        print(f"✅ Successfully loaded fast player with weights from '{args.weights}'")
    except Exception as e:
        print(f"❌ Error loading fast player: {e}")
        return 1

    # Run evaluation
    try:
        results = evaluate_fast_player(fast_player, args.games, args.verbose)
        return 0 if results[0] >= args.games * 0.8 else 1

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())