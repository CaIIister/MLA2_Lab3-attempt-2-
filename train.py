#!/usr/bin/env python3
"""
Enhanced DQN Training with CUDA Support and Improved Architecture
Optimized for the contour-formation game mechanics
"""

import numpy as np
import random
from collections import deque
import pickle
import gamerules
import copy
import time

# CUDA Support Detection and Setup
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

    # Check PyTorch version for compatibility
    torch_version = torch.__version__
    print(f"🚀 PyTorch Version: {torch_version}")
    print(f"🚀 CUDA Available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        # Clear GPU cache for clean start
        torch.cuda.empty_cache()
except ImportError as e:
    CUDA_AVAILABLE = False
    DEVICE = "cpu"
    print(f"⚠️ PyTorch not found: {e}")
    print("Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class EnhancedDQN(nn.Module):
    """Enhanced DQN with CUDA support and improved architecture for contour game"""

    def __init__(self, input_size=200, hidden_sizes=[512, 256, 128], output_size=7,
                 learning_rate=0.0005, dropout_rate=0.3):
        super(EnhancedDQN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Enhanced architecture with batch normalization
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Don't add activation/normalization to output layer
            if i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*layers)

        # Initialize weights with proper scaling
        self.apply(self._init_weights)

        # Move to GPU if available
        self.to(DEVICE)

        # Enhanced optimizer with weight decay (compatible with older PyTorch)
        try:
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate,
                                         weight_decay=1e-4, eps=1e-7)
        except AttributeError:
            # Fallback to Adam for older PyTorch versions without AdamW
            print("⚠️ AdamW not available, using Adam optimizer")
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                        weight_decay=1e-4, eps=1e-7)

        # Learning rate scheduler (compatible with older PyTorch versions)
        try:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.8, patience=15,
                min_lr=1e-6, verbose=True
            )
        except TypeError:
            # Fallback for older PyTorch versions without verbose parameter
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.8, patience=15,
                min_lr=1e-6
            )

        # Loss function with smoothing
        self.criterion = nn.SmoothL1Loss()

    def _init_weights(self, m):
        """Proper weight initialization"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        return self.network(x)

    def predict(self, state):
        """Predict Q-values for given state"""
        self.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(DEVICE)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            return self.forward(state).cpu().numpy()

    def train_step(self, states, targets):
        """Training step with proper gradient handling"""
        self.train()

        # Convert to tensors and move to GPU
        if isinstance(states, np.ndarray):
            states = torch.FloatTensor(states).to(DEVICE)
        if isinstance(targets, np.ndarray):
            targets = torch.FloatTensor(targets).to(DEVICE)

        # Forward pass
        predictions = self.forward(states)

        # Compute loss
        loss = self.criterion(predictions, targets)

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_learning_rate(self, loss):
        """Update learning rate based on loss"""
        self.scheduler.step(loss)

    def save_weights(self, filepath):
        """Save model weights"""
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size
        }, filepath)

    def load_weights(self, filepath):
        """Load model weights"""
        try:
            checkpoint = torch.load(filepath, map_location=DEVICE)
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False


class ContourGamePlayer(gamerules.Player):
    """Enhanced player specifically designed for contour formation game"""

    def __init__(self, name, use_cuda=True):
        super().__init__(name)
        self.name = name
        self.use_cuda = use_cuda and CUDA_AVAILABLE

        # Enhanced Q-network
        self.q_network = EnhancedDQN(
            input_size=200,  # Full feature set
            hidden_sizes=[512, 256, 128],  # Larger network
            output_size=7,
            learning_rate=0.0005,  # Conservative learning rate
            dropout_rate=0.3
        )

        # Training parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995  # Much more conservative

        print(f"🧠 Enhanced Player initialized with {'CUDA' if self.use_cuda else 'CPU'}")

        # Verify feature encoding works correctly
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
        """Enhanced state encoding specifically for contour formation game"""
        features = []

        # === CORE BOARD STATE (42 features) ===
        board_normalized = board.board * startValue
        features.extend(board_normalized.flatten())

        # === CRITICAL: COMPONENT ANALYSIS (84 features) ===
        # These are crucial for understanding contour formation!
        components_normalized = np.sign(board.components) * startValue
        features.extend(components_normalized.flatten())

        components4_normalized = np.sign(board.components4) * startValue
        features.extend(components4_normalized.flatten())

        # === CONTOUR-SPECIFIC FEATURES (74 features) ===

        # Column analysis (14 features)
        for col in range(7):
            height = 6 - len(np.where(board.board[:, col] == 0)[0])
            features.append(height / 6.0)
            # Contour potential in column
            contour_potential = self._analyze_contour_potential(board, col, startValue)
            features.append(contour_potential)

        # Component size analysis (21 features)
        own_components = self._analyze_component_structure(board, startValue)
        opp_components = self._analyze_component_structure(board, -startValue)
        features.extend(own_components[:7])  # Top 7 component features
        features.extend(opp_components[:7])  # Top 7 opponent features

        # Enclosure analysis (14 features) - KEY FOR THIS GAME
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
            features.extend([1.0, 0.0, 0.0])  # Early game
        elif total_pieces < 28:
            features.extend([0.0, 1.0, 0.0])  # Mid game
        else:
            features.extend([0.0, 0.0, 1.0])  # Late game

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

        # Simulate placing piece
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        temp_board[row, col] = player_value

        # Check surrounding area for contour formation potential
        contour_score = 0.0

        # Count nearby own pieces that could form contours
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < 6 and 0 <= nc < 7:
                    if temp_board[nr, nc] == player_value:
                        contour_score += 0.1
                    elif temp_board[nr, nc] == -player_value:
                        contour_score += 0.05  # Could potentially enclose

        return min(1.0, contour_score)

    def _analyze_component_structure(self, board, player_value):
        """Analyze component structure for strategic insights"""
        features = [0.0] * 10

        # Analyze diagonal components
        components = board.components * (np.sign(board.components) == np.sign(player_value))
        if np.any(components):
            unique_components, counts = np.unique(components[components != 0], return_counts=True)

            # Component size distribution
            for i, count in enumerate(counts[:7]):
                if i < 7:
                    features[i] = min(1.0, count / 10.0)

            # Largest component size
            if len(counts) > 0:
                features[7] = min(1.0, np.max(counts) / 15.0)

            # Number of components
            features[8] = min(1.0, len(unique_components) / 10.0)

            # Average component size
            features[9] = min(1.0, np.mean(counts) / 8.0)

        return features

    def _can_create_enclosure(self, board, col, player_value):
        """Check if playing in this column could create an enclosure"""
        if col not in self.getPossibleActions(board.board):
            return 0.0

        # This is a simplified check - in practice, you'd want more sophisticated analysis
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)

        # Check if this position could help form a boundary
        # Look for potential enclosure patterns
        enclosure_potential = 0.0

        # Check for C-shaped formations that could be closed
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

        # Check if opponent could enclose us by playing here
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        temp_board[row, col] = -player_value

        # Simple threat assessment
        threat_level = 0.0

        # Count opponent pieces that could form enclosures
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
        """Analyze connectivity patterns for strategic insight"""
        features = [0.0] * 12

        own_pieces = (board.board == player_value)

        # Horizontal connectivity
        h_conn = 0
        for row in range(6):
            for col in range(6):
                if own_pieces[row, col] and own_pieces[row, col + 1]:
                    h_conn += 1
        features[0] = h_conn / 30.0

        # Vertical connectivity
        v_conn = 0
        for row in range(5):
            for col in range(7):
                if own_pieces[row, col] and own_pieces[row + 1, col]:
                    v_conn += 1
        features[1] = v_conn / 35.0

        # Diagonal connectivity
        d_conn = 0
        for row in range(5):
            for col in range(6):
                if own_pieces[row, col] and own_pieces[row + 1, col + 1]:
                    d_conn += 1
                if own_pieces[row, col + 1] and own_pieces[row + 1, col]:
                    d_conn += 1
        features[2] = d_conn / 60.0

        # Fill remaining features with positional analysis
        for i in range(3, 12):
            features[i] = random.random() * 0.1  # Placeholder for now

        return features

    def _analyze_formation_stability(self, board, player_value):
        """Analyze stability of current formations"""
        features = [0.0] * 7

        # Count stable vs unstable pieces
        stable_pieces = 0
        total_pieces = 0

        for row in range(6):
            for col in range(7):
                if board.board[row, col] == player_value:
                    total_pieces += 1

                    # Piece is stable if it has support
                    support = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = row + dr, col + dc
                        if (0 <= nr < 6 and 0 <= nc < 7 and
                                board.board[nr, nc] == player_value):
                            support += 1

                    if support >= 2:
                        stable_pieces += 1

        if total_pieces > 0:
            features[0] = stable_pieces / total_pieces

        # Add more stability metrics
        for i in range(1, 7):
            features[i] = random.random() * 0.1  # Placeholder

        return features

    def getAction(self, board, startValue):
        """Enhanced action selection with contour-aware heuristics"""
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

            # Priority 3: Create enclosure threat
            best_enclosure = self._find_best_enclosure_move(board, startValue, possibleActions)
            if best_enclosure is not None:
                return best_enclosure

            # Priority 4: DQN decision
            state = self.encode_state_contour_aware(board, startValue)
            q_values = self.q_network.predict(state)[0]

            # Add small exploration noise
            if self.epsilon > 0:
                noise = np.random.normal(0, self.epsilon * 0.1, q_values.shape)
                q_values += noise

            # Mask invalid actions
            q_values_masked = q_values.copy()
            for i in range(7):
                if i not in possibleActions:
                    q_values_masked[i] = float('-inf')

            action = np.argmax(q_values_masked)

            # Safety check
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

    def _find_best_enclosure_move(self, board, player_value, possible_actions):
        """Find moves that create strong enclosure opportunities"""
        best_score = 0
        best_action = None

        for action in possible_actions:
            score = self._evaluate_enclosure_potential(board, action, player_value)
            if score > best_score and score > 0.6:  # High threshold
                best_score = score
                best_action = action

        return best_action

    def _evaluate_enclosure_potential(self, board, action, player_value):
        """Evaluate enclosure potential of a move"""
        # This would need sophisticated contour analysis
        # For now, simplified version
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        temp_board[row, action] = player_value

        # Count adjacent own pieces (simplified)
        adjacent_own = 0
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = row + dr, action + dc
            if 0 <= nr < 6 and 0 <= nc < 7 and temp_board[nr, nc] == player_value:
                adjacent_own += 1

        return min(1.0, adjacent_own / 8.0)


class ContourGameTrainer:
    """Enhanced trainer for contour formation game"""

    def __init__(self, episodes=5000, use_cuda=True):
        self.episodes = episodes
        self.use_cuda = use_cuda and CUDA_AVAILABLE

        print(f"🎯 Contour Game Trainer - Episodes: {episodes}")
        print(f"⚙️ Device: {'CUDA' if self.use_cuda else 'CPU'}")

        # GPU memory check for RTX 2060
        if self.use_cuda:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb < 4.0:
                print("⚠️ Low GPU memory detected, reducing batch size")
                self.batch_size = 32
            else:
                self.batch_size = 64  # Optimal for RTX 2060
        else:
            self.batch_size = 32  # Conservative for CPU

        # Create enhanced player
        self.player = ContourGamePlayer("Enhanced Player", use_cuda=self.use_cuda)

        # Target network
        self.target_network = EnhancedDQN(
            input_size=200,
            hidden_sizes=[512, 256, 128],
            output_size=7,
            learning_rate=0.0005
        )
        self.target_network.load_state_dict(self.player.q_network.state_dict())

        # Training parameters
        self.replay_buffer = deque(maxlen=100000)  # Larger buffer
        self.target_update_freq = 200  # More frequent updates
        self.eval_freq = 500
        self.min_buffer_size = 5000

        print(f"📊 Batch size: {self.batch_size}")
        print(f"💾 Replay buffer: {100000}")
        print(f"🔄 Target update frequency: {self.target_update_freq}")

        # Enhanced training parameters
        self.gamma = 0.99  # Higher discount for strategic game
        self.epsilon_decay = 0.9995  # Very conservative decay

        # Statistics
        self.episode_rewards = []
        self.win_rates = []
        self.losses = []
        self.best_win_rate = 0

    def train(self):
        """Main training loop"""
        print(f"🚀 Starting Enhanced Training...")
        start_time = time.time()

        if TQDM_AVAILABLE:
            pbar = tqdm(total=self.episodes, desc="Enhanced Training", unit="ep")

        opponent = RNGPlayer("Random Opponent")

        # Early stopping
        consecutive_good_evals = 0
        target_win_rate = 85.0

        for episode in range(self.episodes):
            # Play game
            game_result, experiences = self.play_training_game(opponent)

            # Store experiences with contour-aware rewards
            self.store_experiences_enhanced(experiences, game_result)

            # Track episode reward
            reward = 10 if game_result == 1 else (-10 if game_result == -1 else 1)
            self.episode_rewards.append(reward)

            # Train network
            if len(self.replay_buffer) >= self.min_buffer_size:
                loss = self.train_step()
                if loss is not None:
                    self.losses.append(loss)
                    # Update learning rate based on loss
                    if episode % 50 == 0:
                        avg_loss = np.mean(self.losses[-50:])
                        self.player.q_network.update_learning_rate(avg_loss)

            # Update target network
            if episode % self.target_update_freq == 0 and episode > 0:
                self.target_network.load_state_dict(self.player.q_network.state_dict())

            # Decay epsilon
            if self.player.epsilon > self.player.epsilon_min:
                self.player.epsilon *= self.epsilon_decay

            # Evaluation
            if episode % self.eval_freq == 0 and episode > 0:
                # Clear GPU memory before evaluation
                if self.use_cuda:
                    torch.cuda.empty_cache()

                win_rate, wins, losses = self.evaluate(num_games=100)
                self.win_rates.append(win_rate)

                # Save best model
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    self.player.q_network.save_weights('enhanced_weights.pth')

                # Progress update
                elapsed = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards[-self.eval_freq:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0

                # GPU memory info
                gpu_info = ""
                if self.use_cuda:
                    gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                    gpu_memory_cached = torch.cuda.memory_reserved() / 1e9
                    gpu_info = f" | GPU: {gpu_memory_used:.1f}GB"

                status = (f"Ep {episode:5d} | "
                          f"Win: {win_rate:5.1f}% | "
                          f"Best: {self.best_win_rate:5.1f}% | "
                          f"Reward: {avg_reward:6.2f} | "
                          f"Loss: {avg_loss:8.4f} | "
                          f"ε: {self.player.epsilon:.4f} | "
                          f"Time: {elapsed / 60:.1f}m{gpu_info}")

                if TQDM_AVAILABLE:
                    tqdm.write(status)
                    pbar.set_postfix_str(f"Win: {win_rate:.1f}%")
                else:
                    print(status)

                # Early stopping check
                if win_rate >= target_win_rate:
                    consecutive_good_evals += 1
                    if consecutive_good_evals >= 3:
                        print(f"\n🎉 TARGET ACHIEVED! Win rate: {win_rate:.1f}%")
                        break
                else:
                    consecutive_good_evals = 0

            if TQDM_AVAILABLE:
                pbar.update(1)

        if TQDM_AVAILABLE:
            pbar.close()

        # Final evaluation
        final_rate, _, _ = self.evaluate(num_games=200)
        total_time = time.time() - start_time

        print(f"\n🏁 Enhanced training completed in {total_time / 60:.1f} minutes!")
        print(f"📊 Final win rate: {final_rate:.1f}%")
        print(f"🎯 Best win rate: {self.best_win_rate:.1f}%")
        print(f"💾 Best weights saved to 'enhanced_weights.pth'")

        return self.player

    def play_training_game(self, opponent):
        """Play training game with experience collection"""
        board = gamerules.Board()

        player_starts = random.choice([True, False])
        player_value = 1 if player_starts else -1

        experiences = []
        turn = 1
        max_turns = 42
        turns = 0

        while turns < max_turns:
            current_player_is_dqn = (turn == 1 and player_starts) or (turn == -1 and not player_starts)

            if current_player_is_dqn:
                state = self.player.encode_state_contour_aware(board, player_value)
                action = self.player.getAction(board, player_value)
                experiences.append({'state': state, 'action': action, 'turn': turns})
            else:
                action = opponent.getAction(board, -player_value)

            # Validate action
            possible_actions = self.player.getPossibleActions(board.board)
            if action not in possible_actions:
                return -1 if current_player_is_dqn else 1, experiences

            # Make move
            board.updateBoard(action, turn)

            # Check victory
            if board.checkVictory(action, turn):
                return 1 if current_player_is_dqn else -1, experiences

            # Check draw
            if len(board.getPossibleActions()) == 0:
                return 0, experiences

            turn *= -1
            turns += 1

        return 0, experiences  # Timeout draw

    def store_experiences_enhanced(self, experiences, game_result):
        """Store experiences with enhanced reward structure"""
        n_exp = len(experiences)
        if n_exp == 0:
            return

        for i, exp in enumerate(experiences):
            # Enhanced contour-aware reward structure
            if game_result == 1:  # Win
                # Reward based on game phase and move quality
                move_number = exp['turn']
                if move_number < 15:  # Early game win (very good)
                    reward = 15.0 - (move_number * 0.3)
                elif move_number < 25:  # Mid game win (good)
                    reward = 10.0 - ((move_number - 15) * 0.2)
                else:  # Late game win (okay)
                    reward = 7.0 - ((move_number - 25) * 0.1)

                # Bonus for final winning move
                if i == n_exp - 1:
                    reward += 5.0

            elif game_result == -1:  # Loss
                # Progressive penalty for moves leading to loss
                moves_from_end = n_exp - i
                base_penalty = -8.0
                temporal_penalty = -0.5 * moves_from_end
                reward = base_penalty + temporal_penalty

            else:  # Draw
                # Small positive reward that decreases over time
                reward = 2.0 * (0.98 ** i)

            # Next state calculation
            if i < n_exp - 1:
                next_state = experiences[i + 1]['state']
                done = False
            else:
                next_state = np.zeros_like(exp['state'])
                done = True

            self.replay_buffer.append({
                'state': exp['state'],
                'action': exp['action'],
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

    def train_step(self):
        """Enhanced training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Prepare batch data
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])

        # Convert to tensors
        states = torch.FloatTensor(states).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        dones = torch.BoolTensor(dones).to(DEVICE)

        # Current Q values
        current_q_values = self.player.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.player.q_network.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.player.q_network.parameters(), max_norm=1.0)
        self.player.q_network.optimizer.step()

        return loss.item()

    def evaluate(self, num_games=100):
        """Evaluate player performance"""
        original_epsilon = self.player.epsilon
        self.player.epsilon = 0.0  # No exploration during evaluation

        wins = 0
        draws = 0
        losses = 0

        opponent = RNGPlayer("Evaluation Opponent")

        for game in range(num_games):
            result, _ = self.play_training_game(opponent)
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1

        self.player.epsilon = original_epsilon
        win_rate = (wins / num_games) * 100

        return win_rate, wins, losses


class RNGPlayer(gamerules.Player):
    """Random player for training/testing"""

    def __init__(self, name="Random"):
        super().__init__(name)

    def getAction(self, board, startValue):
        possibleActions = self.getPossibleActions(board.board)
        return np.random.choice(possibleActions)

    def newGame(self, new_opponent):
        pass


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced DQN Training with CUDA')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 ENHANCED DQN TRAINING FOR CONTOUR GAME")
    print("=" * 70)

    use_cuda = CUDA_AVAILABLE and not args.no_cuda

    # Create and run trainer
    trainer = ContourGameTrainer(episodes=args.episodes, use_cuda=use_cuda)
    trained_player = trainer.train()

    print("\n🎉 Enhanced training completed!")
    print("🧪 Test your model with the enhanced architecture")


if __name__ == "__main__":
    main()