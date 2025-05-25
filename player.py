import numpy as np
import pickle
import os
import gamerules


class CustomDQN:
    """Custom Deep Q-Network implementation with enhanced stability features"""

    def __init__(self, input_size=150, hidden_sizes=[128, 64], output_size=7, learning_rate=0.0001,
                 min_learning_rate=1e-6, learning_rate_patience=5):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.no_improvement_count = 0
        self.best_loss = float('inf')

        # Initialize network architecture with batch normalization
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Initialize batch normalization parameters
        self.bn_params = []

        for i in range(len(layer_sizes) - 1):
            std = np.sqrt(2.0 / layer_sizes[i])  # He initialization
            layer = {
                'weights': np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * std,
                'biases': np.zeros((1, layer_sizes[i + 1])),
                'weights_momentum': np.zeros((layer_sizes[i], layer_sizes[i + 1])),
                'biases_momentum': np.zeros((1, layer_sizes[i + 1])),
                'weights_velocity': np.zeros((layer_sizes[i], layer_sizes[i + 1])),
                'biases_velocity': np.zeros((1, layer_sizes[i + 1])),
                'dropout_mask': None
            }
            self.layers.append(layer)

            # Add batch normalization parameters for hidden layers
            if i < len(layer_sizes) - 2:  # Not for output layer
                self.bn_params.append({
                    'gamma': np.ones((1, layer_sizes[i + 1])),
                    'beta': np.zeros((1, layer_sizes[i + 1])),
                    'running_mean': np.zeros((1, layer_sizes[i + 1])),
                    'running_var': np.ones((1, layer_sizes[i + 1])),
                    'momentum': 0.99
                })

        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

        # Dropout rate
        self.dropout_rate = 0.2
        self.is_training = True

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def batch_normalize(self, x, bn_param, layer_idx):
        """Apply batch normalization"""
        if self.is_training:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True) + 1e-8

            # Update running mean and variance
            bn_param['running_mean'] = bn_param['momentum'] * bn_param['running_mean'] + \
                                       (1 - bn_param['momentum']) * mu
            bn_param['running_var'] = bn_param['momentum'] * bn_param['running_var'] + \
                                      (1 - bn_param['momentum']) * var
        else:
            mu = bn_param['running_mean']
            var = bn_param['running_var']

        # Normalize
        x_norm = (x - mu) / np.sqrt(var + 1e-8)

        # Scale and shift
        out = bn_param['gamma'] * x_norm + bn_param['beta']

        if self.is_training:
            # Cache for backward pass
            bn_param['x_norm'] = x_norm
            bn_param['x_centered'] = x - mu
            bn_param['std_inv'] = 1.0 / np.sqrt(var + 1e-8)

        return out

    def dropout(self, x, layer_idx):
        """Apply dropout during training"""
        if not self.is_training:
            return x

        mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) / (1 - self.dropout_rate)
        self.layers[layer_idx]['dropout_mask'] = mask
        return x * mask

    def forward(self, x):
        """Forward pass with batch normalization and dropout"""
        self.activations = [x]
        self.z_values = []

        current_input = x

        for i, layer in enumerate(self.layers):
            # Linear transformation
            z = np.dot(current_input, layer['weights']) + layer['biases']
            self.z_values.append(z)

            # Apply batch normalization and activation for hidden layers
            if i < len(self.layers) - 1:
                # Batch normalization
                z = self.batch_normalize(z, self.bn_params[i], i)
                # ReLU activation
                activation = self.relu(z)
                # Dropout
                activation = self.dropout(activation, i)
            else:
                activation = z  # Linear output for Q-values

            self.activations.append(activation)
            current_input = activation

        return self.activations[-1]

    def backward(self, y_true, y_pred):
        """Backpropagation to compute gradients"""
        batch_size = y_true.shape[0]

        # Compute output layer error (MSE loss derivative)
        delta = (y_pred - y_true) / batch_size

        # Store gradients
        gradients = []

        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            # Compute gradients for weights and biases
            weights_grad = np.dot(self.activations[i].T, delta)
            biases_grad = np.sum(delta, axis=0, keepdims=True)

            gradients.insert(0, {
                'weights': weights_grad,
                'biases': biases_grad
            })

            # Compute delta for previous layer (except for input layer)
            if i > 0:
                # Gradient of loss w.r.t. activation of previous layer
                delta = np.dot(delta, self.layers[i]['weights'].T)
                # Apply derivative of activation function
                delta = delta * self.relu_derivative(self.z_values[i - 1])

        return gradients

    def update_weights(self, gradients):
        """Update weights using Adam optimizer"""
        self.t += 1

        for i, (layer, grad) in enumerate(zip(self.layers, gradients)):
            # Update momentum (first moment)
            layer['weights_momentum'] = self.beta1 * layer['weights_momentum'] + (1 - self.beta1) * grad['weights']
            layer['biases_momentum'] = self.beta1 * layer['biases_momentum'] + (1 - self.beta1) * grad['biases']

            # Update velocity (second moment)
            layer['weights_velocity'] = self.beta2 * layer['weights_velocity'] + (1 - self.beta2) * (
                    grad['weights'] ** 2)
            layer['biases_velocity'] = self.beta2 * layer['biases_velocity'] + (1 - self.beta2) * (grad['biases'] ** 2)

            # Bias correction
            weights_momentum_corrected = layer['weights_momentum'] / (1 - self.beta1 ** self.t)
            biases_momentum_corrected = layer['biases_momentum'] / (1 - self.beta1 ** self.t)
            weights_velocity_corrected = layer['weights_velocity'] / (1 - self.beta2 ** self.t)
            biases_velocity_corrected = layer['biases_velocity'] / (1 - self.beta2 ** self.t)

            # Update parameters
            layer['weights'] -= self.learning_rate * weights_momentum_corrected / (
                    np.sqrt(weights_velocity_corrected) + self.epsilon)
            layer['biases'] -= self.learning_rate * biases_momentum_corrected / (
                    np.sqrt(biases_velocity_corrected) + self.epsilon)

    def predict(self, x):
        """Predict Q-values for given state(s)"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return self.forward(x)

    def copy_weights_from(self, other_network):
        """Copy weights from another network (for target network updates)"""
        for i, (self_layer, other_layer) in enumerate(zip(self.layers, other_network.layers)):
            self_layer['weights'] = other_layer['weights'].copy()
            self_layer['biases'] = other_layer['biases'].copy()

    def save_weights(self, filepath):
        """Save network weights to file"""
        weights_data = {
            'layers': [],
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate
        }

        for layer in self.layers:
            weights_data['layers'].append({
                'weights': layer['weights'],
                'biases': layer['biases']
            })

        with open(filepath, 'wb') as f:
            pickle.dump(weights_data, f)

    def load_weights(self, filepath):
        """Load network weights from file"""
        try:
            with open(filepath, 'rb') as f:
                weights_data = pickle.load(f)

            # Verify architecture compatibility (relaxed check)
            if (weights_data['input_size'] != self.input_size or
                    weights_data['output_size'] != self.output_size):
                print("Warning: Architecture mismatch (input/output), using random weights")
                return False

            # Allow different hidden layer architectures for flexibility
            if len(weights_data['layers']) != len(self.layers):
                print("Warning: Different number of layers, using random weights")
                return False

            # Load weights and biases
            for i, layer_data in enumerate(weights_data['layers']):
                # Check if layer shapes match
                if (layer_data['weights'].shape != self.layers[i]['weights'].shape or
                        layer_data['biases'].shape != self.layers[i]['biases'].shape):
                    print(f"Warning: Layer {i} shape mismatch, using random weights")
                    return False

                self.layers[i]['weights'] = layer_data['weights']
                self.layers[i]['biases'] = layer_data['biases']

            return True

        except Exception as e:
            print(f"Error loading weights: {e}")
            return False

    def adjust_learning_rate(self, loss):
        """Adjust learning rate based on loss improvement with moving average"""
        # Initialize loss history if not exists
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
            self.best_loss = None
            self.best_loss_epoch = 0
            self.current_epoch = 0
            self.window_best_loss = float('inf')
            self.last_lr_update = 0

        self.current_epoch += 1

        # Add current loss to history
        self.loss_history.append(loss)

        # Keep only last 100 losses
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)

        # Calculate average loss over a window
        window_size = min(50, len(self.loss_history))
        if window_size < 10:  # Need minimum history
            return False

        recent_losses = self.loss_history[-window_size:]
        avg_loss = np.mean(recent_losses)

        # Initialize best loss if needed
        if self.best_loss is None:
            self.best_loss = avg_loss
            return False

        # Calculate relative improvement safely
        if avg_loss < 1e-10 and self.best_loss < 1e-10:
            rel_improvement = 0.0
        else:
            rel_improvement = (self.best_loss - avg_loss) / max(self.best_loss, avg_loss)

        # Update window best loss
        if avg_loss < self.window_best_loss:
            self.window_best_loss = avg_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Update best loss if significant improvement
        if rel_improvement > 0.01:  # 1% improvement threshold
            self.best_loss = avg_loss
            self.best_loss_epoch = self.current_epoch
            self.no_improvement_count = 0
            return False

        # Only adjust learning rate if enough epochs have passed since last update
        epochs_since_update = self.current_epoch - self.last_lr_update
        if epochs_since_update < 20:  # Minimum epochs between updates
            return False

        # Only adjust if we have consistent lack of improvement
        if self.no_improvement_count >= self.learning_rate_patience:
            # More gradual reduction based on how long since last improvement
            epochs_since_improvement = self.current_epoch - self.best_loss_epoch
            reduction_factor = max(0.9, 0.95 ** (epochs_since_improvement // 20))
            new_lr = max(self.learning_rate * reduction_factor, self.min_learning_rate)

            # Only update if significant change
            if new_lr < self.learning_rate * 0.95:
                self.learning_rate = new_lr
                self.last_lr_update = self.current_epoch
                self.no_improvement_count = 0
                self.window_best_loss = float('inf')  # Reset window best
                return True

        return False

    def train_step(self, state, target):
        """Perform a single training step"""
        # Set training mode
        self.is_training = True

        # Forward pass
        predicted = self.forward(state)

        # Compute gradients
        gradients = self.backward(target, predicted)

        # Update weights
        self.update_weights(gradients)

        # Calculate loss (MSE)
        loss = np.mean((predicted - target) ** 2)

        # Set back to evaluation mode
        self.is_training = False

        return loss

    def state_dict(self):
        """Get network state for saving best model"""
        return {
            'layers': [{
                'weights': layer['weights'].copy(),
                'biases': layer['biases'].copy()
            } for layer in self.layers],
            'bn_params': [{
                'gamma': param['gamma'].copy(),
                'beta': param['beta'].copy(),
                'running_mean': param['running_mean'].copy(),
                'running_var': param['running_var'].copy()
            } for param in self.bn_params] if hasattr(self, 'bn_params') else []
        }

    def load_state_dict(self, state_dict):
        """Load network state from best model"""
        for i, layer_state in enumerate(state_dict['layers']):
            self.layers[i]['weights'] = layer_state['weights'].copy()
            self.layers[i]['biases'] = layer_state['biases'].copy()

        if 'bn_params' in state_dict and hasattr(self, 'bn_params'):
            for i, bn_state in enumerate(state_dict['bn_params']):
                self.bn_params[i]['gamma'] = bn_state['gamma'].copy()
                self.bn_params[i]['beta'] = bn_state['beta'].copy()
                self.bn_params[i]['running_mean'] = bn_state['running_mean'].copy()
                self.bn_params[i]['running_var'] = bn_state['running_var'].copy()


class HeuristicEngine:
    """Moderate complexity heuristic engine for tactical decision making"""

    def __init__(self):
        # Center preference weights (higher is better)
        self.column_weights = [0.5, 0.7, 0.85, 1.0, 0.85, 0.7, 0.5]

    def find_immediate_win(self, board, player_value):
        """Priority 1: Find immediate winning move"""
        possible_actions = self._get_possible_actions(board)

        for action in possible_actions:
            if self._can_win_with_move(board, action, player_value):
                return action, "IMMEDIATE_WIN"
        return None, None

    def find_must_block(self, board, player_value):
        """Priority 2: Block opponent's immediate win"""
        possible_actions = self._get_possible_actions(board)
        opponent_value = -player_value

        for action in possible_actions:
            if self._can_win_with_move(board, action, opponent_value):
                return action, "MUST_BLOCK"
        return None, None

    def find_winning_threat(self, board, player_value):
        """Priority 3: Create winning threat (offensive focus)"""
        possible_actions = self._get_possible_actions(board)
        best_action = None
        best_score = 0

        for action in possible_actions:
            score = self._evaluate_threat_creation(board, action, player_value)
            if score > best_score and score >= 0.7:  # High threshold for override
                best_score = score
                best_action = action

        if best_action is not None:
            return best_action, "CREATE_THREAT"
        return None, None

    def find_dangerous_block(self, board, player_value):
        """Priority 4: Block dangerous opponent threats"""
        possible_actions = self._get_possible_actions(board)
        opponent_value = -player_value
        best_action = None
        best_threat_level = 0

        for action in possible_actions:
            threat_level = self._evaluate_opponent_threat_level(board, action, opponent_value)
            if threat_level > best_threat_level and threat_level >= 0.8:  # High threshold
                best_threat_level = threat_level
                best_action = action

        if best_action is not None:
            return best_action, "BLOCK_THREAT"
        return None, None

    def get_positional_preference(self, board, player_value):
        """Priority 6: Positional preferences for DQN tiebreaking"""
        possible_actions = self._get_possible_actions(board)

        if not possible_actions:
            return None, None

        # Score each action based on positional factors
        action_scores = []
        for action in possible_actions:
            score = self._evaluate_positional_move(board, action, player_value)
            action_scores.append((action, score))

        # Sort by score (highest first)
        action_scores.sort(key=lambda x: x[1], reverse=True)

        # Return best action if it's significantly better than random
        best_action, best_score = action_scores[0]
        if best_score > 0.3:  # Reasonable threshold
            return best_action, "POSITIONAL"

        return None, None

    def _get_possible_actions(self, board):
        """Get list of possible actions"""
        return np.unique(np.where(board.board == 0)[1]).tolist()

    def _can_win_with_move(self, board, action, player_value):
        """Check if a move results in immediate win"""
        # Simulate the move
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return False

        row = np.max(empty_rows)
        temp_board[row, action] = player_value

        # Create temporary board object to check victory
        temp_board_obj = gamerules.Board()
        temp_board_obj.board = temp_board
        temp_board_obj.components = board.components.copy()
        temp_board_obj.components4 = board.components4.copy()
        temp_board_obj.updateComponents(row, action, player_value)
        temp_board_obj.updateComponents4(row, action, player_value)

        return temp_board_obj.checkVictory(action, player_value)

    def _evaluate_threat_creation(self, board, action, player_value):
        """Evaluate how good a move is for creating threats (0.0-1.0)"""
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        temp_board[row, action] = player_value

        score = 0.0

        # Check if this move creates multiple winning opportunities
        threats_created = 0
        for next_action in self._get_possible_actions(board):
            if next_action == action:
                continue
            # Simulate opponent's move
            temp_board2 = temp_board.copy()
            empty_rows2 = np.where(temp_board2[:, next_action] == 0)[0]
            if len(empty_rows2) > 0:
                row2 = np.max(empty_rows2)
                temp_board2[row2, next_action] = -player_value

                # Check if we can still win after opponent's move
                temp_board_obj = gamerules.Board()
                temp_board_obj.board = temp_board2
                for final_action in self._get_possible_actions(board):
                    if self._would_win_after_move(temp_board_obj, final_action, player_value):
                        threats_created += 1
                        break

        # Multiple threats = strong position
        if threats_created >= 2:
            score += 0.8
        elif threats_created == 1:
            score += 0.5

        # Check for formation building
        score += self._evaluate_formation_strength(temp_board, row, action, player_value) * 0.3

        # Bonus for center play
        score += self.column_weights[action] * 0.2

        return min(1.0, score)

    def _evaluate_opponent_threat_level(self, board, action, opponent_value):
        """Evaluate threat level if opponent plays this move (0.0-1.0)"""
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        temp_board[row, action] = opponent_value

        # Check how many ways opponent can win after this move
        win_opportunities = 0
        for next_action in self._get_possible_actions(board):
            if next_action == action:
                continue
            temp_board_obj = gamerules.Board()
            temp_board_obj.board = temp_board.copy()
            if self._would_win_after_move(temp_board_obj, next_action, opponent_value):
                win_opportunities += 1

        # Multiple winning opportunities = dangerous
        if win_opportunities >= 2:
            return 1.0
        elif win_opportunities == 1:
            return 0.6

        # Check formation strength
        formation_strength = self._evaluate_formation_strength(temp_board, row, action, opponent_value)
        return min(0.8, formation_strength)

    def _would_win_after_move(self, board, action, player_value):
        """Check if a move would result in win"""
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return False

        row = np.max(empty_rows)
        temp_board[row, action] = player_value

        temp_board_obj = gamerules.Board()
        temp_board_obj.board = temp_board
        temp_board_obj.components = board.components.copy()
        temp_board_obj.components4 = board.components4.copy()
        temp_board_obj.updateComponents(row, action, player_value)
        temp_board_obj.updateComponents4(row, action, player_value)

        return temp_board_obj.checkVictory(action, player_value)

    def _evaluate_formation_strength(self, board, row, col, player_value):
        """Evaluate strength of formation created by move (0.0-1.0)"""
        score = 0.0

        # Count adjacent same-color pieces
        adjacent_count = 0
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 6 and 0 <= nc < 7 and board[nr, nc] == player_value:
                adjacent_count += 1

        # More adjacent pieces = stronger formation
        score += min(0.6, adjacent_count * 0.1)

        # Check for line potential (simplified)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1)]  # Four main directions
        for dr, dc in directions:
            line_length = 1  # Current piece

            # Count in positive direction
            nr, nc = row + dr, col + dc
            while 0 <= nr < 6 and 0 <= nc < 7 and board[nr, nc] == player_value:
                line_length += 1
                nr, nc = nr + dr, nc + dc

            # Count in negative direction
            nr, nc = row - dr, col - dc
            while 0 <= nr < 6 and 0 <= nc < 7 and board[nr, nc] == player_value:
                line_length += 1
                nr, nc = nr - dr, nc - dc

            # Longer lines are better
            if line_length >= 3:
                score += 0.4
            elif line_length >= 2:
                score += 0.2

        return min(1.0, score)

    def _evaluate_positional_move(self, board, action, player_value):
        """Evaluate positional quality of a move (0.0-1.0)"""
        score = 0.0

        # Base score from column preference (center better)
        score += self.column_weights[action] * 0.4

        # Avoid filling up columns too early
        column_height = 6 - len(np.where(board.board[:, action] == 0)[0])
        if column_height >= 5:
            score *= 0.3  # Heavy penalty for nearly full columns
        elif column_height >= 4:
            score *= 0.7  # Moderate penalty

        # Small bonus for building on own pieces
        if column_height > 0 and board.board[6 - column_height, action] == player_value:
            score += 0.2

        # Penalty for helping opponent
        if column_height > 0 and board.board[6 - column_height, action] == -player_value:
            score *= 0.8

        return min(1.0, score)


class Player(gamerules.Player):
    """Enhanced Hybrid Player: DQN + Heuristic Overrides with Offensive Focus"""

    def __init__(self, name, weights_file=None):
        super().__init__(name)
        self.name = name
        self.weights_file = weights_file

        # Initialize enhanced custom DQN network
        self.q_network = CustomDQN(
            input_size=200,
            hidden_sizes=[256, 128, 64],
            output_size=7,
            learning_rate=0.0001
        )

        # Initialize heuristic engine
        self.heuristics = HeuristicEngine()

        # Statistics for analysis
        self.decision_stats = {
            'dqn_decisions': 0,
            'heuristic_overrides': 0,
            'override_types': {}
        }

        # Load weights if provided
        if weights_file and os.path.exists(weights_file):
            if self.q_network.load_weights(weights_file):
                print(f"✅ Loaded DQN weights from {weights_file}")
            else:
                print("⚠️ Failed to load weights, using random weights")
        else:
            print("ℹ️ No weights file found, using random DQN weights")

    def getName(self):
        return self.name

    def newGame(self, new_opponent):
        """Called at the beginning of each game"""
        pass

    def getAction(self, board, startValue):
        """
        Hybrid Override Decision System:
        1. 🏆 IMMEDIATE WIN (100% priority)
        2. 🛡️ BLOCK OPPONENT WIN (100% priority)
        3. ⚔️ CREATE WINNING THREAT (override if strong)
        4. 🎯 BLOCK OPPONENT THREAT (override if dangerous)
        5. 🧠 DQN DECISION (for complex positions)
        6. 📍 POSITIONAL PREFERENCE (tiebreaker)
        """
        possibleActions = self.getPossibleActions(board.board)

        if len(possibleActions) == 0:
            return 0  # Fallback

        try:
            # Priority 1: IMMEDIATE WIN - Always override
            action, reason = self.heuristics.find_immediate_win(board, startValue)
            if action is not None:
                self._record_decision(reason)
                return action

            # Priority 2: MUST BLOCK - Always override
            action, reason = self.heuristics.find_must_block(board, startValue)
            if action is not None:
                self._record_decision(reason)
                return action

            # Priority 3: CREATE WINNING THREAT - Override if strong (offensive focus)
            action, reason = self.heuristics.find_winning_threat(board, startValue)
            if action is not None:
                self._record_decision(reason)
                return action

            # Priority 4: BLOCK DANGEROUS THREAT - Override if very dangerous
            action, reason = self.heuristics.find_dangerous_block(board, startValue)
            if action is not None:
                self._record_decision(reason)
                return action

            # Priority 5: DQN DECISION - For complex strategic positions
            dqn_action = self._get_dqn_action(board, startValue, possibleActions)

            # Priority 6: POSITIONAL PREFERENCE - Enhance DQN with positional wisdom
            pos_action, pos_reason = self.heuristics.get_positional_preference(board, startValue)
            if pos_action is not None and dqn_action in possibleActions:
                # Use positional preference to break ties or improve poor DQN choices
                dqn_q_values = self.q_network.predict(self._encode_state_enhanced(board, startValue))[0]
                dqn_confidence = abs(dqn_q_values[dqn_action])

                # If DQN confidence is low, use positional preference
                if dqn_confidence < 0.1:  # Low confidence threshold
                    self._record_decision(pos_reason)
                    return pos_action

            # Default to DQN decision
            self._record_decision("DQN")
            return dqn_action

        except Exception as e:
            print(f"Error in hybrid decision system: {e}")
            # Emergency fallback: center preference
            center_preferences = [3, 2, 4, 1, 5, 0, 6]
            for col in center_preferences:
                if col in possibleActions:
                    self._record_decision("EMERGENCY_FALLBACK")
                    return col
            return possibleActions[0]

    def _get_dqn_action(self, board, startValue, possibleActions):
        """Get action from DQN with safety checks"""
        try:
            # Encode state with enhanced features
            state = self._encode_state_enhanced(board, startValue)

            # Get Q-values from network
            q_values = self.q_network.predict(state)[0]

            # Add small exploration noise for non-deterministic play
            exploration_noise = np.random.normal(0, 0.005, size=q_values.shape)
            q_values += exploration_noise

            # Mask invalid actions
            q_values_masked = q_values.copy()
            for i in range(7):
                if i not in possibleActions:
                    q_values_masked[i] = float('-inf')

            # Select best action
            action = np.argmax(q_values_masked)

            # Safety check
            if action not in possibleActions:
                action = np.random.choice(possibleActions)

            return int(action)

        except Exception as e:
            print(f"Error in DQN action selection: {e}")
            return np.random.choice(possibleActions)

    def _record_decision(self, decision_type):
        """Record decision statistics for analysis"""
        if decision_type == "DQN":
            self.decision_stats['dqn_decisions'] += 1
        else:
            self.decision_stats['heuristic_overrides'] += 1
            if decision_type not in self.decision_stats['override_types']:
                self.decision_stats['override_types'][decision_type] = 0
            self.decision_stats['override_types'][decision_type] += 1

    def get_decision_stats(self):
        """Get statistics about decision making"""
        total_decisions = self.decision_stats['dqn_decisions'] + self.decision_stats['heuristic_overrides']
        if total_decisions == 0:
            return {}

        stats = {
            'total_decisions': total_decisions,
            'dqn_percentage': (self.decision_stats['dqn_decisions'] / total_decisions) * 100,
            'heuristic_percentage': (self.decision_stats['heuristic_overrides'] / total_decisions) * 100,
            'override_breakdown': {}
        }

        for override_type, count in self.decision_stats['override_types'].items():
            stats['override_breakdown'][override_type] = {
                'count': count,
                'percentage': (count / total_decisions) * 100
            }

        return stats

    def _encode_state_enhanced(self, board, startValue):
        """Enhanced state encoding with advanced game analysis"""
        features = []

        # Basic board state (42 features)
        board_normalized = board.board * startValue
        features.extend(board_normalized.flatten())

        # Component information (84 features)
        components_normalized = np.sign(board.components) * startValue
        features.extend(components_normalized.flatten())

        components4_normalized = np.sign(board.components4) * startValue
        features.extend(components4_normalized.flatten())

        # Enhanced strategic features (74 features to reach 200 total)
        strategic_features = self._extract_advanced_features(board, startValue)
        features.extend(strategic_features)

        return np.array(features, dtype=np.float32)

    def _extract_advanced_features(self, board, startValue):
        """Extract advanced game-specific features"""
        features = []

        # === BASIC STRATEGIC FEATURES (16 features) ===
        column_heights = []
        for col in range(7):
            height = 6 - len(np.where(board.board[:, col] == 0)[0])
            column_heights.append(height)
            features.append(height / 6.0)

        center_control = sum(np.sum(board.board[:, col] == startValue) for col in [2, 3, 4])
        features.append(center_control / 18.0)

        edge_control = sum(np.sum(board.board[:, col] == startValue) for col in [0, 6])
        features.append(edge_control / 12.0)

        possible_actions = len(self.getPossibleActions(board.board))
        features.append(possible_actions / 7.0)

        total_pieces = np.sum(board.board != 0)
        features.append(total_pieces / 42.0)

        if total_pieces < 14:
            features.extend([1.0, 0.0, 0.0])
        elif total_pieces < 28:
            features.extend([0.0, 1.0, 0.0])
        else:
            features.extend([0.0, 0.0, 1.0])

        features.append(1.0 if startValue == 1 else 0.0)

        left_pieces = np.sum(board.board[:, :3] == startValue)
        right_pieces = np.sum(board.board[:, 4:] == startValue)
        total_own = left_pieces + right_pieces
        balance = 1.0 - abs(left_pieces - right_pieces) / max(total_own, 1)
        features.append(balance)

        # === ADVANCED PATTERN RECOGNITION (21 features) ===
        for col in range(7):
            can_win = self._can_win_in_column(board, col, startValue)
            features.append(1.0 if can_win else 0.0)

        for col in range(7):
            must_block = self._must_block_column(board, col, startValue)
            features.append(1.0 if must_block else 0.0)

        for col in range(7):
            threat_level = self._analyze_threats_in_column(board, col, startValue)
            features.append(threat_level)

        # === COMPONENT AND FORMATION ANALYSIS (14 features) ===
        own_component_sizes = self._analyze_component_sizes(board, startValue)
        features.extend(own_component_sizes)

        opp_component_sizes = self._analyze_component_sizes(board, -startValue)
        features.extend(opp_component_sizes)

        # === POSITIONAL EVALUATION (14 features) ===
        corners = [(0, 0), (0, 6), (5, 0), (5, 6)]
        for row, col in corners:
            if board.board[row, col] == startValue:
                features.append(1.0)
            elif board.board[row, col] == -startValue:
                features.append(-1.0)
            else:
                features.append(0.0)

        edge_positions = [(0, 3), (5, 3), (2, 0), (2, 6)]
        for row, col in edge_positions:
            if board.board[row, col] == startValue:
                features.append(1.0)
            elif board.board[row, col] == -startValue:
                features.append(-1.0)
            else:
                features.append(0.0)

        own_connectivity = self._calculate_connectivity(board, startValue)
        opp_connectivity = self._calculate_connectivity(board, -startValue)
        features.append(own_connectivity / 50.0)
        features.append(opp_connectivity / 50.0)

        center_own = np.sum(board.board[:, 3] == startValue)
        center_opp = np.sum(board.board[:, 3] == -startValue)
        features.append(center_own / 6.0)
        features.append(center_opp / 6.0)

        main_diag_own = sum(1 for i in range(min(6, 7)) if i < 6 and i < 7 and board.board[i, i] == startValue)
        anti_diag_own = sum(
            1 for i in range(min(6, 7)) if i < 6 and (6 - 1 - i) < 7 and board.board[i, 6 - 1 - i] == startValue)
        features.append(main_diag_own / 6.0)
        features.append(anti_diag_own / 6.0)

        # === TACTICAL FEATURES (9 features) ===
        for col in range(7):
            pressure = self._calculate_column_pressure(board, col, startValue, column_heights[col])
            features.append(pressure)

        mobility = self._calculate_mobility(board, startValue)
        tempo = self._calculate_tempo_advantage(board, startValue)
        features.append(mobility)
        features.append(tempo)

        return features

    def _can_win_in_column(self, board, col, player_value):
        """Check if player can win immediately by playing in this column"""
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

    def _must_block_column(self, board, col, player_value):
        """Check if opponent can win if we don't block this column"""
        return self._can_win_in_column(board, col, -player_value)

    def _analyze_threats_in_column(self, board, col, player_value):
        """Analyze threat level in a column (0.0 to 1.0)"""
        if col not in self.getPossibleActions(board.board):
            return 0.0

        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)

        connections = 0
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 6 and 0 <= nc < 7:
                if temp_board[nr, nc] == player_value:
                    connections += 1
                elif temp_board[nr, nc] == -player_value:
                    connections -= 0.5

        return max(0.0, min(1.0, connections / 8.0))

    def _analyze_component_sizes(self, board, player_value):
        """Analyze component sizes for a player (returns 7 features)"""
        features = [0.0] * 7

        components = board.components * (np.sign(board.components) == np.sign(player_value))
        if np.any(components):
            unique_components = components[components != 0]
            if len(unique_components) > 0:
                sizes = np.bincount(np.abs(unique_components.astype(int)))
                for size in sizes[1:]:
                    if size <= 7:
                        features[size - 1] = min(1.0, features[size - 1] + 0.2)

        return features

    def _calculate_connectivity(self, board, player_value):
        """Calculate how well connected a player's pieces are"""
        connectivity = 0

        for row in range(6):
            for col in range(7):
                if board.board[row, col] == player_value:
                    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < 6 and 0 <= nc < 7 and board.board[nr, nc] == player_value:
                            connectivity += 1

        return connectivity

    def _calculate_column_pressure(self, board, col, player_value, height):
        """Calculate pressure/importance of a column"""
        pressure = 0.0

        if height >= 4:
            pressure += 0.3
        if height >= 5:
            pressure += 0.4

        center_bonus = max(0, 1.0 - abs(col - 3) * 0.2)
        pressure += center_bonus * 0.3

        return pressure

    def _calculate_mobility(self, board, player_value):
        """Calculate mobility advantage"""
        possible_actions = len(self.getPossibleActions(board.board))
        return possible_actions / 7.0

    def _calculate_tempo_advantage(self, board, player_value):
        """Calculate tempo/initiative advantage"""
        own_pieces = np.sum(board.board == player_value)
        opp_pieces = np.sum(board.board == -player_value)

        if own_pieces + opp_pieces == 0:
            return 0.5

        return own_pieces / (own_pieces + opp_pieces)

    def getPossibleActions(self, board):
        """Get possible actions from board"""
        return np.unique(np.where(board == 0)[1])