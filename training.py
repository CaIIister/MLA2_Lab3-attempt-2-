import numpy as np
import time
import copy
from collections import defaultdict
import gamerules
from player import Player

try:
    import cupy as cp

    CUDA_AVAILABLE = True
    print("CUDA acceleration available")
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA not available, using CPU")


class RNGPlayer(gamerules.Player):
    """Random player for training and testing"""

    def __init__(self, name):
        super().__init__(name)

    def getAction(self, board, startValue):
        possibleActions = self.getPossibleActions(board.board)
        return np.random.choice(possibleActions)

    def newGame(self, new_opponent):
        pass


class QLearningTrainer:
    def __init__(self, cuda_batch_size=100):
        self.cuda_batch_size = cuda_batch_size if CUDA_AVAILABLE else 1

    def simulate_game_batch(self, agent, num_games=100):
        """Simulate multiple games in parallel for faster training"""
        results = []
        experiences = []

        if CUDA_AVAILABLE and num_games >= 10:
            # Parallel simulation on GPU
            batch_results = self._simulate_batch_cuda(agent, num_games)
            results.extend(batch_results[0])
            experiences.extend(batch_results[1])
        else:
            # Sequential simulation on CPU
            for _ in range(num_games):
                result, game_experiences = self._simulate_single_game(agent)
                results.append(result)
                experiences.extend(game_experiences)

        return results, experiences

    def _simulate_batch_cuda(self, agent, num_games):
        """CUDA-accelerated batch simulation"""
        # For simplicity, still run games sequentially but process experiences in batches
        results = []
        all_experiences = []

        for _ in range(num_games):
            result, experiences = self._simulate_single_game(agent)
            results.append(result)
            all_experiences.extend(experiences)

        return results, all_experiences

    def _simulate_single_game(self, agent):
        """Simulate single training game"""
        board = gamerules.Board()
        opponent = RNGPlayer("RandomOpponent")

        game_experiences = []
        game_finished = False

        # Randomly decide who starts
        players = [agent, opponent] if np.random.random() < 0.5 else [opponent, agent]
        start_values = {players[0]: 1, players[1]: -1}

        agent.newGame(True)
        opponent.newGame(True)

        move_count = 0
        while not game_finished and move_count < 42:  # Max possible moves
            for player in players:
                if game_finished:
                    break

                current_board = copy.deepcopy(board)

                # Get action
                try:
                    action = player.getAction(current_board, start_values[player])
                except:
                    # Player crashed, opponent wins
                    game_finished = True
                    winner = -1 if player == agent else 1
                    break

                # Validate action
                possible_actions = board.getPossibleActions()
                if action not in possible_actions:
                    # Invalid action, opponent wins
                    game_finished = True
                    winner = -1 if player == agent else 1
                    break

                # Store experience for agent
                if player == agent:
                    player_board = current_board.prepareBoardForPlayer(start_values[player])
                    state = agent.extract_features(player_board, 1)
                    game_experiences.append((state, action, current_board, start_values[player]))

                # Make move
                board.updateBoard(action, start_values[player])
                move_count += 1

                # Check victory
                if board.checkVictory(action, start_values[player]):
                    game_finished = True
                    winner = 1 if player == agent else -1
                    break

                # Check draw
                if len(board.getPossibleActions()) == 0:
                    game_finished = True
                    winner = 0  # Draw
                    break

        # If no winner determined, it's a draw
        if not game_finished:
            winner = 0

        # Assign rewards to experiences
        final_experiences = []
        for i, (state, action, prev_board, player_value) in enumerate(game_experiences):
            # Reward based on game outcome
            if winner == 1:  # Agent won
                reward = 10
            elif winner == -1:  # Agent lost
                reward = -10
            else:  # Draw
                reward = 1

            # Add small reward for making valid moves (survival)
            reward += 0.1

            # Get next state
            if i < len(game_experiences) - 1:
                next_state, _, next_board, _ = game_experiences[i + 1]
                next_possible_actions = next_board.getPossibleActions()
            else:
                next_state = None
                next_possible_actions = []

            final_experiences.append((state, action, reward, next_state, next_possible_actions))

        return winner, final_experiences

    def train_agent(self, training_minutes=15, test_interval=5000):
        """Train the Q-learning agent"""
        agent = Player("QLearning Agent")
        agent.epsilon = 0.3  # Start with exploration

        print(f"Starting training for {training_minutes} minutes...")
        print("=" * 60)

        start_time = time.time()
        end_time = start_time + (training_minutes * 60)

        total_games = 0
        wins = 0
        losses = 0
        draws = 0

        training_stats = {
            'games_per_minute': [],
            'win_rates': [],
            'q_table_sizes': []
        }

        last_test_time = start_time
        last_games = 0

        while time.time() < end_time:
            batch_start = time.time()

            # Simulate batch of games
            results, experiences = self.simulate_game_batch(agent, self.cuda_batch_size)

            # Update Q-table with experiences
            for state, action, reward, next_state, next_possible_actions in experiences:
                agent.update_q_value(state, action, reward, next_state, next_possible_actions)

            # Update statistics
            for result in results:
                total_games += 1
                if result == 1:
                    wins += 1
                elif result == -1:
                    losses += 1
                else:
                    draws += 1

            # Decay epsilon (reduce exploration over time)
            agent.epsilon = max(0.05, agent.epsilon * 0.9995)

            # Print progress every test_interval games
            if total_games - last_games >= test_interval:
                current_time = time.time()
                elapsed = (current_time - start_time) / 60
                games_per_min = total_games / elapsed if elapsed > 0 else 0
                win_rate = wins / total_games if total_games > 0 else 0

                print(f"Games: {total_games:6d} | "
                      f"Time: {elapsed:4.1f}m | "
                      f"Win Rate: {win_rate:5.1%} | "
                      f"Q-States: {len(agent.q_table):6d} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Speed: {games_per_min:.0f} games/min")

                training_stats['games_per_minute'].append(games_per_min)
                training_stats['win_rates'].append(win_rate)
                training_stats['q_table_sizes'].append(len(agent.q_table))

                last_games = total_games

        training_time = (time.time() - start_time) / 60
        final_win_rate = wins / total_games if total_games > 0 else 0

        print("=" * 60)
        print(f"Training completed in {training_time:.1f} minutes")
        print(f"Total games: {total_games}")
        print(f"Final win rate: {final_win_rate:.1%} (W:{wins} L:{losses} D:{draws})")
        print(f"Q-table size: {len(agent.q_table)} states")
        print(f"Average speed: {total_games / training_time:.0f} games/minute")

        # Save trained model
        agent.epsilon = 0.0  # Disable exploration for play
        agent.save_weights("weights.pkl")

        return agent, training_stats


def benchmark_agent(agent, num_test_games=100):
    """Comprehensive testing against random players"""
    print("\n" + "=" * 60)
    print("BENCHMARKING PHASE")
    print("=" * 60)

    opponent = RNGPlayer("TestOpponent")
    results = {'wins': 0, 'losses': 0, 'draws': 0}
    game_lengths = []

    print(f"Testing against random player ({num_test_games} games)...")

    for game_num in range(num_test_games):
        board = gamerules.Board()

        # Alternate who starts
        if game_num % 2 == 0:
            players = [agent, opponent]
            start_values = {agent: 1, opponent: -1}
            agent_starts = True
        else:
            players = [opponent, agent]
            start_values = {opponent: 1, agent: -1}
            agent_starts = False

        agent.newGame(True)
        opponent.newGame(True)

        game_finished = False
        moves = 0

        while not game_finished and moves < 42:
            for player in players:
                if game_finished:
                    break

                action = player.getAction(copy.deepcopy(board), start_values[player])

                # Validate action
                possible_actions = board.getPossibleActions()
                if action not in possible_actions:
                    # Invalid move
                    winner = -1 if player == agent else 1
                    game_finished = True
                    break

                board.updateBoard(action, start_values[player])
                moves += 1

                if board.checkVictory(action, start_values[player]):
                    winner = 1 if player == agent else -1
                    game_finished = True
                    break

                if len(board.getPossibleActions()) == 0:
                    winner = 0  # Draw
                    game_finished = True
                    break

        if not game_finished:
            winner = 0

        game_lengths.append(moves)

        if winner == 1:
            results['wins'] += 1
        elif winner == -1:
            results['losses'] += 1
        else:
            results['draws'] += 1

        # Progress indicator
        if (game_num + 1) % 20 == 0:
            current_wr = results['wins'] / (game_num + 1)
            print(f"  Progress: {game_num + 1:3d}/{num_test_games} games | "
                  f"Current win rate: {current_wr:.1%}")

    # Final results
    total = sum(results.values())
    win_rate = results['wins'] / total
    avg_game_length = np.mean(game_lengths)

    print("\n" + "-" * 40)
    print("FINAL BENCHMARK RESULTS")
    print("-" * 40)
    print(f"Win Rate: {win_rate:.1%} ({results['wins']}/{total})")
    print(f"Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")
    print(f"Average game length: {avg_game_length:.1f} moves")
    print(f"Requirement (80%): {'✓ PASSED' if win_rate >= 0.8 else '✗ FAILED'}")

    if win_rate >= 0.8:
        print("🎉 Agent meets the requirement!")
    else:
        print("⚠️  Agent needs more training or better features")

    return win_rate, results


if __name__ == "__main__":
    trainer = QLearningTrainer()

    # Train the agent
    agent, training_stats = trainer.train_agent(training_minutes=15, test_interval=2000)

    # Benchmark the trained agent
    final_win_rate, results = benchmark_agent(agent, num_test_games=100)

    print(f"\nTraining complete. Model saved as 'weights.pkl'")
    print(f"Final performance: {final_win_rate:.1%} win rate")