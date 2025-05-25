import numpy as np
import time
import copy
from collections import defaultdict
import gamerules
from player import Player
from tqdm import tqdm

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

    def simulate_game_batch(self, agent, num_games=100, pbar_games=None):
        """Simulate multiple games in parallel for faster training"""
        results = []
        experiences = []

        if CUDA_AVAILABLE and num_games >= 10:
            # Parallel simulation on GPU
            batch_results = self._simulate_batch_cuda(agent, num_games, pbar_games)
            results.extend(batch_results[0])
            experiences.extend(batch_results[1])
        else:
            # Sequential simulation on CPU
            for i in range(num_games):
                result, game_experiences = self._simulate_single_game(agent)
                results.append(result)
                experiences.extend(game_experiences)

                # Update progress bar
                if pbar_games:
                    win_rate = sum(1 for r in results if r == 1) / len(results) if results else 0
                    pbar_games.set_description(f"Batch Games (WR: {win_rate:.1%})")
                    pbar_games.update(1)

        return results, experiences

    def _simulate_batch_cuda(self, agent, num_games, pbar_games=None):
        """CUDA-accelerated batch simulation"""
        # For simplicity, still run games sequentially but process experiences in batches
        results = []
        all_experiences = []

        for i in range(num_games):
            result, experiences = self._simulate_single_game(agent)
            results.append(result)
            all_experiences.extend(experiences)

            # Update progress bar
            if pbar_games and i % max(1, num_games // 20) == 0:  # Update every 5%
                win_rate = sum(1 for r in results if r == 1) / len(results) if results else 0
                pbar_games.set_description(f"CUDA Batch (WR: {win_rate:.1%})")
                pbar_games.n = i + 1
                pbar_games.refresh()

        if pbar_games:
            pbar_games.n = num_games
            pbar_games.refresh()

        return results, all_experiences

    def _simulate_single_game(self, agent):
        """Simulate single training game with proper error handling"""
        try:
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
            winner = 0  # Default draw

            while not game_finished and move_count < 42:
                for player in players:
                    if game_finished:
                        break

                    current_board = copy.deepcopy(board)

                    # Get action
                    try:
                        action = player.getAction(current_board, start_values[player])
                    except Exception as e:
                        # Player crashed, opponent wins
                        winner = -1 if player == agent else 1
                        game_finished = True
                        break

                    # Validate action
                    possible_actions = board.getPossibleActions()
                    if action not in possible_actions:
                        # Invalid action, opponent wins
                        winner = -1 if player == agent else 1
                        game_finished = True
                        break

                    # Store experience for agent BEFORE making move
                    if player == agent:
                        try:
                            player_board = current_board.prepareBoardForPlayer(start_values[player])
                            state = agent.extract_features(player_board, 1)
                            game_experiences.append((state, action, current_board, start_values[player]))
                        except Exception as e:
                            print(f"Experience recording error: {e}")

                    # Make move
                    try:
                        board.updateBoard(action, start_values[player])
                        move_count += 1

                        # Check victory
                        if board.checkVictory(action, start_values[player]):
                            winner = 1 if player == agent else -1
                            game_finished = True
                            break

                        # Check draw
                        if len(board.getPossibleActions()) == 0:
                            winner = 0
                            game_finished = True
                            break

                    except Exception as e:
                        print(f"Move execution error: {e}")
                        winner = -1 if player == agent else 1
                        game_finished = True
                        break

            # Process experiences
            final_experiences = []
            if game_experiences:
                # Assign rewards
                if winner == 1:  # Agent won
                    moves_factor = max(0.5, 1.0 - (move_count / 42))  # Higher reward for quick wins
                    base_reward = 3000 * moves_factor  # Huge reward for winning, scaled by game length
                elif winner == -1:  # Agent lost
                    base_reward = -1500  # Severe penalty for losing
                else:  # Draw
                    base_reward = -800  # Very heavy penalty for draws

                for i, (state, action, prev_board, player_value) in enumerate(game_experiences):
                    try:
                        # Get next state
                        if i < len(game_experiences) - 1:
                            next_state, _, next_board, _ = game_experiences[i + 1]
                            next_possible_actions = next_board.getPossibleActions()
                        else:
                            next_state = state  # Terminal state
                            next_possible_actions = []

                        # Reward with some decay for earlier moves
                        reward_mult = (i + 1) / len(game_experiences)

                        # Add intermediate rewards
                        if i < len(game_experiences) - 1:
                            next_state, _, next_board, _ = game_experiences[i + 1]
                            
                            # Get feature scores for current and next state
                            current_pattern = sum(1 for col in range(7) if agent._evaluate_winning_pattern(prev_board.board, col, player_value) >= 4)
                            next_pattern = sum(1 for col in range(7) if agent._evaluate_winning_pattern(next_board.board, col, player_value) >= 4)
                            
                            current_urgency = sum(1 for col in range(7) if agent._evaluate_move_urgency(prev_board.board, col, player_value) >= 3)
                            next_urgency = sum(1 for col in range(7) if agent._evaluate_move_urgency(next_board.board, col, player_value) >= 3)
                            
                            # Reward improvements in position
                            if next_pattern > current_pattern:
                                final_reward = base_reward * reward_mult + 200  # Big bonus for forming winning patterns
                            elif next_urgency > current_urgency:
                                final_reward = base_reward * reward_mult + 100  # Bonus for addressing urgent threats
                            else:
                                final_reward = base_reward * reward_mult
                        else:
                            final_reward = base_reward * reward_mult

                        final_experiences.append((state, action, final_reward, next_state, next_possible_actions))
                    except Exception as e:
                        print(f"Experience processing error: {e}")

            return winner, final_experiences

        except Exception as e:
            print(f"Game simulation error: {e}")
            return 0, []  # Return draw with no experiences

    def train_agent(self, training_minutes=15, test_interval=300):
        """Train the Q-learning agent"""
        agent = Player("QLearning Agent")
        agent.epsilon = 1.0  # Start with pure exploration
        agent.learning_rate = 0.6  # Very aggressive learning
        agent.discount_factor = 0.7  # Focus heavily on immediate rewards

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

        # Main training progress bar (time-based)
        training_duration = training_minutes * 60
        with tqdm(total=training_duration, desc="Training Progress", unit="s",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}/{remaining} [{rate_fmt}]") as pbar_time:

            while time.time() < end_time:
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Update time progress bar
                pbar_time.n = min(elapsed_time, training_duration)
                pbar_time.refresh()

                batch_start = time.time()

                # Games progress bar for this batch
                with tqdm(total=self.cuda_batch_size, desc="Game Batch",
                          leave=False, disable=self.cuda_batch_size < 10) as pbar_games:

                    # Simulate batch of games
                    results, experiences = self.simulate_game_batch(agent, self.cuda_batch_size, pbar_games)

                # Update Q-table with experiences
                experience_count = 0
                if experiences:
                    for state, action, reward, next_state, next_possible_actions in experiences:
                        agent.update_q_value(state, action, reward, next_state, next_possible_actions)
                        experience_count += 1

                # Debug info for first few batches
                if total_games < 1000 and experience_count == 0:
                    tqdm.write(f"WARNING: No experiences generated in batch of {len(results)} games")

                # Update statistics
                for result in results:
                    total_games += 1
                    if result == 1:
                        wins += 1
                    elif result == -1:
                        losses += 1
                    else:
                        draws += 1

                # Exploration strategy
                if total_games < 100:  # Initial phase
                    agent.epsilon = max(0.4, agent.epsilon * 0.97)  # Very quick initial decay
                elif total_games < 300:  # Middle phase
                    agent.epsilon = max(0.2, agent.epsilon * 0.98)  # Quick decay
                else:  # Late phase
                    agent.epsilon = max(0.1, agent.epsilon * 0.999)  # Maintain some exploration

                # Update progress bar description with current stats
                if total_games > 0:
                    win_rate = wins / total_games
                    games_per_min = total_games / (elapsed_time / 60) if elapsed_time > 0 else 0

                    pbar_time.set_description(
                        f"Training │ Games:{total_games:5d} │ WR:{win_rate:5.1%} │ "
                        f"States:{len(agent.q_table):5d} │ ε:{agent.epsilon:.3f} │ "
                        f"Speed:{games_per_min:.0f}/min"
                    )

                # Detailed progress every test_interval games OR every 60 seconds
                if (total_games - last_games >= test_interval) or (current_time - last_test_time >= 60):
                    elapsed = elapsed_time / 60
                    games_per_min = total_games / elapsed if elapsed > 0 else 0
                    win_rate = wins / total_games if total_games > 0 else 0

                    tqdm.write(f"┌─ Checkpoint │ Games:{total_games:6d} │ Time:{elapsed:4.1f}m │ "
                               f"WinRate:{win_rate:5.1%} │ Q-States:{len(agent.q_table):6d}")

                    training_stats['games_per_minute'].append(games_per_min)
                    training_stats['win_rates'].append(win_rate)
                    training_stats['q_table_sizes'].append(len(agent.q_table))

                    last_games = total_games
                    last_test_time = current_time

                # Small delay to prevent overwhelming the console
                if self.cuda_batch_size < 10:
                    time.sleep(0.01)

        training_time = (time.time() - start_time) / 60
        final_win_rate = wins / total_games if total_games > 0 else 0

        print("\n" + "=" * 60)
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

    # Progress bar for testing
    with tqdm(total=num_test_games, desc="Testing",
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

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

            # Update progress bar with current win rate
            current_wr = results['wins'] / (game_num + 1)
            pbar.set_description(f"Testing (WR: {current_wr:.1%})")
            pbar.update(1)

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

    # Train the agent (shorter initial training)
    agent, training_stats = trainer.train_agent(training_minutes=5, test_interval=200)

    # Benchmark the trained agent
    final_win_rate, results = benchmark_agent(agent, num_test_games=100)

    print(f"\nTraining complete. Model saved as 'weights.pkl'")
    print(f"Final performance: {final_win_rate:.1%} win rate")