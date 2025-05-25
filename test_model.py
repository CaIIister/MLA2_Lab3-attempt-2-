import numpy as np
import copy
import time
import gamerules
from player import Player


class RNGPlayer(gamerules.Player):
    """Random player for testing"""

    def __init__(self, name):
        super().__init__(name)

    def getAction(self, board, startValue):
        possibleActions = self.getPossibleActions(board.board)
        return np.random.choice(possibleActions)

    def newGame(self, new_opponent):
        pass


def quick_test(num_games=20, show_games=False):
    """Quick test of trained model"""
    print("=" * 50)
    print("QUICK MODEL TEST")
    print("=" * 50)

    # Load trained agent
    try:
        agent = Player("Trained Agent", "weights.pkl")
        if not agent.q_table:
            print("❌ No trained model found! Run training.py first.")
            return
        print(f"✓ Loaded model with {len(agent.q_table)} learned states")
    except:
        print("❌ Failed to load model! Check if weights.pkl exists.")
        return

    opponent = RNGPlayer("Random")

    results = {'wins': 0, 'losses': 0, 'draws': 0}
    game_details = []

    print(f"\nTesting against random player ({num_games} games)...")
    start_time = time.time()

    for game_num in range(num_games):
        board = gamerules.Board()

        # Alternate starting player
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
        move_history = []

        while not game_finished and moves < 42:
            for player in players:
                if game_finished:
                    break

                action = player.getAction(copy.deepcopy(board), start_values[player])
                move_history.append((player.getName()[:6], action))

                # Validate action
                possible_actions = board.getPossibleActions()
                if action not in possible_actions:
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
                    winner = 0
                    game_finished = True
                    break

        if not game_finished:
            winner = 0

        # Record results
        if winner == 1:
            results['wins'] += 1
            outcome = "WIN"
        elif winner == -1:
            results['losses'] += 1
            outcome = "LOSS"
        else:
            results['draws'] += 1
            outcome = "DRAW"

        game_details.append({
            'game': game_num + 1,
            'outcome': outcome,
            'moves': moves,
            'agent_starts': agent_starts
        })

        # Show individual game results if requested
        if show_games:
            starter = "Agent" if agent_starts else "Random"
            print(f"  Game {game_num + 1:2d}: {outcome:4s} ({moves:2d} moves, {starter} started)")

    test_time = time.time() - start_time
    total = sum(results.values())
    win_rate = results['wins'] / total

    # Summary statistics
    avg_moves = np.mean([g['moves'] for g in game_details])
    wins_as_first = sum(1 for g in game_details if g['outcome'] == 'WIN' and g['agent_starts'])
    wins_as_second = sum(1 for g in game_details if g['outcome'] == 'WIN' and not g['agent_starts'])
    games_as_first = sum(1 for g in game_details if g['agent_starts'])
    games_as_second = total - games_as_first

    print("\n" + "-" * 50)
    print("RESULTS SUMMARY")
    print("-" * 50)
    print(f"Overall Win Rate: {win_rate:.1%} ({results['wins']}/{total})")
    print(f"Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")
    print(f"Average game length: {avg_moves:.1f} moves")
    print(f"Test time: {test_time:.1f} seconds")

    print(f"\nDetailed Performance:")
    if games_as_first > 0:
        first_wr = wins_as_first / games_as_first
        print(f"  As first player:  {first_wr:.1%} ({wins_as_first}/{games_as_first})")
    if games_as_second > 0:
        second_wr = wins_as_second / games_as_second
        print(f"  As second player: {second_wr:.1%} ({wins_as_second}/{games_as_second})")

    # Requirement check
    requirement_met = win_rate >= 0.8
    print(f"\nRequirement (≥80%): {'✅ PASSED' if requirement_met else '❌ FAILED'}")

    if requirement_met:
        print("🎉 Model is ready for submission!")
    else:
        print("⚠️  Model needs more training")

    return win_rate, results


def detailed_test(num_games=100):
    """More comprehensive test"""
    print("=" * 50)
    print("DETAILED MODEL TEST")
    print("=" * 50)

    # Run comprehensive test
    win_rate, results = quick_test(num_games, show_games=False)

    # Additional analysis
    agent = Player("Trained Agent", "weights.pkl")
    print(f"\nModel Statistics:")
    print(f"  Q-table size: {len(agent.q_table)} states")
    print(f"  Memory usage: ~{len(agent.q_table) * 8 / 1024:.1f} KB")

    # Feature analysis
    sample_board = np.zeros((6, 7))
    sample_features = agent.extract_features(sample_board, 1)
    print(f"  Feature vector size: {len(sample_features)} dimensions")

    return win_rate, results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "detailed":
            detailed_test(100)
        elif sys.argv[1] == "quick":
            quick_test(10, show_games=True)
        else:
            print("Usage: python test_model.py [quick|detailed]")
            print("  quick   - Fast test with 10 games")
            print("  detailed - Comprehensive test with 100 games")
    else:
        # Default: medium test
        quick_test(20, show_games=False)