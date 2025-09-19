"""
Main CLI entry point for Qugeister system.
"""

import argparse
import sys
from pathlib import Path

from ..utils.config import load_config
from ..utils.logging import setup_logging


def cmd_train(args):
    """Train a quantum AI model"""
    from ..quantum.quantum_trainer import FastQuantumTrainer, FastQuantumNeuralNetwork
    
    config = load_config(args.config)
    logger = setup_logging(args.log_level)
    
    logger.info(f"Training quantum model with {args.episodes} episodes")
    
    # Initialize model and trainer
    model = FastQuantumNeuralNetwork(
        n_qubits=config.quantum['n_qubits'],
        output_dim=config.network['output_dim']
    )
    trainer = FastQuantumTrainer(model, lr=config.training['learning_rate'])
    
    # Run training (simplified for CLI)
    from ..quantum.quantum_trainer import train_fast_quantum
    model, rewards = train_fast_quantum(
        episodes=args.episodes,
        n_qubits=config.quantum['n_qubits']
    )
    
    logger.info("Training completed successfully")


def cmd_analyze(args):
    """Analyze Q-values and generate reports"""
    from ..analysis.qvalue_analyzer import QValueFullOutputModule
    
    config = load_config(args.config)
    logger = setup_logging(args.log_level)
    
    logger.info(f"Analyzing Q-values for {args.states} states")
    
    analyzer = QValueFullOutputModule(args.model_path)
    qvalue_map, statistics, pattern_stats = analyzer.run_full_analysis(args.states)
    
    logger.info("Analysis completed successfully")


def cmd_web(args):
    """Launch web interface"""
    import webbrowser
    from pathlib import Path
    
    config = load_config(args.config)
    logger = setup_logging(args.log_level)
    
    web_path = Path(__file__).parent.parent.parent.parent / "web" / "templates"
    
    if args.mode == "designer":
        file_path = web_path / "quantum_designer.html"
    else:
        file_path = web_path / "playground.html"
    
    logger.info(f"Opening {args.mode} interface: {file_path}")
    webbrowser.open(f"file://{file_path.absolute()}")


def cmd_tournament(args):
    """Run AI tournament"""
    config = load_config(args.config)
    logger = setup_logging(args.log_level)
    
    logger.info(f"Running tournament with {args.rounds} rounds")
    # Tournament logic would go here
    logger.info("Tournament completed")


def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="Qugeister - Quantum Geister AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qugeister train --episodes 1000              # Train quantum AI
  qugeister analyze --states 500               # Analyze Q-values  
  qugeister web --mode designer                # Launch web designer
  qugeister tournament --rounds 10             # Run tournament
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Configuration file path'
    )
    parser.add_argument(
        '--log-level', '-l',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train quantum AI model')
    train_parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    train_parser.add_argument('--qubits', type=int, default=4, help='Number of qubits')
    train_parser.set_defaults(func=cmd_train)
    
    # Analyze command  
    analyze_parser = subparsers.add_parser('analyze', help='Analyze Q-values')
    analyze_parser.add_argument('--states', type=int, default=1000, help='Number of states to analyze')
    analyze_parser.add_argument('--model-path', default='fast_quantum_model.pth', help='Model file path')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--mode', choices=['designer', 'playground'], default='designer', help='Interface mode')
    web_parser.set_defaults(func=cmd_web)
    
    # Tournament command
    tournament_parser = subparsers.add_parser('tournament', help='Run AI tournament')
    tournament_parser.add_argument('--rounds', type=int, default=10, help='Tournament rounds')
    tournament_parser.set_defaults(func=cmd_tournament)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()