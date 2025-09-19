#!/usr/bin/env python3
"""
Analysis script for Qugeister Q-value analysis and visualization.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qugeister.utils.config import load_config
from qugeister.utils.logging import setup_logging
from qugeister.analysis.qvalue_analyzer import QValueFullOutputModule


def main():
    parser = argparse.ArgumentParser(description="Analyze Qugeister Q-values")
    parser.add_argument("--model", type=str, default="fast_quantum_model.pth", help="Model path")
    parser.add_argument("--states", type=int, default=1000, help="Number of states to analyze")
    parser.add_argument("--output", type=str, default="results/analysis", help="Output directory")
    parser.add_argument("--config", type=Path, help="Configuration file")
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    logger = setup_logging("INFO")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Q-value analysis for {args.states} states")
    
    # Run analysis
    analyzer = QValueFullOutputModule(args.model)
    qvalue_map, statistics, pattern_stats = analyzer.run_full_analysis(args.states)
    
    logger.info(f"Analysis completed. Results saved to: {output_path}")


if __name__ == "__main__":
    main()