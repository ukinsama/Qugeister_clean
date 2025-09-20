"""
Game State representation for Qugeister system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GameState:
    """Immutable game state representation"""

    board: np.ndarray
    turn: int
    current_player: str
    player_a_pieces: Dict[Tuple[int, int], str]  # position -> piece_type
    player_b_pieces: Dict[Tuple[int, int], str]
    move_history: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    game_over: bool = False
    winner: Optional[str] = None

    def copy(self) -> "GameState":
        """Create a deep copy of the game state"""
        return GameState(
            board=self.board.copy(),
            turn=self.turn,
            current_player=self.current_player,
            player_a_pieces=self.player_a_pieces.copy(),
            player_b_pieces=self.player_b_pieces.copy(),
            move_history=self.move_history.copy(),
            game_over=self.game_over,
            winner=self.winner,
        )

    def to_vector(self) -> np.ndarray:
        """Convert game state to 252-dimensional vector for neural network"""
        # 7 channels × 6×6 board = 252 dimensions
        channels = np.zeros((7, 6, 6))

        # Channel 0: Player A good pieces
        for pos, piece_type in self.player_a_pieces.items():
            if piece_type == "good":
                channels[0, pos[1], pos[0]] = 1.0

        # Channel 1: Player A bad pieces
        for pos, piece_type in self.player_a_pieces.items():
            if piece_type == "bad":
                channels[1, pos[1], pos[0]] = 1.0

        # Channel 2: Player B pieces (visible but unknown type)
        for pos in self.player_b_pieces.keys():
            channels[2, pos[1], pos[0]] = 1.0

        # Channel 3: Known Player B good pieces (captured info)
        # Channel 4: Known Player B bad pieces (captured info)
        # These would be filled based on game history analysis

        # Channel 5: Current player indicator
        if self.current_player == "A":
            channels[5] = 1.0
        else:
            channels[5] = -1.0

        # Channel 6: Escape positions
        # Player A escape positions
        channels[6, 5, 0] = 1.0  # (0,5)
        channels[6, 5, 5] = 1.0  # (5,5)
        # Player B escape positions
        channels[6, 0, 0] = -1.0  # (0,0)
        channels[6, 0, 5] = -1.0  # (5,0)

        return channels.flatten()
