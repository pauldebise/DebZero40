import chess


def generate_moves_list():

    """Generates the move list."""

    moves = []

    for from_sq in range(64):

        for to_sq in range(64):

            if from_sq == to_sq: continue

            rank_dist = abs(chess.square_rank(from_sq) - chess.square_rank(to_sq))
            file_dist = abs(chess.square_file(from_sq) - chess.square_file(to_sq))

            is_knight = (rank_dist == 1 and file_dist == 2) or (rank_dist == 2 and file_dist == 1)
            is_queen = (rank_dist == 0) or (file_dist == 0) or (rank_dist == file_dist)

            if is_knight or is_queen: #Is the move geometrically valid ?

                moves.append(chess.Move(from_sq, to_sq).uci())

                is_promo_rank = (chess.square_rank(to_sq) == 7 and chess.square_rank(from_sq) == 6)

                if is_promo_rank and is_queen and file_dist <= 1:

                     moves.append(chess.Move(from_sq, to_sq, chess.KNIGHT).uci())
                     moves.append(chess.Move(from_sq, to_sq, chess.BISHOP).uci())
                     moves.append(chess.Move(from_sq, to_sq, chess.ROOK).uci())
                    #Otherwise queen promoting is inplicit

    return moves


class FlatMapper:

    """
    This class deals with the move mapping between the uci convention and the LeelaChessZero 1858 dense move representation.
    Caution : LeelaChessZero 1858 dense move representation only accounts for white moves, black moves first have to be mirrored.
    """

    def __init__(self):

        self.moves_list = generate_moves_list()

        self.move_to_index = {move: i for i, move in enumerate(self.moves_list)}

    def get_move_string(self, index):

        """Conversion index -> uci"""

        if 0 <= index < len(self.moves_list):
            return self.moves_list[index]

        return "Unknown"

    def get_move_index(self, uci_move):

        """Conversion uci -> index"""

        #If is not a queen promotion
        if uci_move in self.move_to_index:
            return self.move_to_index[uci_move]

        #If is a queen promotion
        if len(uci_move) == 5 and uci_move.endswith('q'):
            base_move = uci_move[:4]
            if base_move in self.move_to_index:
                return self.move_to_index[base_move]

        return None