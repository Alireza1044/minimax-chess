import chess
from math import inf
import numpy as np

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

pieces = {1: "PAWN",
          2: "KNIGHT",
          3: "BISHOP",
          4: "ROOK",
          5: "QUEEN",
          6: "KING"}

scores = {"PAWN": 10,
          "KNIGHT": 30,
          "BISHOP": 30,
          "ROOK": 50,
          "QUEEN": 90}

pawn_piece = [0, 0, 0, 0, 0, 0, 0, 0,
              50, 50, 50, 50, 50, 50, 50, 50,
              10, 10, 20, 30, 30, 20, 10, 10,
              5, 5, 10, 25, 25, 10, 5, 5,
              0, 0, 0, 20, 20, 0, 0, 0,
              5, -5, -10, 0, 0, -10, -5, 5,
              5, 10, 10, -20, -20, 10, 10, 5,
              0, 0, 0, 0, 0, 0, 0, 0]

knight_piece = [-50, -40, -30, -30, -30, -30, -40, -50,
                -40, -20, 0, 0, 0, 0, -20, -40,
                -30, 0, 10, 15, 15, 10, 0, -30,
                -30, 5, 15, 20, 20, 15, 5, -30,
                -30, 0, 15, 20, 20, 15, 0, -30,
                -30, 5, 10, 15, 15, 10, 5, -30,
                -40, -20, 0, 5, 5, 0, -20, -40,
                -50, -40, -30, -30, -30, -30, -40, -50]

bishop_piece = [-20, -10, -10, -10, -10, -10, -10, -20,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -10, 0, 5, 10, 10, 5, 0, -10,
                -10, 5, 5, 10, 10, 5, 5, -10,
                -10, 0, 10, 10, 10, 10, 0, -10,
                -10, 10, 10, 10, 10, 10, 10, -10,
                -10, 5, 0, 0, 0, 0, 5, -10,
                -20, -10, -10, -10, -10, -10, -10, -20]

rook_piece = [0, 0, 0, 0, 0, 0, 0, 0,
              5, 10, 10, 10, 10, 10, 10, 5,
              -5, 0, 0, 0, 0, 0, 0, -5,
              -5, 0, 0, 0, 0, 0, 0, -5,
              -5, 0, 0, 0, 0, 0, 0, -5,
              -5, 0, 0, 0, 0, 0, 0, -5,
              -5, 0, 0, 0, 0, 0, 0, -5,
              0, 0, 0, 5, 5, 0, 0, 0]

queen_piece = [-20, -10, -10, -5, -5, -10, -10, -20,
               -10, 0, 0, 0, 0, 0, 0, -10,
               -10, 0, 5, 5, 5, 5, 0, -10,
               -5, 0, 5, 5, 5, 5, 0, -5,
               0, 0, 5, 5, 5, 5, 0, -5,
               -10, 5, 5, 5, 5, 5, 0, -10,
               -10, 0, 5, 0, 0, 0, 0, -10,
               -20, -10, -10, -5, -5, -10, -10, -20]

king_piece_mid = [-30, -40, -40, -50, -50, -40, -40, -30,
                  -30, -40, -40, -50, -50, -40, -40, -30,
                  -30, -40, -40, -50, -50, -40, -40, -30,
                  -30, -40, -40, -50, -50, -40, -40, -30,
                  -20, -30, -30, -40, -40, -30, -30, -20,
                  -10, -20, -20, -20, -20, -20, -20, -10,
                  20, 20, 0, 0, 0, 0, 20, 20,
                  20, 30, 10, 0, 0, 10, 30, 20]

king_piece_end = [-50, -40, -30, -20, -20, -30, -40, -50,
                  -30, -20, -10, 0, 0, -10, -20, -30,
                  -30, -10, 20, 30, 30, 20, -10, -30,
                  -30, -10, 30, 40, 40, 30, -10, -30,
                  -30, -10, 30, 40, 40, 30, -10, -30,
                  -30, -10, 20, 30, 30, 20, -10, -30,
                  -30, -30, 0, 0, 0, 0, -30, -30,
                  -50, -30, -30, -30, -30, -30, -30, -50]

class AlphaBetaAI():
    def __init__(self, depth):
        self.depth = depth

    def choose_move(self, board):
        result = self.minimax(board, 0, float('-inf'), float('inf'))[0]
        return result

    def minimax(self, board, depth, alpha, beta):
        if self.depth == depth or board.is_game_over():
            return "", self.evaluate(board, depth % 2 == 0)

        moves = list(board.legal_moves)
        scores = []

        if depth % 2 == 0:  # max
            bestVal = float('-inf')
            for move in moves:
                board.push(move)
                # print("asdasdasd")
                # print("jgisdhgjklsdfhgjksdbgadsgakjsdg;kadsf\n", board)
                # input()
                value = self.minimax(board, depth + 1, alpha, beta)[1]
                scores.append(value)
                board.pop()
                bestVal = max(bestVal, value)
                alpha = max(alpha, bestVal)
                if beta <= alpha:
                    break

            idx = np.argmax(scores)
            move = moves[idx], scores[idx]

        else:  # min
            bestVal = float('inf')
            for move in moves:
                board.push(move)
                value = self.minimax(board, depth + 1, alpha, beta)[1]
                scores.append(value)
                board.pop()
                bestVal = min(bestVal, value)
                beta = min(beta, bestVal)
                if beta <= alpha:
                    break

            idx = np.argmin(scores)
            move = moves[idx], scores[idx]

        return move


    def evaluate(self, board, is_max):

        score = 0
        is_opponent = True

        if board.is_checkmate():
            return float('inf')
        if board.is_stalemate():
            return 0

        for i in range(2):
            for piece in pieces:
                if piece == 6:
                    continue

                if is_opponent:
                    score -= len(list(board.pieces(piece, is_opponent))) * scores[pieces[piece]]
                    score = score - 0.1 * len(list(board.legal_moves))
                else:
                    score += len(list(board.pieces(piece, is_opponent))) * scores[pieces[piece]]
                    score = score + 0.1 * len(list(board.legal_moves))
            is_opponent = False

        is_mid = True

        if is_max:
            for piece in pieces:
                if pieces[piece] == "PAWN":
                    pawns_places = list(board.pieces(piece, is_max))
                    for mohre in pawns_places:
                        score += pawn_piece[mohre]
                elif pieces[piece] == "KNIGHT":
                    knight_places = list(board.pieces(piece, is_max))
                    for mohre in knight_places:
                        score += knight_piece[mohre]
                elif pieces[piece] == "BISHOP":
                    bishop_places = list(board.pieces(piece, is_max))
                    for mohre in bishop_places:
                        score += bishop_piece[mohre]
                elif pieces[piece] == "ROOK":
                    rook_places = list(board.pieces(piece, is_max))
                    for mohre in rook_places:
                        score += rook_piece[mohre]
                elif pieces[piece] == "KING":
                    king_places = list(board.pieces(piece, is_max))
                    for mohre in king_places:
                        if is_mid:
                            score += king_piece_mid[mohre]
                        else:
                            score += king_piece_end[mohre]
                elif pieces[piece] == "QUEEN":
                    queen_places = list(board.pieces(piece, is_max))
                    for mohre in queen_places:
                        score += queen_piece[mohre]

        else:
            pawn_piece.reverse()
            knight_piece.reverse()
            bishop_piece.reverse()
            rook_piece.reverse()
            queen_piece.reverse()
            king_piece_mid.reverse()
            king_piece_end.reverse()
            for piece in pieces:
                if pieces[piece] == "PAWN":
                    pawns_places = list(board.pieces(piece, is_max))
                    for mohre in pawns_places:
                        score += pawn_piece[mohre]
                elif pieces[piece] == "KNIGHT":
                    knight_places = list(board.pieces(piece, is_max))
                    for mohre in knight_places:
                        score += knight_piece[mohre]
                elif pieces[piece] == "BISHOP":
                    bishop_places = list(board.pieces(piece, is_max))
                    for mohre in bishop_places:
                        score += bishop_piece[mohre]
                elif pieces[piece] == "ROOK":
                    rook_places = list(board.pieces(piece, is_max))
                    for mohre in rook_places:
                        score += rook_piece[mohre]
                elif pieces[piece] == "KING":
                    king_places = list(board.pieces(piece, is_max))
                    for mohre in king_places:
                        if is_mid:
                            score += king_piece_mid[mohre]
                        else:
                            score += king_piece_end[mohre]
                elif pieces[piece] == "QUEEN":
                    queen_places = list(board.pieces(piece, is_max))
                    for mohre in queen_places:
                        score += queen_piece[mohre]
            pawn_piece.reverse()
            knight_piece.reverse()
            bishop_piece.reverse()
            rook_piece.reverse()
            queen_piece.reverse()
            king_piece_mid.reverse()
            king_piece_end.reverse()

        return score
