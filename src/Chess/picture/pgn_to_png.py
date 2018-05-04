import sys
import os
import subprocess
import chess.svg
import chess.pgn


TEMP = 'temp.svg'


def main():
    if len(sys.argv) <= 1 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print('Run `npm install svgexport -g` before using this script')
        print('Usage: python pgn_to_png.py PGN_FILE')
        return

    pgn_path = sys.argv[1]

    base, _ = os.path.splitext(os.path.basename(pgn_path))

    os.mkdir(base)

    with open(pgn_path) as pgn:
        game = chess.pgn.read_game(pgn)
        board = game.board()

        i = 0
        board_to_png(board, base, i, None)
        for move in game.main_line():
            i += 1
            board.push(move)
            board_to_png(board, base, i, move)


    if os.path.exists(TEMP):
        os.remove(TEMP)


def board_to_png(board, base, i, lastmove):
    png_filename = '{}/board{}.png'.format(base, i)
    with open(TEMP, 'w+') as temp:
        svg_text = chess.svg.board(board=board, lastmove=lastmove)
        temp.write(svg_text)
    subprocess.run(['svgexport', TEMP, png_filename, '1024:1024'])


if __name__ == '__main__':
    main()
