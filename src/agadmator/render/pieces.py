"""Chess piece rendering using Unicode symbols on PIL images."""

import chess
from PIL import Image, ImageDraw, ImageFont

# Unicode chess pieces
UNICODE_PIECES = {
    (chess.KING, chess.WHITE): "\u2654",
    (chess.QUEEN, chess.WHITE): "\u2655",
    (chess.ROOK, chess.WHITE): "\u2656",
    (chess.BISHOP, chess.WHITE): "\u2657",
    (chess.KNIGHT, chess.WHITE): "\u2658",
    (chess.PAWN, chess.WHITE): "\u2659",
    (chess.KING, chess.BLACK): "\u265A",
    (chess.QUEEN, chess.BLACK): "\u265B",
    (chess.ROOK, chess.BLACK): "\u265C",
    (chess.BISHOP, chess.BLACK): "\u265D",
    (chess.KNIGHT, chess.BLACK): "\u265E",
    (chess.PAWN, chess.BLACK): "\u265F",
}

# Cache rendered piece images
_cache: dict[tuple[int, bool, int], Image.Image] = {}

# Fonts that support Unicode chess pieces, in priority order
_FONT_SEARCH_PATHS = [
    # Linux (Ubuntu/Debian — most cloud GPUs)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arch Linux
    # macOS
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    # Windows
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _find_chess_font(size: int) -> ImageFont.FreeTypeFont:
    """Find a font that renders chess Unicode symbols."""
    for path in _FONT_SEARCH_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def get_piece_image(
    piece: chess.Piece,
    square_size: int,
) -> Image.Image | None:
    """Render a chess piece as a transparent PIL image."""
    cache_key = (piece.piece_type, piece.color, square_size)
    if cache_key in _cache:
        return _cache[cache_key]

    symbol = UNICODE_PIECES.get((piece.piece_type, piece.color))
    if not symbol:
        return None

    img = Image.new("RGBA", (square_size, square_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_size = int(square_size * 0.8)
    font = _find_chess_font(font_size)

    # Center the piece in the square
    bbox = draw.textbbox((0, 0), symbol, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (square_size - text_w) // 2 - bbox[0]
    y = (square_size - text_h) // 2 - bbox[1]

    # Draw with outline for visibility
    outline_color = (0, 0, 0, 255)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), symbol, fill=outline_color, font=font)

    fill_color = (255, 255, 255, 255) if piece.color == chess.WHITE else (40, 40, 40, 255)
    draw.text((x, y), symbol, fill=fill_color, font=font)

    _cache[cache_key] = img
    return img
