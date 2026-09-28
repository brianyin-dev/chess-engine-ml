const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const RANKS = ["8", "7", "6", "5", "4", "3", "2", "1"];
const PIECES = {
  wp: new URL("./assets/pieces/wP.svg", import.meta.url).href,
  wr: new URL("./assets/pieces/wR.svg", import.meta.url).href,
  wn: new URL("./assets/pieces/wN.svg", import.meta.url).href,
  wb: new URL("./assets/pieces/wB.svg", import.meta.url).href,
  wq: new URL("./assets/pieces/wQ.svg", import.meta.url).href,
  wk: new URL("./assets/pieces/wK.svg", import.meta.url).href,
  bp: new URL("./assets/pieces/bP.svg", import.meta.url).href,
  br: new URL("./assets/pieces/bR.svg", import.meta.url).href,
  bn: new URL("./assets/pieces/bN.svg", import.meta.url).href,
  bb: new URL("./assets/pieces/bB.svg", import.meta.url).href,
  bq: new URL("./assets/pieces/bQ.svg", import.meta.url).href,
  bk: new URL("./assets/pieces/bK.svg", import.meta.url).href,
};

// Use the same deployed origin. Keep the standalone port-8000 workflow working locally.
const API_BASE = window.location.port === "8000" ? "http://127.0.0.1:5000" : "";
const ENGINE_DEPTH = 8; // Upper bound; the engine returns its last completed depth.
const ENGINE_TIME_MS = 1500;
const USE_OPENING_BOOK = true;
const MAX_BOOK_PLY = 20; // Ten full moves; search starts earlier when the book has no entry.
const HUMAN_COLOR = "w";

const boardEl = document.getElementById("chessboard");
const resetButton = document.getElementById("reset-board");

let gameState = createInitialState();
let selectedSquare = null;
let legalTargets = [];
let engineThinking = false;

function createInitialBoard() {
  return [
    ["br", "bn", "bb", "bq", "bk", "bb", "bn", "br"],
    ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp"],
    [null, null, null, null, null, null, null, null],
    [null, null, null, null, null, null, null, null],
    [null, null, null, null, null, null, null, null],
    [null, null, null, null, null, null, null, null],
    ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp"],
    ["wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr"],
  ];
}

function createInitialState() {
  return {
    board: createInitialBoard(),
    turn: "w",
    moveHistory: [],
    uciHistory: [],
    enPassant: null,
    castling: {
      w: { kingSide: true, queenSide: true },
      b: { kingSide: true, queenSide: true },
    },
  };
}

function createSquareLabels() {
  const bottomFiles = document.getElementById("file-labels-bottom");
  const leftRanks = document.getElementById("rank-labels-left");

  const fileMarkup = ["", ...FILES].map((label) => `<span>${label}</span>`).join("");
  const rankMarkup = RANKS.map((label) => `<span>${label}</span>`).join("");

  if (bottomFiles) bottomFiles.innerHTML = fileMarkup;
  if (leftRanks) leftRanks.innerHTML = rankMarkup;
}

function renderBoard() {
  boardEl.innerHTML = "";

  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const square = document.createElement("button");
      const piece = gameState.board[row][col];
      const squareName = toSquare(row, col);
      const isLight = (row + col) % 2 === 0;
      const isSelected = selectedSquare?.row === row && selectedSquare?.col === col;
      const legalMove = legalTargets.find((move) => move.row === row && move.col === col);

      square.type = "button";
      square.className = `square ${isLight ? "light" : "dark"}`;
      if (isSelected) square.classList.add("selected");
      if (legalMove) square.classList.add(legalMove.capture ? "capture" : "move");
      square.dataset.row = String(row);
      square.dataset.col = String(col);
      square.disabled = engineThinking || gameState.turn !== HUMAN_COLOR;
      square.setAttribute("aria-label", `${squareName} ${piece ? describePiece(piece) : "empty square"}`);

      if (piece) {
        const sprite = document.createElement("img");
        sprite.className = "piece-sprite";
        sprite.src = PIECES[piece];
        sprite.alt = describePiece(piece);
        sprite.draggable = false;
        square.appendChild(sprite);
      }

      square.addEventListener("click", () => {
        void handleSquareClick(row, col);
      });

      boardEl.appendChild(square);
    }
  }
}

async function handleSquareClick(row, col) {
  if (engineThinking || gameState.turn !== HUMAN_COLOR) return;

  const piece = gameState.board[row][col];

  if (selectedSquare) {
    const chosenMove = legalTargets.find((move) => move.row === row && move.col === col);
    if (chosenMove) {
      applyMove(selectedSquare, chosenMove);
      clearSelection();
      renderBoard();

      if (!hasAnyLegalMove(gameState, gameState.turn)) {
        return;
      }

      await requestEngineMove();
      return;
    }
  }

  if (piece && piece[0] === gameState.turn) {
    selectedSquare = { row, col };
    legalTargets = getLegalMovesForPiece(gameState, row, col);
  } else {
    clearSelection();
  }

  renderBoard();
}

async function requestEngineMove() {
  engineThinking = true;
  renderBoard();

  try {
    const response = await fetch(`${API_BASE}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        moves: gameState.uciHistory,
        depth: ENGINE_DEPTH,
        time_ms: ENGINE_TIME_MS,
        use_book: USE_OPENING_BOOK,
        max_book_ply: MAX_BOOK_PLY,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      if (payload.error !== "game over") {
        throw new Error(payload.detail || payload.error || "Engine request failed.");
      }
      return;
    }

    applyUciMove(payload.move);
    clearSelection();
  } catch (error) {
    console.error(error);
    alert("Could not reach the chess engine. Make sure the backend is running.");
  } finally {
    engineThinking = false;
    renderBoard();
  }
}

function clearSelection() {
  selectedSquare = null;
  legalTargets = [];
}

function applyMove(from, move) {
  const piece = gameState.board[from.row][from.col];
  const targetPiece = gameState.board[move.row][move.col];
  const nextBoard = cloneBoard(gameState.board);
  const nextState = {
    ...gameState,
    board: nextBoard,
    moveHistory: [...gameState.moveHistory],
    uciHistory: [...gameState.uciHistory],
    castling: {
      w: { ...gameState.castling.w },
      b: { ...gameState.castling.b },
    },
    enPassant: null,
  };

  nextBoard[from.row][from.col] = null;

  if (move.enPassantCapture) {
    nextBoard[move.enPassantCapture.row][move.enPassantCapture.col] = null;
  }

  if (move.castle) {
    const rookFromCol = move.castle === "king" ? 7 : 0;
    const rookToCol = move.castle === "king" ? 5 : 3;
    nextBoard[move.row][rookToCol] = nextBoard[move.row][rookFromCol];
    nextBoard[move.row][rookFromCol] = null;
  }

  let finalPiece = piece;
  if (piece[1] === "p" && (move.row === 0 || move.row === 7)) {
    finalPiece = `${piece[0]}q`;
  }
  nextBoard[move.row][move.col] = finalPiece;

  updateCastlingRights(nextState, piece, from, move, targetPiece);

  if (piece[1] === "p" && Math.abs(move.row - from.row) === 2) {
    nextState.enPassant = {
      row: (move.row + from.row) / 2,
      col: from.col,
    };
  }

  nextState.uciHistory.push(toUciMove(from, move, piece));
  nextState.turn = nextState.turn === "w" ? "b" : "w";
  nextState.moveHistory.push(formatMove(piece, from, move, targetPiece));
  gameState = nextState;
}

function applyUciMove(uci) {
  const from = fromSquare(uci.slice(0, 2));
  const to = fromSquare(uci.slice(2, 4));
  const promotion = uci[4] || null;
  const piece = gameState.board[from.row][from.col];

  if (!piece) {
    throw new Error(`No piece found on ${uci.slice(0, 2)}.`);
  }

  const legalMove = getLegalMovesForPiece(gameState, from.row, from.col).find((move) => (
    move.row === to.row &&
    move.col === to.col &&
    promotionIsSupported(piece, move, promotion)
  ));

  if (!legalMove) {
    throw new Error(`Illegal engine move received: ${uci}`);
  }

  applyMove(from, legalMove);
}

function promotionIsSupported(piece, move, promotion) {
  if (piece[1] !== "p" || (move.row !== 0 && move.row !== 7)) {
    return true;
  }
  return !promotion || promotion === "q";
}

function toUciMove(from, move, piece) {
  let uci = `${toSquare(from.row, from.col)}${toSquare(move.row, move.col)}`;
  if (piece[1] === "p" && (move.row === 0 || move.row === 7)) {
    uci += "q";
  }
  return uci;
}

function fromSquare(square) {
  return {
    row: 8 - Number(square[1]),
    col: FILES.indexOf(square[0]),
  };
}

function updateCastlingRights(state, piece, from, move, capturedPiece) {
  const color = piece[0];
  if (piece[1] === "k") {
    state.castling[color].kingSide = false;
    state.castling[color].queenSide = false;
  }

  if (piece[1] === "r") {
    if (from.row === 7 && from.col === 0) state.castling.w.queenSide = false;
    if (from.row === 7 && from.col === 7) state.castling.w.kingSide = false;
    if (from.row === 0 && from.col === 0) state.castling.b.queenSide = false;
    if (from.row === 0 && from.col === 7) state.castling.b.kingSide = false;
  }

  if (capturedPiece === "wr") {
    if (move.row === 7 && move.col === 0) state.castling.w.queenSide = false;
    if (move.row === 7 && move.col === 7) state.castling.w.kingSide = false;
  }

  if (capturedPiece === "br") {
    if (move.row === 0 && move.col === 0) state.castling.b.queenSide = false;
    if (move.row === 0 && move.col === 7) state.castling.b.kingSide = false;
  }
}

function getLegalMovesForPiece(state, row, col) {
  const piece = state.board[row][col];
  if (!piece || piece[0] !== state.turn) return [];

  const pseudoLegalMoves = getPseudoLegalMoves(state, row, col, true);
  return pseudoLegalMoves.filter((move) => !wouldLeaveKingInCheck(state, { row, col }, move));
}

function hasAnyLegalMove(state, color) {
  const snapshot = state.turn;
  state.turn = color;

  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const piece = state.board[row][col];
      if (!piece || piece[0] !== color) continue;
      if (getLegalMovesForPiece(state, row, col).length > 0) {
        state.turn = snapshot;
        return true;
      }
    }
  }

  state.turn = snapshot;
  return false;
}

function wouldLeaveKingInCheck(state, from, move) {
  const simulated = simulateMove(state, from, move);
  return isKingInCheck(simulated, state.turn);
}

function simulateMove(state, from, move) {
  const cloned = {
    board: cloneBoard(state.board),
    turn: state.turn,
    enPassant: state.enPassant ? { ...state.enPassant } : null,
    castling: {
      w: { ...state.castling.w },
      b: { ...state.castling.b },
    },
  };

  const piece = cloned.board[from.row][from.col];
  cloned.board[from.row][from.col] = null;

  if (move.enPassantCapture) {
    cloned.board[move.enPassantCapture.row][move.enPassantCapture.col] = null;
  }

  if (move.castle) {
    const rookFromCol = move.castle === "king" ? 7 : 0;
    const rookToCol = move.castle === "king" ? 5 : 3;
    cloned.board[move.row][rookToCol] = cloned.board[move.row][rookFromCol];
    cloned.board[move.row][rookFromCol] = null;
  }

  cloned.board[move.row][move.col] = piece[1] === "p" && (move.row === 0 || move.row === 7) ? `${piece[0]}q` : piece;
  return cloned;
}

function isKingInCheck(state, color) {
  const kingSquare = findKing(state.board, color);
  if (!kingSquare) return false;

  const enemyColor = color === "w" ? "b" : "w";
  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const piece = state.board[row][col];
      if (!piece || piece[0] !== enemyColor) continue;
      const moves = getPseudoLegalMoves(state, row, col, false);
      if (moves.some((move) => move.row === kingSquare.row && move.col === kingSquare.col)) {
        return true;
      }
    }
  }
  return false;
}

function findKing(board, color) {
  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      if (board[row][col] === `${color}k`) return { row, col };
    }
  }
  return null;
}

function getPseudoLegalMoves(state, row, col, includeCastling) {
  const piece = state.board[row][col];
  if (!piece) return [];

  const color = piece[0];
  const type = piece[1];

  if (type === "p") return getPawnMoves(state, row, col, color);
  if (type === "n") return getKnightMoves(state, row, col, color);
  if (type === "b") return getSlidingMoves(state, row, col, color, [[1, 1], [1, -1], [-1, 1], [-1, -1]]);
  if (type === "r") return getSlidingMoves(state, row, col, color, [[1, 0], [-1, 0], [0, 1], [0, -1]]);
  if (type === "q") return getSlidingMoves(state, row, col, color, [[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]]);
  if (type === "k") return getKingMoves(state, row, col, color, includeCastling);

  return [];
}

function getPawnMoves(state, row, col, color) {
  const moves = [];
  const direction = color === "w" ? -1 : 1;
  const startRow = color === "w" ? 6 : 1;
  const nextRow = row + direction;

  if (isInside(nextRow, col) && !state.board[nextRow][col]) {
    moves.push({ row: nextRow, col, capture: false });
    const jumpRow = row + direction * 2;
    if (row === startRow && !state.board[jumpRow][col]) {
      moves.push({ row: jumpRow, col, capture: false });
    }
  }

  for (const deltaCol of [-1, 1]) {
    const captureCol = col + deltaCol;
    if (!isInside(nextRow, captureCol)) continue;
    const target = state.board[nextRow][captureCol];
    if (target && target[0] !== color) {
      moves.push({ row: nextRow, col: captureCol, capture: true });
    }
    if (state.enPassant && state.enPassant.row === nextRow && state.enPassant.col === captureCol) {
      moves.push({
        row: nextRow,
        col: captureCol,
        capture: true,
        enPassantCapture: { row, col: captureCol },
      });
    }
  }

  return moves;
}

function getKnightMoves(state, row, col, color) {
  const jumps = [
    [-2, -1], [-2, 1], [-1, -2], [-1, 2],
    [1, -2], [1, 2], [2, -1], [2, 1],
  ];

  return jumps
    .map(([dr, dc]) => ({ row: row + dr, col: col + dc }))
    .filter(({ row: nextRow, col: nextCol }) => isInside(nextRow, nextCol))
    .filter(({ row: nextRow, col: nextCol }) => !state.board[nextRow][nextCol] || state.board[nextRow][nextCol][0] !== color)
    .map(({ row: nextRow, col: nextCol }) => ({
      row: nextRow,
      col: nextCol,
      capture: Boolean(state.board[nextRow][nextCol]),
    }));
}

function getSlidingMoves(state, row, col, color, directions) {
  const moves = [];

  for (const [dr, dc] of directions) {
    let nextRow = row + dr;
    let nextCol = col + dc;

    while (isInside(nextRow, nextCol)) {
      const target = state.board[nextRow][nextCol];
      if (!target) {
        moves.push({ row: nextRow, col: nextCol, capture: false });
      } else {
        if (target[0] !== color) {
          moves.push({ row: nextRow, col: nextCol, capture: true });
        }
        break;
      }
      nextRow += dr;
      nextCol += dc;
    }
  }

  return moves;
}

function getKingMoves(state, row, col, color, includeCastling) {
  const moves = [];

  for (let dr = -1; dr <= 1; dr += 1) {
    for (let dc = -1; dc <= 1; dc += 1) {
      if (dr === 0 && dc === 0) continue;
      const nextRow = row + dr;
      const nextCol = col + dc;
      if (!isInside(nextRow, nextCol)) continue;
      const target = state.board[nextRow][nextCol];
      if (!target || target[0] !== color) {
        moves.push({ row: nextRow, col: nextCol, capture: Boolean(target) });
      }
    }
  }

  if (!includeCastling) return moves;

  const rights = state.castling[color];
  const homeRow = color === "w" ? 7 : 0;
  if (row !== homeRow || col !== 4) return moves;
  if (isKingInCheck(state, color)) return moves;

  if (
    rights.kingSide &&
    !state.board[homeRow][5] &&
    !state.board[homeRow][6] &&
    !isSquareAttacked(state, homeRow, 5, color) &&
    !isSquareAttacked(state, homeRow, 6, color)
  ) {
    moves.push({ row: homeRow, col: 6, capture: false, castle: "king" });
  }

  if (
    rights.queenSide &&
    !state.board[homeRow][1] &&
    !state.board[homeRow][2] &&
    !state.board[homeRow][3] &&
    !isSquareAttacked(state, homeRow, 2, color) &&
    !isSquareAttacked(state, homeRow, 3, color)
  ) {
    moves.push({ row: homeRow, col: 2, capture: false, castle: "queen" });
  }

  return moves;
}

function isSquareAttacked(state, row, col, defendingColor) {
  const enemyColor = defendingColor === "w" ? "b" : "w";
  for (let r = 0; r < 8; r += 1) {
    for (let c = 0; c < 8; c += 1) {
      const piece = state.board[r][c];
      if (!piece || piece[0] !== enemyColor) continue;
      const moves = getPseudoLegalMoves(state, r, c, false);
      if (moves.some((move) => move.row === row && move.col === col)) {
        return true;
      }
    }
  }
  return false;
}

function formatMove(piece, from, move, targetPiece) {
  const pieceLetterMap = { p: "", n: "N", b: "B", r: "R", q: "Q", k: "K" };

  if (move.castle === "king") return "O-O";
  if (move.castle === "queen") return "O-O-O";

  const pieceLetter = pieceLetterMap[piece[1]];
  const capture = targetPiece || move.enPassantCapture ? "x" : "";
  const destination = toSquare(move.row, move.col);

  if (piece[1] === "p") {
    const pawnPrefix = capture ? FILES[from.col] : "";
    const promotion = move.row === 0 || move.row === 7 ? "=Q" : "";
    return `${pawnPrefix}${capture}${destination}${promotion}`;
  }

  return `${pieceLetter}${capture}${destination}`;
}

function toSquare(row, col) {
  return `${FILES[col]}${8 - row}`;
}

function describePiece(piece) {
  const color = piece[0] === "w" ? "white" : "black";
  const names = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };
  return `${color} ${names[piece[1]]}`;
}

function isInside(row, col) {
  return row >= 0 && row < 8 && col >= 0 && col < 8;
}

function cloneBoard(board) {
  return board.map((rank) => [...rank]);
}

function resetBoard() {
  engineThinking = false;
  gameState = createInitialState();
  clearSelection();
  renderBoard();
}

resetButton.addEventListener("click", resetBoard);

createSquareLabels();
renderBoard();
