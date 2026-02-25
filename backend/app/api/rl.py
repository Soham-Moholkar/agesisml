"""RL TicTacToe endpoints."""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from app.core.schemas import RLMoveRequest
from app.core.config import ARTIFACTS_DIR
from app.ml.rl.tictactoe import load_rl_agent, agent_move, TicTacToeEnv

router = APIRouter()


@router.post("/move")
async def rl_move(req: RLMoveRequest):
    """Get agent's next move given board state."""
    try:
        agent = load_rl_agent(req.run_id)
    except FileNotFoundError:
        raise HTTPException(404, "RL agent not found. Train one first.")

    # Validate board
    if len(req.board) != 9:
        raise HTTPException(400, "Board must have 9 cells")

    move = agent_move(agent, req.board)

    # Check game state after move
    env = TicTacToeEnv()
    env.board = list(req.board)
    if move is not None:
        state, reward, done = env.step(move, 2)
        return {
            "move": move,
            "board": list(state),
            "done": done,
            "winner": env.winner,
        }
    return {
        "move": None,
        "board": req.board,
        "done": True,
        "winner": None,
    }


@router.get("/history/{run_id}")
async def rl_history(run_id: str):
    """Get training history for an RL run."""
    history_path = ARTIFACTS_DIR / run_id / "training_history.json"
    if not history_path.exists():
        raise HTTPException(404, "Training history not found")
    with open(str(history_path)) as f:
        history = json.load(f)
    return {"run_id": run_id, "history": history}
