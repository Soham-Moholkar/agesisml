'use client';

import { useState, useCallback } from 'react';
import { apiFetch, API_URL } from '@/lib/api';
import { Gamepad2, Play, RotateCcw, Trophy } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

type Cell = 0 | 1 | 2; // 0=empty, 1=X(human), 2=O(agent)

export default function RLArenaPage() {
  // Training state
  const [episodes, setEpisodes] = useState(50000);
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<any>(null);
  const [runId, setRunId] = useState('');
  const [history, setHistory] = useState<any[]>([]);

  // Game state
  const [board, setBoard] = useState<Cell[]>(Array(9).fill(0));
  const [gameOver, setGameOver] = useState(false);
  const [winner, setWinner] = useState<number | null>(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const trainAgent = async () => {
    setTraining(true);
    setError('');
    try {
      const data = await apiFetch('/train/rl/tictactoe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ episodes }),
      });
      setTrainResult(data);
      setRunId(data.run_id);
      // Fetch full history
      const hist = await apiFetch(`/rl/history/${data.run_id}`);
      setHistory(hist.history);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setTraining(false);
    }
  };

  const resetGame = () => {
    setBoard(Array(9).fill(0));
    setGameOver(false);
    setWinner(null);
    setMessage('Your turn (X). Click a cell.');
  };

  const handleCellClick = async (idx: number) => {
    if (gameOver || board[idx] !== 0 || !runId) return;

    // Human move
    const newBoard = [...board];
    newBoard[idx] = 1;
    setBoard(newBoard);

    // Check if human won or board full
    if (checkWin(newBoard, 1)) {
      setGameOver(true);
      setWinner(1);
      setMessage('You win! 🎉');
      return;
    }
    if (newBoard.every(c => c !== 0)) {
      setGameOver(true);
      setWinner(0);
      setMessage("It's a draw!");
      return;
    }

    // Agent move
    try {
      const data = await apiFetch('/rl/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board: newBoard, run_id: runId }),
      });
      setBoard(data.board);
      if (data.done) {
        setGameOver(true);
        if (data.winner === 2) {
          setWinner(2);
          setMessage('Agent wins! 🤖');
        } else {
          setWinner(0);
          setMessage("It's a draw!");
        }
      }
    } catch (e: any) {
      setError(e.message);
    }
  };

  const checkWin = (b: Cell[], player: Cell) => {
    const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
    return lines.some(l => l.every(i => b[i] === player));
  };

  const cellSymbol = (cell: Cell) => {
    if (cell === 1) return <span className="text-blue-600 text-3xl font-bold">X</span>;
    if (cell === 2) return <span className="text-red-600 text-3xl font-bold">O</span>;
    return null;
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <Gamepad2 className="h-6 w-6 text-red-600" /> RL Arena — TicTacToe
        </h1>
        <p className="text-slate-500 mt-1">Train a Q-learning agent and play against it. Agent plays as O.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Training */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl border p-6 space-y-4">
            <h2 className="font-semibold text-slate-900">Train Agent</h2>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Episodes</label>
              <input type="number" value={episodes} onChange={e => setEpisodes(+e.target.value)}
                min={1000} step={5000} className="w-full px-3 py-2 border rounded-lg text-sm" />
            </div>
            <button onClick={trainAgent} disabled={training}
              className="w-full flex items-center justify-center gap-2 py-2.5 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 disabled:opacity-50">
              <Play className="h-4 w-4" />
              {training ? 'Training...' : 'Train Q-Learning Agent'}
            </button>
            {error && <p className="text-red-500 text-sm">{error}</p>}
          </div>

          {trainResult && (
            <div className="bg-white rounded-xl border p-6">
              <h3 className="font-semibold text-slate-900 mb-3">Training Results</h3>
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-green-50 rounded-lg p-3 text-center">
                  <div className="text-lg font-bold text-green-700">{trainResult.stats.wins}</div>
                  <div className="text-xs text-green-600">Wins</div>
                </div>
                <div className="bg-blue-50 rounded-lg p-3 text-center">
                  <div className="text-lg font-bold text-blue-700">{trainResult.stats.draws}</div>
                  <div className="text-xs text-blue-600">Draws</div>
                </div>
                <div className="bg-red-50 rounded-lg p-3 text-center">
                  <div className="text-lg font-bold text-red-700">{trainResult.stats.losses}</div>
                  <div className="text-xs text-red-600">Losses</div>
                </div>
              </div>
              <p className="text-xs text-slate-400 mt-2">Q-table size: {trainResult.q_table_size} states</p>
            </div>
          )}
        </div>

        {/* Game Board */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl border p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-slate-900">Play vs Agent</h2>
              <button onClick={resetGame} className="flex items-center gap-1 px-3 py-1.5 text-sm bg-slate-100 rounded-lg hover:bg-slate-200">
                <RotateCcw className="h-3 w-3" /> Reset
              </button>
            </div>

            {!runId ? (
              <p className="text-slate-400 text-sm text-center py-8">Train an agent first to play.</p>
            ) : (
              <>
                <div className="grid grid-cols-3 gap-2 max-w-64 mx-auto">
                  {board.map((cell, i) => (
                    <button key={i} onClick={() => handleCellClick(i)}
                      className={`w-20 h-20 rounded-lg border-2 flex items-center justify-center transition-colors ${
                        cell === 0 ? 'border-slate-200 hover:border-blue-300 hover:bg-blue-50 cursor-pointer' : 'border-slate-200 cursor-default'
                      } ${gameOver ? 'cursor-default' : ''}`}>
                      {cellSymbol(cell)}
                    </button>
                  ))}
                </div>
                {message && (
                  <div className={`mt-4 p-3 rounded-lg text-center text-sm font-medium ${
                    winner === 1 ? 'bg-green-50 text-green-700' :
                    winner === 2 ? 'bg-red-50 text-red-700' :
                    winner === 0 ? 'bg-blue-50 text-blue-700' :
                    'bg-slate-50 text-slate-600'
                  }`}>
                    {message}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Training Curve */}
      {history.length > 0 && (
        <div className="bg-white rounded-xl border p-6">
          <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <Trophy className="h-4 w-4" /> Training Curve
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="episode" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="win_rate" stroke="#10b981" name="Win Rate" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="draw_rate" stroke="#3b82f6" name="Draw Rate" dot={false} strokeWidth={1} />
              <Line type="monotone" dataKey="loss_rate" stroke="#ef4444" name="Loss Rate" dot={false} strokeWidth={1} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
