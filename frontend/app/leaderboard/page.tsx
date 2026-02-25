'use client';

import { useState, useEffect } from 'react';
import { apiFetch } from '@/lib/api';
import { BarChart3, Trophy } from 'lucide-react';

export default function LeaderboardPage() {
  const [runs, setRuns] = useState<any[]>([]);
  const [filter, setFilter] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await apiFetch(`/runs/${filter ? `?task=${filter}` : ''}`);
        setRuns(data.runs || []);
      } catch { }
      setLoading(false);
    };
    load();
  }, [filter]);

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <BarChart3 className="h-6 w-6 text-blue-600" /> Leaderboard
        </h1>
        <p className="text-slate-500 mt-1">All training runs ranked by performance.</p>
      </div>

      <div className="flex gap-2 mb-4">
        {['', 'tabular', 'text', 'rl_tictactoe', 'ga_feature_selection'].map(t => (
          <button key={t} onClick={() => setFilter(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium border ${
              filter === t ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-slate-600 border-slate-200'
            }`}>
            {t || 'All'}
          </button>
        ))}
      </div>

      <div className="bg-white rounded-xl border overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-slate-400">Loading...</div>
        ) : runs.length === 0 ? (
          <div className="p-8 text-center text-slate-400">No runs yet. Train a model first.</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-slate-50">
                <th className="text-left py-3 px-4">#</th>
                <th className="text-left py-3 px-4">Run ID</th>
                <th className="text-left py-3 px-4">Task</th>
                <th className="text-left py-3 px-4">Model</th>
                <th className="text-left py-3 px-4">Accuracy</th>
                <th className="text-left py-3 px-4">F1</th>
                <th className="text-left py-3 px-4">Created</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run, i) => (
                <tr key={run.id} className="border-b hover:bg-slate-50">
                  <td className="py-3 px-4">
                    {i === 0 && <Trophy className="h-4 w-4 text-yellow-500 inline" />}
                    {i > 0 && <span className="text-slate-400">{i + 1}</span>}
                  </td>
                  <td className="py-3 px-4 font-mono text-xs">{run.id}</td>
                  <td className="py-3 px-4">
                    <span className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs">{run.task}</span>
                  </td>
                  <td className="py-3 px-4 font-medium">{run.model_type}</td>
                  <td className="py-3 px-4">
                    {run.metrics?.accuracy != null ? `${(run.metrics.accuracy * 100).toFixed(1)}%` : 'N/A'}
                  </td>
                  <td className="py-3 px-4">
                    {run.metrics?.f1 != null ? `${(run.metrics.f1 * 100).toFixed(1)}%` : 'N/A'}
                  </td>
                  <td className="py-3 px-4 text-xs text-slate-400">{run.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
