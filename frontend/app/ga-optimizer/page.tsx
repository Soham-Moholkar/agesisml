'use client';

import { useState } from 'react';
import { apiFetch } from '@/lib/api';
import { Dna, Play, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function GAOptimizerPage() {
  const [datasetId, setDatasetId] = useState('');
  const [targetCol, setTargetCol] = useState('');
  const [modelType, setModelType] = useState('dt');
  const [popSize, setPopSize] = useState(30);
  const [generations, setGenerations] = useState(50);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  const runGA = async () => {
    if (!datasetId || !targetCol) { setError('Enter dataset ID and target column'); return; }
    setRunning(true);
    setError('');
    try {
      const data = await apiFetch('/ga/feature_select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          target_column: targetCol,
          model_type: modelType,
          population_size: popSize,
          generations: generations,
        }),
      });
      setResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <Dna className="h-6 w-6 text-orange-600" /> GA Optimizer
        </h1>
        <p className="text-slate-500 mt-1">Genetic Algorithm feature selection — evolutionary paradigm for optimal feature subsets.</p>
      </div>

      {/* Config */}
      <div className="bg-white rounded-xl border p-6 space-y-4">
        <h2 className="font-semibold">Configuration</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Dataset ID</label>
            <input value={datasetId} onChange={e => setDatasetId(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg text-sm" />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Target Column</label>
            <input value={targetCol} onChange={e => setTargetCol(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg text-sm" />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Model</label>
            <select value={modelType} onChange={e => setModelType(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg text-sm">
              <option value="dt">Decision Tree</option>
              <option value="nb">Naive Bayes</option>
              <option value="svm">SVM</option>
              <option value="knn">kNN</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Population Size</label>
            <input type="number" value={popSize} onChange={e => setPopSize(+e.target.value)}
              min={10} max={100} className="w-full px-3 py-2 border rounded-lg text-sm" />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Generations</label>
            <input type="number" value={generations} onChange={e => setGenerations(+e.target.value)}
              min={10} max={200} className="w-full px-3 py-2 border rounded-lg text-sm" />
          </div>
        </div>

        <button onClick={runGA} disabled={running}
          className="flex items-center gap-2 px-6 py-2.5 bg-orange-600 text-white rounded-lg font-medium hover:bg-orange-700 disabled:opacity-50">
          <Play className="h-4 w-4" />
          {running ? 'Running GA...' : 'Run Feature Selection'}
        </button>
        {error && <p className="text-red-500 text-sm">{error}</p>}
      </div>

      {/* Results */}
      {result && (
        <>
          <div className="bg-white rounded-xl border p-6">
            <h2 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
              <TrendingUp className="h-4 w-4" /> Results
            </h2>
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-orange-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-orange-700">{result.n_selected}</div>
                <div className="text-sm text-orange-600">Features Selected</div>
              </div>
              <div className="bg-green-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-green-700">{(result.best_fitness * 100).toFixed(1)}%</div>
                <div className="text-sm text-green-600">Best Fitness</div>
              </div>
              <div className="bg-blue-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-blue-700">{result.run_id}</div>
                <div className="text-sm text-blue-600">Run ID</div>
              </div>
            </div>

            <h3 className="text-sm font-medium text-slate-700 mb-2">Selected Features</h3>
            <div className="flex flex-wrap gap-2 mb-6">
              {result.selected_features.map((f: string, i: number) => (
                <span key={i} className="px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-sm font-medium">
                  {f}
                </span>
              ))}
            </div>
          </div>

          {/* Convergence Plot */}
          <div className="bg-white rounded-xl border p-6">
            <h3 className="font-semibold text-slate-900 mb-4">Convergence Plot</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={result.history}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="generation" label={{ value: 'Generation', position: 'bottom' }} />
                <YAxis label={{ value: 'Fitness', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="best_fitness" stroke="#f59e0b" name="Best Fitness" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="avg_fitness" stroke="#94a3b8" name="Avg Fitness" strokeWidth={1} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
