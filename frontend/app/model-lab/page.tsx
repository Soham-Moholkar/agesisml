'use client';

import { useState } from 'react';
import { apiFetch } from '@/lib/api';
import { API_URL } from '@/lib/api';
import { FlaskConical, Play, Trophy, GitBranch } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts';

const MODEL_TYPES = [
  { value: 'dt', label: 'Decision Tree', color: '#3b82f6' },
  { value: 'nb', label: 'Naive Bayes', color: '#8b5cf6' },
  { value: 'svm', label: 'SVM', color: '#ef4444' },
  { value: 'knn', label: 'kNN', color: '#f59e0b' },
  { value: 'mlp', label: 'MLP (Neural Net)', color: '#10b981' },
];

export default function ModelLabPage() {
  const [datasetId, setDatasetId] = useState('');
  const [targetCol, setTargetCol] = useState('');
  const [selectedModels, setSelectedModels] = useState<string[]>(['dt']);
  const [testSize, setTestSize] = useState(0.2);
  const [training, setTraining] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState('');
  const [dtRules, setDtRules] = useState<string[]>([]);
  const [hyperparams, setHyperparams] = useState<Record<string, any>>({});

  const toggleModel = (m: string) => {
    setSelectedModels(prev =>
      prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]
    );
  };

  const trainAll = async () => {
    if (!datasetId || !targetCol) { setError('Enter dataset ID and target column'); return; }
    setTraining(true);
    setError('');
    setResults([]);
    setDtRules([]);

    try {
      const newResults = [];
      for (const mt of selectedModels) {
        const res = await apiFetch('/train/tabular', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_id: datasetId,
            target_column: targetCol,
            model_type: mt,
            test_size: testSize,
            hyperparams: hyperparams[mt] || {},
          }),
        });
        newResults.push({ ...res, model_type: mt });
        if (mt === 'dt' && res.rules) {
          setDtRules(res.rules);
        }
      }
      setResults(newResults);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setTraining(false);
    }
  };

  const metricsData = results.map(r => ({
    model: MODEL_TYPES.find(m => m.value === r.model_type)?.label || r.model_type,
    Accuracy: +(r.metrics.accuracy * 100).toFixed(1),
    Precision: +(r.metrics.precision * 100).toFixed(1),
    Recall: +(r.metrics.recall * 100).toFixed(1),
    F1: +(r.metrics.f1 * 100).toFixed(1),
    'ROC-AUC': +((r.metrics.roc_auc || 0) * 100).toFixed(1),
  }));

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <FlaskConical className="h-6 w-6 text-purple-600" /> Model Lab
        </h1>
        <p className="text-slate-500 mt-1">Train, compare, and evaluate classification models.</p>
      </div>

      {/* Config */}
      <div className="bg-white rounded-xl border p-6 space-y-4">
        <h2 className="font-semibold text-slate-900">Training Configuration</h2>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Dataset ID</label>
            <input value={datasetId} onChange={e => setDatasetId(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg text-sm" placeholder="e.g. abc12345" />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Target Column</label>
            <input value={targetCol} onChange={e => setTargetCol(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg text-sm" placeholder="e.g. is_fraud" />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Test Size</label>
            <input type="number" step="0.05" min="0.1" max="0.5" value={testSize}
              onChange={e => setTestSize(+e.target.value)}
              className="w-full px-3 py-2 border rounded-lg text-sm" />
          </div>
        </div>

        {/* Model Selection */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">Models to Train</label>
          <div className="flex gap-2 flex-wrap">
            {MODEL_TYPES.map(m => (
              <button key={m.value}
                onClick={() => toggleModel(m.value)}
                className={`px-4 py-2 rounded-lg text-sm font-medium border transition-colors ${
                  selectedModels.includes(m.value)
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-slate-600 border-slate-200 hover:border-blue-300'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <button onClick={trainAll} disabled={training}
          className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50">
          <Play className="h-4 w-4" />
          {training ? 'Training...' : `Train ${selectedModels.length} Model(s)`}
        </button>

        {error && <p className="text-red-500 text-sm">{error}</p>}
      </div>

      {/* Results */}
      {results.length > 0 && (
        <>
          {/* Metrics Table */}
          <div className="bg-white rounded-xl border p-6">
            <h2 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
              <Trophy className="h-4 w-4 text-yellow-500" /> Results Comparison
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 px-3">Model</th>
                    <th className="text-left py-2 px-3">Accuracy</th>
                    <th className="text-left py-2 px-3">Precision</th>
                    <th className="text-left py-2 px-3">Recall</th>
                    <th className="text-left py-2 px-3">F1</th>
                    <th className="text-left py-2 px-3">ROC-AUC</th>
                    <th className="text-left py-2 px-3">CV Accuracy</th>
                    <th className="text-left py-2 px-3">Run ID</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <tr key={i} className="border-b hover:bg-slate-50">
                      <td className="py-2 px-3 font-medium">{MODEL_TYPES.find(m => m.value === r.model_type)?.label}</td>
                      <td className="py-2 px-3">{(r.metrics.accuracy * 100).toFixed(1)}%</td>
                      <td className="py-2 px-3">{(r.metrics.precision * 100).toFixed(1)}%</td>
                      <td className="py-2 px-3">{(r.metrics.recall * 100).toFixed(1)}%</td>
                      <td className="py-2 px-3">{(r.metrics.f1 * 100).toFixed(1)}%</td>
                      <td className="py-2 px-3">{r.metrics.roc_auc ? (r.metrics.roc_auc * 100).toFixed(1) + '%' : 'N/A'}</td>
                      <td className="py-2 px-3">{r.metrics.cv_accuracy_mean ? (r.metrics.cv_accuracy_mean * 100).toFixed(1) + '%' : 'N/A'}</td>
                      <td className="py-2 px-3 font-mono text-xs text-slate-400">{r.run_id}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-white rounded-xl border p-6">
              <h3 className="font-semibold text-slate-900 mb-4">Metrics Comparison</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Accuracy" fill="#3b82f6" />
                  <Bar dataKey="F1" fill="#10b981" />
                  <Bar dataKey="ROC-AUC" fill="#f59e0b" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-xl border p-6">
              <h3 className="font-semibold text-slate-900 mb-4">Radar View</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={metricsData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="model" tick={{ fontSize: 11 }} />
                  <PolarRadiusAxis domain={[0, 100]} />
                  <Radar name="Accuracy" dataKey="Accuracy" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                  <Radar name="F1" dataKey="F1" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion Matrices */}
          <div className="bg-white rounded-xl border p-6">
            <h3 className="font-semibold text-slate-900 mb-4">Artifacts</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {results.map(r => (
                <div key={r.run_id} className="text-center">
                  <p className="text-sm font-medium text-slate-700 mb-2">{MODEL_TYPES.find(m => m.value === r.model_type)?.label}</p>
                  <img src={`${API_URL}/runs/${r.run_id}/artifacts/confusion_matrix.png`}
                    alt="Confusion Matrix" className="rounded-lg border w-full" />
                </div>
              ))}
            </div>
          </div>

          {/* DT Rules */}
          {dtRules.length > 0 && (
            <div className="bg-white rounded-xl border p-6">
              <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
                <GitBranch className="h-4 w-4" /> Decision Tree Rules
              </h3>
              <pre className="bg-slate-50 p-4 rounded-lg text-xs text-slate-700 overflow-x-auto max-h-64 overflow-y-auto">
                {dtRules.join('\n')}
              </pre>
            </div>
          )}
        </>
      )}
    </div>
  );
}
