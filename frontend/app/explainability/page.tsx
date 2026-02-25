'use client';

import { useState } from 'react';
import { apiFetch, API_URL } from '@/lib/api';
import { Brain, Search, Users } from 'lucide-react';

export default function ExplainabilityPage() {
  const [runId, setRunId] = useState('');
  const [importance, setImportance] = useState<any>(null);
  const [cases, setCases] = useState<any>(null);
  const [features, setFeatures] = useState('');
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchImportance = async () => {
    if (!runId) return;
    setLoading(true);
    setError('');
    try {
      const data = await apiFetch('/explain/tabular/permutation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId, n_repeats: 10 }),
      });
      setImportance(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchCases = async () => {
    if (!runId || !features) return;
    setLoading(true);
    setError('');
    try {
      let parsedFeatures: Record<string, number> = {};
      try {
        parsedFeatures = JSON.parse(features);
      } catch {
        setError('Features must be valid JSON, e.g. {"age": 30, "income": 50000}');
        setLoading(false);
        return;
      }
      const data = await apiFetch('/explain/tabular/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId, features: parsedFeatures, k }),
      });
      setCases(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <Brain className="h-6 w-6 text-green-600" /> Explainability
        </h1>
        <p className="text-slate-500 mt-1">Understand model predictions with feature importance and case-based reasoning.</p>
      </div>

      {/* Config */}
      <div className="bg-white rounded-xl border p-6 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Run ID</label>
            <input value={runId} onChange={e => setRunId(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg text-sm" placeholder="Enter run ID from Model Lab" />
          </div>
        </div>

        {/* Permutation Importance */}
        <div className="border rounded-lg p-4">
          <h3 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
            <Search className="h-4 w-4" /> Permutation Importance
          </h3>
          <button onClick={fetchImportance} disabled={loading || !runId}
            className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 disabled:opacity-50">
            {loading ? 'Computing...' : 'Compute Importance'}
          </button>

          {importance && (
            <div className="mt-4">
              <img src={`${API_URL}/runs/${runId}/artifacts/permutation_importance.png`}
                alt="Permutation Importance" className="rounded-lg border max-w-lg" />
              <div className="mt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-2">Top Features</h4>
                <div className="space-y-1">
                  {Object.entries(importance.importances).slice(0, 10).map(([feat, imp]: [string, any]) => (
                    <div key={feat} className="flex items-center gap-3">
                      <span className="text-sm text-slate-600 w-40 truncate">{feat}</span>
                      <div className="flex-1 bg-slate-100 rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full"
                          style={{ width: `${Math.max(0, imp * 1000)}%` }} />
                      </div>
                      <span className="text-xs text-slate-400 w-16 text-right">{(imp as number).toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Case-Based Reasoning */}
        <div className="border rounded-lg p-4">
          <h3 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
            <Users className="h-4 w-4" /> Case-Based Reasoning
          </h3>
          <div className="space-y-3">
            <div>
              <label className="block text-sm text-slate-600 mb-1">Features (JSON)</label>
              <textarea value={features} onChange={e => setFeatures(e.target.value)}
                rows={3} className="w-full px-3 py-2 border rounded-lg text-sm font-mono"
                placeholder='{"age": 30, "income": 50000, "credit_score": 700}' />
            </div>
            <div className="flex items-center gap-4">
              <div>
                <label className="block text-sm text-slate-600 mb-1">K neighbors</label>
                <input type="number" value={k} onChange={e => setK(+e.target.value)}
                  min={1} max={20} className="w-24 px-3 py-2 border rounded-lg text-sm" />
              </div>
              <button onClick={fetchCases} disabled={loading || !runId || !features}
                className="mt-5 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
                Find Similar Cases
              </button>
            </div>
          </div>

          {cases && (
            <div className="mt-4">
              <p className="text-sm text-slate-600 mb-3">{cases.explanation}</p>
              <div className="space-y-2">
                {cases.similar_cases.map((c: any, i: number) => (
                  <div key={i} className="bg-slate-50 rounded-lg p-3">
                    <div className="flex items-center gap-3 mb-1">
                      <span className="text-sm font-medium text-slate-700">Case #{i + 1}</span>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        c.label === 1 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                      }`}>
                        Label: {c.label}
                      </span>
                      <span className="text-xs text-slate-400">Distance: {c.distance.toFixed(4)}</span>
                    </div>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {Object.entries(c.feature_values).slice(0, 8).map(([f, v]: [string, any]) => (
                        <span key={f} className="text-xs bg-white px-2 py-1 rounded border text-slate-600">
                          {f}: {typeof v === 'number' ? v.toFixed(2) : v}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {error && <p className="text-red-500 text-sm">{error}</p>}
      </div>
    </div>
  );
}
