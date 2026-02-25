'use client';

import { useState } from 'react';
import { apiFetch } from '@/lib/api';
import { FileText, MessageSquare, Play } from 'lucide-react';

export default function NLPPage() {
  const [datasetId, setDatasetId] = useState('');
  const [textCol, setTextCol] = useState('text');
  const [targetCol, setTargetCol] = useState('label');
  const [modelType, setModelType] = useState('nb');
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  // Prediction state
  const [predText, setPredText] = useState('');
  const [prediction, setPrediction] = useState<any>(null);
  const [predicting, setPredicting] = useState(false);

  const trainText = async () => {
    if (!datasetId) { setError('Enter a dataset ID'); return; }
    setTraining(true);
    setError('');
    try {
      const data = await apiFetch('/train/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          text_column: textCol,
          target_column: targetCol,
          model_type: modelType,
        }),
      });
      setResult(data);

      // Auto-activate
      await apiFetch('/registry/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: 'text', run_id: data.run_id }),
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setTraining(false);
    }
  };

  const predict = async () => {
    if (!predText) return;
    setPredicting(true);
    try {
      const data = await apiFetch('/predict/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: predText }),
      });
      setPrediction(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <FileText className="h-6 w-6 text-teal-600" /> NLP Studio
        </h1>
        <p className="text-slate-500 mt-1">Text classification for sentiment analysis and fake job detection.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Training */}
        <div className="bg-white rounded-xl border p-6 space-y-4">
          <h2 className="font-semibold text-slate-900">Train Text Model</h2>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Dataset ID</label>
              <input value={datasetId} onChange={e => setDatasetId(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg text-sm" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Text Column</label>
                <input value={textCol} onChange={e => setTextCol(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg text-sm" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Label Column</label>
                <input value={targetCol} onChange={e => setTargetCol(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg text-sm" />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Model</label>
              <select value={modelType} onChange={e => setModelType(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg text-sm">
                <option value="nb">Naive Bayes (MultinomialNB)</option>
                <option value="svm">SVM (LinearSVC)</option>
                <option value="mlp">MLP (Neural Net)</option>
              </select>
            </div>
            <button onClick={trainText} disabled={training}
              className="w-full flex items-center justify-center gap-2 py-2.5 bg-teal-600 text-white rounded-lg font-medium hover:bg-teal-700 disabled:opacity-50">
              <Play className="h-4 w-4" />
              {training ? 'Training...' : 'Train Model'}
            </button>
          </div>
          {error && <p className="text-red-500 text-sm mt-2">{error}</p>}

          {result && (
            <div className="bg-teal-50 rounded-lg p-4 mt-4">
              <p className="text-sm font-medium text-teal-800">Run ID: {result.run_id}</p>
              <div className="grid grid-cols-2 gap-2 mt-2 text-sm text-teal-700">
                <span>Accuracy: {(result.metrics.accuracy * 100).toFixed(1)}%</span>
                <span>F1: {(result.metrics.f1 * 100).toFixed(1)}%</span>
                <span>Precision: {(result.metrics.precision * 100).toFixed(1)}%</span>
                <span>Recall: {(result.metrics.recall * 100).toFixed(1)}%</span>
              </div>
            </div>
          )}
        </div>

        {/* Prediction */}
        <div className="bg-white rounded-xl border p-6 space-y-4">
          <h2 className="font-semibold text-slate-900 flex items-center gap-2">
            <MessageSquare className="h-4 w-4" /> Live Prediction
          </h2>
          <textarea value={predText} onChange={e => setPredText(e.target.value)}
            rows={4} className="w-full px-3 py-2 border rounded-lg text-sm"
            placeholder="Enter text to classify..." />
          <button onClick={predict} disabled={predicting || !predText}
            className="w-full py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50">
            {predicting ? 'Predicting...' : 'Predict'}
          </button>

          {prediction && (
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="text-center">
                <span className={`inline-block px-4 py-2 rounded-full text-lg font-bold ${
                  prediction.prediction === 'positive' || prediction.prediction_idx === 1
                    ? 'bg-green-100 text-green-700'
                    : 'bg-red-100 text-red-700'
                }`}>
                  {prediction.prediction}
                </span>
              </div>
              <div className="mt-3 text-sm text-slate-600 text-center">
                Confidence: {(Math.max(...prediction.probabilities) * 100).toFixed(1)}%
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
