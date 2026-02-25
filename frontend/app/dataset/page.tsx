'use client';

import { useState, useCallback } from 'react';
import { apiUpload, apiFetch } from '@/lib/api';
import { Database, Upload, Eye, Table } from 'lucide-react';

export default function DatasetPage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [preview, setPreview] = useState<any>(null);
  const [error, setError] = useState('');

  const loadDatasets = useCallback(async () => {
    try {
      const data = await apiFetch('/data/list');
      setDatasets(data);
    } catch { }
  }, []);

  useState(() => { loadDatasets(); });

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      const fd = new FormData();
      fd.append('file', file);
      await apiUpload('/data/upload', fd);
      setFile(null);
      await loadDatasets();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  };

  const handlePreview = async (datasetId: string) => {
    try {
      const data = await apiFetch(`/data/${datasetId}/preview`);
      setPreview(data);
    } catch (e: any) {
      setError(e.message);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <Database className="h-6 w-6 text-blue-600" /> Dataset Studio
        </h1>
        <p className="text-slate-500 mt-1">Upload CSV datasets, preview data, and prepare for training.</p>
      </div>

      {/* Upload */}
      <div className="bg-white rounded-xl border p-6">
        <h2 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
          <Upload className="h-4 w-4" /> Upload Dataset
        </h2>
        <div className="flex items-center gap-4">
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="flex-1 text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-50 file:text-blue-700 file:font-medium hover:file:bg-blue-100"
          />
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="px-5 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {uploading ? 'Uploading...' : 'Upload'}
          </button>
        </div>
        {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
      </div>

      {/* Datasets List */}
      <div className="bg-white rounded-xl border p-6">
        <h2 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
          <Table className="h-4 w-4" /> Uploaded Datasets
        </h2>
        {datasets.length === 0 ? (
          <p className="text-slate-400 text-sm">No datasets uploaded yet.</p>
        ) : (
          <div className="space-y-2">
            {datasets.map((ds: any) => (
              <div key={ds.id} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                <div>
                  <span className="font-medium text-slate-900">{ds.filename}</span>
                  <span className="text-slate-400 text-sm ml-3">
                    {ds.n_rows} rows × {ds.n_cols} cols
                  </span>
                  <span className="text-slate-400 text-xs ml-3">ID: {ds.id}</span>
                </div>
                <button
                  onClick={() => handlePreview(ds.id)}
                  className="flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100"
                >
                  <Eye className="h-3 w-3" /> Preview
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Preview */}
      {preview && (
        <div className="bg-white rounded-xl border p-6">
          <h2 className="font-semibold text-slate-900 mb-4">
            Preview: {preview.dataset_id}
            <span className="text-sm text-slate-400 ml-2">({preview.n_rows_total} total rows)</span>
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  {preview.columns.map((col: string) => (
                    <th key={col} className="text-left py-2 px-3 font-medium text-slate-700">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.preview.slice(0, 10).map((row: any, i: number) => (
                  <tr key={i} className="border-b hover:bg-slate-50">
                    {preview.columns.map((col: string) => (
                      <td key={col} className="py-2 px-3 text-slate-600">
                        {String(row[col]).substring(0, 50)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
