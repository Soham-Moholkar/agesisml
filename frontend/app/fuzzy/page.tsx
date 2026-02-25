'use client';

import { useState } from 'react';
import { apiFetch } from '@/lib/api';
import { Target, Sliders } from 'lucide-react';

export default function FuzzyPage() {
  const [attendance, setAttendance] = useState(75);
  const [assignment, setAssignment] = useState(70);
  const [quiz, setQuiz] = useState(65);
  const [project, setProject] = useState(80);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const evaluate = async () => {
    setLoading(true);
    try {
      const data = await apiFetch('/fuzzy/grade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ attendance, assignment, quiz, project }),
      });
      setResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const gradeColors: Record<string, string> = {
    A: 'bg-green-500', B: 'bg-blue-500', C: 'bg-yellow-500', D: 'bg-orange-500', F: 'bg-red-500',
  };

  const SliderInput = ({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) => (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-slate-700">{label}</label>
        <span className="text-sm font-bold text-slate-900">{value}</span>
      </div>
      <input type="range" min={0} max={100} value={value} onChange={e => onChange(+e.target.value)}
        className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600" />
      <div className="flex justify-between text-xs text-slate-400">
        <span>0</span><span>50</span><span>100</span>
      </div>
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <Target className="h-6 w-6 text-pink-600" /> Fuzzy Grading System
        </h1>
        <p className="text-slate-500 mt-1">Mamdani-style fuzzy inference for educational grading with triangular/trapezoidal membership functions.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Input Sliders */}
        <div className="bg-white rounded-xl border p-6 space-y-6">
          <h2 className="font-semibold text-slate-900 flex items-center gap-2">
            <Sliders className="h-4 w-4" /> Input Parameters
          </h2>
          <SliderInput label="Attendance (%)" value={attendance} onChange={setAttendance} />
          <SliderInput label="Assignment Score (%)" value={assignment} onChange={setAssignment} />
          <SliderInput label="Quiz Score (%)" value={quiz} onChange={setQuiz} />
          <SliderInput label="Project Score (%)" value={project} onChange={setProject} />
          <button onClick={evaluate} disabled={loading}
            className="w-full py-2.5 bg-pink-600 text-white rounded-lg font-medium hover:bg-pink-700 disabled:opacity-50">
            {loading ? 'Evaluating...' : 'Evaluate Grade'}
          </button>
        </div>

        {/* Output */}
        <div className="space-y-4">
          {result ? (
            <>
              {/* Grade Display */}
              <div className="bg-white rounded-xl border p-6 text-center">
                <div className={`inline-flex items-center justify-center w-24 h-24 rounded-full ${gradeColors[result.grade] || 'bg-slate-500'} text-white mb-4`}>
                  <span className="text-4xl font-bold">{result.grade}</span>
                </div>
                <p className="text-lg font-semibold text-slate-900">Numeric Score: {result.numeric_score}</p>
              </div>

              {/* Input Memberships */}
              <div className="bg-white rounded-xl border p-6">
                <h3 className="font-semibold text-slate-900 mb-3">Input Memberships</h3>
                {Object.entries(result.input_memberships).map(([var_name, sets]: [string, any]) => (
                  <div key={var_name} className="mb-3">
                    <p className="text-sm font-medium text-slate-700 capitalize">{var_name}</p>
                    <div className="flex gap-2 mt-1">
                      {Object.entries(sets).map(([set_name, val]: [string, any]) => (
                        <div key={set_name} className="flex items-center gap-1">
                          <span className="text-xs text-slate-500">{set_name}:</span>
                          <div className="w-16 bg-slate-100 rounded-full h-1.5">
                            <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: `${val * 100}%` }} />
                          </div>
                          <span className="text-xs text-slate-400">{(val as number).toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {/* Grade Activations */}
              <div className="bg-white rounded-xl border p-6">
                <h3 className="font-semibold text-slate-900 mb-3">Grade Activations</h3>
                <div className="space-y-2">
                  {Object.entries(result.grade_activations).map(([grade, val]: [string, any]) => (
                    <div key={grade} className="flex items-center gap-3">
                      <span className={`w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-sm ${gradeColors[grade]}`}>{grade}</span>
                      <div className="flex-1 bg-slate-100 rounded-full h-3">
                        <div className={`${gradeColors[grade]} h-3 rounded-full`} style={{ width: `${(val as number) * 100}%` }} />
                      </div>
                      <span className="text-sm text-slate-600 w-12 text-right">{(val as number).toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Fired Rules */}
              <div className="bg-white rounded-xl border p-6">
                <h3 className="font-semibold text-slate-900 mb-3">Fired Rules</h3>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {result.fired_rules.map((rule: any, i: number) => (
                    <div key={i} className="bg-slate-50 rounded-lg p-2 text-xs">
                      <span className="font-medium text-slate-700">→ {rule.grade}</span>
                      <span className="text-slate-400 ml-2">(strength: {rule.strength})</span>
                      <div className="text-slate-500 mt-1">
                        {Object.entries(rule.conditions).map(([k, v]) => `${k}=${v}`).join(', ')}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="bg-white rounded-xl border p-6 text-center text-slate-400">
              <Target className="h-12 w-12 mx-auto mb-3 text-slate-300" />
              <p>Adjust sliders and click Evaluate to see the fuzzy grade.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
