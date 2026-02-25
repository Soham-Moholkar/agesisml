'use client';

import { BookOpen, Layers, GitBranch, BarChart3 } from 'lucide-react';

const taxonomy = [
  {
    category: 'By Learning Type',
    items: [
      { name: 'Supervised', desc: 'Labeled training data → predict labels', models: 'DT, NB, SVM, kNN, MLP', inProject: true },
      { name: 'Unsupervised', desc: 'Find patterns without labels', models: 'K-Means, PCA, DBSCAN', inProject: false },
      { name: 'Semi-supervised', desc: 'Mix of labeled + unlabeled data', models: 'Self-training, Label Propagation', inProject: false },
      { name: 'Reinforcement', desc: 'Agent learns from environment rewards', models: 'Q-Learning, DQN', inProject: true },
    ],
  },
  {
    category: 'By Model Type',
    items: [
      { name: 'Probabilistic', desc: 'Models based on probability distributions', models: 'Naive Bayes, Bayesian Networks', inProject: true },
      { name: 'Geometric', desc: 'Models based on geometric boundaries', models: 'SVM, Perceptron', inProject: true },
      { name: 'Logical / Symbolic', desc: 'Rule-based decision logic', models: 'Decision Trees, Rule Classifiers', inProject: true },
      { name: 'Grouping / Instance', desc: 'Based on similarity or distance', models: 'kNN, Case-Based Reasoning', inProject: true },
      { name: 'Connectionist', desc: 'Neural network architectures', models: 'MLP, RNN/LSTM, CNN', inProject: true },
      { name: 'Evolutionary', desc: 'Optimization via natural selection', models: 'Genetic Algorithm', inProject: true },
      { name: 'Fuzzy', desc: 'Fuzzy logic & approximate reasoning', models: 'Mamdani FIS', inProject: true },
    ],
  },
  {
    category: 'By Task Type',
    items: [
      { name: 'Predictive', desc: 'Predict future outcomes from data', models: 'All classifiers, regressors', inProject: true },
      { name: 'Descriptive', desc: 'Describe/summarize patterns in data', models: 'Clustering, Association Rules', inProject: false },
    ],
  },
  {
    category: 'By Parameterization',
    items: [
      { name: 'Parametric', desc: 'Fixed number of parameters', models: 'NB, SVM (linear), MLP', inProject: true },
      { name: 'Non-parametric', desc: 'Parameters grow with data', models: 'kNN, Decision Trees', inProject: true },
    ],
  },
];

export default function TaxonomyPage() {
  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <BookOpen className="h-6 w-6 text-indigo-600" /> Model Taxonomy
        </h1>
        <p className="text-slate-500 mt-1">Classification of ML model types as per DS2012 curriculum (Unit 1).</p>
      </div>

      {taxonomy.map((group) => (
        <div key={group.category} className="bg-white rounded-xl border p-6">
          <h2 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <Layers className="h-4 w-4 text-indigo-500" /> {group.category}
          </h2>
          <div className="grid gap-3">
            {group.items.map((item) => (
              <div key={item.name} className={`rounded-lg p-4 border ${item.inProject ? 'bg-indigo-50 border-indigo-200' : 'bg-slate-50 border-slate-200'}`}>
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-slate-900">{item.name}</h3>
                  {item.inProject && (
                    <span className="px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded text-xs font-medium">
                      In AegisML
                    </span>
                  )}
                </div>
                <p className="text-sm text-slate-600 mt-1">{item.desc}</p>
                <p className="text-xs text-slate-400 mt-1">Examples: {item.models}</p>
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* DS2012 Unit Mapping */}
      <div className="bg-white rounded-xl border p-6">
        <h2 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-indigo-500" /> DS2012 Unit Mapping
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2 px-3">Unit</th>
                <th className="text-left py-2 px-3">Topics</th>
                <th className="text-left py-2 px-3">AegisML Module</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b">
                <td className="py-2 px-3 font-medium">Unit 1</td>
                <td className="py-2 px-3">ML Introduction, Problem Types, Taxonomy</td>
                <td className="py-2 px-3">This page + Home</td>
              </tr>
              <tr className="border-b">
                <td className="py-2 px-3 font-medium">Unit 2</td>
                <td className="py-2 px-3">DT, NB, SVM, ANN, Evaluation Metrics</td>
                <td className="py-2 px-3">Model Lab + Metrics</td>
              </tr>
              <tr className="border-b">
                <td className="py-2 px-3 font-medium">Unit 3</td>
                <td className="py-2 px-3">GA, kNN, Fuzzy, RL, Transfer Learning</td>
                <td className="py-2 px-3">GA Optimizer + Fuzzy + RL Arena</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-medium">Unit 4</td>
                <td className="py-2 px-3">NLP, Healthcare, Applications</td>
                <td className="py-2 px-3">NLP Studio + Healthcare Module</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
