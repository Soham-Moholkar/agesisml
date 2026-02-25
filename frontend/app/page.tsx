'use client';

import Link from 'next/link';
import { Shield, Database, FlaskConical, Brain, Dna, BookOpen, Gamepad2, FileText, Target, Sparkles } from 'lucide-react';

export default function Home() {
  const modules = [
    { title: 'Dataset Studio', desc: 'Upload, preview, and prepare datasets', href: '/dataset', icon: Database, color: 'bg-blue-500' },
    { title: 'Model Lab', desc: 'Train DT, NB, SVM, kNN, MLP and compare', href: '/model-lab', icon: FlaskConical, color: 'bg-purple-500' },
    { title: 'Explainability', desc: 'Feature importance & case-based reasoning', href: '/explainability', icon: Brain, color: 'bg-green-500' },
    { title: 'GA Optimizer', desc: 'Genetic Algorithm feature selection', href: '/ga-optimizer', icon: Dna, color: 'bg-orange-500' },
    { title: 'Fuzzy Grading', desc: 'Mamdani fuzzy inference grading system', href: '/fuzzy', icon: Target, color: 'bg-pink-500' },
    { title: 'RL Arena', desc: 'Train Q-learning agent & play TicTacToe', href: '/rl-arena', icon: Gamepad2, color: 'bg-red-500' },
    { title: 'NLP Studio', desc: 'Text sentiment & fake job detection', href: '/nlp', icon: FileText, color: 'bg-teal-500' },
    { title: 'Model Taxonomy', desc: 'ML model categories & DS2012 mapping', href: '/taxonomy', icon: BookOpen, color: 'bg-indigo-500' },
  ];

  return (
    <div className="max-w-6xl mx-auto">
      {/* Hero */}
      <div className="text-center mb-12">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Shield className="h-12 w-12 text-blue-600" />
          <h1 className="text-4xl font-bold text-slate-900">AegisML Suite</h1>
        </div>
        <p className="text-lg text-slate-600 max-w-2xl mx-auto">
          Trustworthy Risk & Decision Support Platform — End-to-end ML for classification, 
          NLP, reinforcement learning, evolutionary optimization, and fuzzy systems.
        </p>
        <div className="mt-4 flex items-center justify-center gap-2">
          <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">DS2012: Machine Learning</span>
          <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">Full Stack</span>
          <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">Recruiter Ready</span>
        </div>
      </div>

      {/* Module Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {modules.map((m) => (
          <Link key={m.href} href={m.href}
            className="group bg-white rounded-xl border border-slate-200 p-5 hover:shadow-lg hover:border-blue-300 transition-all duration-200">
            <div className={`${m.color} w-10 h-10 rounded-lg flex items-center justify-center mb-3`}>
              <m.icon className="h-5 w-5 text-white" />
            </div>
            <h3 className="font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">{m.title}</h3>
            <p className="text-sm text-slate-500 mt-1">{m.desc}</p>
          </Link>
        ))}
      </div>

      {/* Quick Stats */}
      <div className="mt-12 grid grid-cols-3 gap-6">
        <div className="bg-white rounded-xl border p-6 text-center">
          <Sparkles className="h-8 w-8 text-blue-500 mx-auto mb-2" />
          <div className="text-2xl font-bold text-slate-900">5+</div>
          <div className="text-sm text-slate-500">ML Models</div>
        </div>
        <div className="bg-white rounded-xl border p-6 text-center">
          <Shield className="h-8 w-8 text-green-500 mx-auto mb-2" />
          <div className="text-2xl font-bold text-slate-900">4</div>
          <div className="text-sm text-slate-500">DS2012 Units Covered</div>
        </div>
        <div className="bg-white rounded-xl border p-6 text-center">
          <Brain className="h-8 w-8 text-purple-500 mx-auto mb-2" />
          <div className="text-2xl font-bold text-slate-900">3</div>
          <div className="text-sm text-slate-500">Demo Applications</div>
        </div>
      </div>
    </div>
  );
}
