'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Shield, Database, FlaskConical, Brain, Dna, Target,
  Gamepad2, FileText, BookOpen, Home, BarChart3,
} from 'lucide-react';

const navItems = [
  { href: '/', label: 'Home', icon: Home },
  { href: '/dataset', label: 'Dataset Studio', icon: Database },
  { href: '/model-lab', label: 'Model Lab', icon: FlaskConical },
  { href: '/explainability', label: 'Explainability', icon: Brain },
  { href: '/ga-optimizer', label: 'GA Optimizer', icon: Dna },
  { href: '/fuzzy', label: 'Fuzzy Grading', icon: Target },
  { href: '/rl-arena', label: 'RL Arena', icon: Gamepad2 },
  { href: '/nlp', label: 'NLP Studio', icon: FileText },
  { href: '/taxonomy', label: 'Model Taxonomy', icon: BookOpen },
  { href: '/leaderboard', label: 'Leaderboard', icon: BarChart3 },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-64 bg-white border-r border-slate-200 flex flex-col z-50">
      {/* Logo */}
      <div className="p-5 border-b border-slate-200">
        <Link href="/" className="flex items-center gap-2">
          <Shield className="h-7 w-7 text-blue-600" />
          <span className="font-bold text-lg text-slate-900">AegisML</span>
        </Link>
        <p className="text-xs text-slate-400 mt-1">Decision Support Suite</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              <item.icon className={`h-4 w-4 ${isActive ? 'text-blue-600' : 'text-slate-400'}`} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-200">
        <p className="text-xs text-slate-400">DS2012: Machine Learning</p>
        <p className="text-xs text-slate-400">v1.0.0</p>
      </div>
    </aside>
  );
}
