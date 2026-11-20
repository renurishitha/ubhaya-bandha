import type { ActiveTab } from '../types.ts';

interface TopBarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
}

const NAV_ITEMS: Array<{ id: ActiveTab; label: string }> = [
  { id: 'home', label: 'Home' },
  { id: 'predict', label: 'Predict pairs' },
  { id: 'leaderboard', label: 'Leaderboard' },
  { id: 'model-card', label: 'Model Card' },
  { id: 'glossary', label: 'Glossary' },
];

export function TopBar({ activeTab, onSelectTab }: TopBarProps) {
  return (
    <header className="sticky top-0 z-30 bg-[#FAF6F0]/95 backdrop-blur-sm border-b border-[#E3D9CA]">
      <div className="max-w-[1240px] mx-auto px-6 h-16 flex items-center justify-between gap-8">
        {/* Zone 1: Clean Brand Lockup */}
        <button
          type="button"
          onClick={() => onSelectTab('home')}
          className="flex items-center gap-2.5 text-left whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#7C5333]"
        >
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            className="shrink-0"
            aria-hidden="true"
          >
            <path
              d="M12 13V20.5"
              stroke="#231D18"
              strokeWidth="2"
              strokeLinecap="round"
            />
            <path
              d="M12 13L6.2 6.8"
              stroke="#7C5333"
              strokeWidth="2"
              strokeLinecap="round"
            />
            <path
              d="M12 13L17.8 6.8"
              stroke="#2F5A3B"
              strokeWidth="2"
              strokeLinecap="round"
            />
            <circle cx="5.5" cy="5.8" r="2.2" fill="#7C5333" />
            <circle cx="18.5" cy="5.8" r="2.2" fill="#2F5A3B" />
          </svg>
          <span
            className="text-[17px] font-semibold tracking-tight text-[#231D18]"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Ubhaya-Bandha{' '}
            <span className="font-normal text-sm text-[#7A6E61]">
              (Dual Binding)
            </span>
          </span>
        </button>

        {/* Zone 2: Clean Single-Line Text Navigation */}
        <nav
          aria-label="Primary navigation"
          className="flex items-center gap-6 overflow-x-auto"
        >
          {NAV_ITEMS.map((item) => {
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectTab(item.id)}
                className={`py-5 text-[13.5px] whitespace-nowrap shrink-0 border-b-2 transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#7C5333] ${
                  active
                    ? 'border-[#231D18] text-[#231D18] font-semibold'
                    : 'border-transparent text-[#665B4F] hover:text-[#231D18] hover:border-[#D5C7B4] font-medium'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
