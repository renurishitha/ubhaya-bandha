import { Dna } from 'lucide-react';
import { useState } from 'react';

interface ResilientImageProps {
  src: string;
  alt: string;
  className?: string;
  fallbackTitle: string;
  accentColor?: string;
}

export function ResilientImage({
  src,
  alt,
  className = '',
  fallbackTitle,
  accentColor = '#2563EB',
}: ResilientImageProps) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div
        className={`flex flex-col items-center justify-center p-6 text-center bg-gradient-to-br from-blue-50 via-teal-50 to-violet-50 ${className}`}
        role="img"
        aria-label={alt}
      >
        <div
          className="w-12 h-12 rounded-lg flex items-center justify-center mb-2 bg-white border border-slate-200"
          style={{ color: accentColor }}
        >
          <Dna className="w-6 h-6" />
        </div>
        <span className="text-xs font-medium text-slate-700">{fallbackTitle}</span>
      </div>
    );
  }

  return (
    <img
      src={src}
      alt={alt}
      referrerPolicy="no-referrer"
      onError={() => setFailed(true)}
      className={className}
    />
  );
}
