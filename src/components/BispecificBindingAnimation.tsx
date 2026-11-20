import { RotateCcw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { BindingVerdict, MechanismType } from '../types.ts';
import {
  mountBispecific3DScene,
  type BispecificSceneHandle,
} from './three/bispecificScene.ts';

interface BispecificBindingAnimationProps {
  gene1: string;
  gene2: string;
  bias1?: string;
  bias2?: string;
  mechanismType?: MechanismType;
  bindingVerdict?: BindingVerdict;
  score?: number;
  loading?: boolean;
  evalTrigger: number;
}

const MECH_PALETTE: Record<
  string,
  {
    arm1: string;
    arm2: string;
    arm1Hex: number;
    arm2Hex: number;
    fc: string;
    fcHex: number;
    label: string;
    modeText: string;
  }
> = {
  'bridge-two-cells': {
    arm1: '#2563EB',
    arm2: '#059669',
    arm1Hex: 0x2563eb,
    arm2Hex: 0x059669,
    fc: '#4F46E5',
    fcHex: 0x4f46e5,
    label: 'Trans-Cellular Bridge',
    modeText: 'Cross-links tumor antigen and effector immune receptor across two cells',
  },
  'TAA+TAA': {
    arm1: '#7C3AED',
    arm2: '#D97706',
    arm1Hex: 0x7c3aed,
    arm2Hex: 0xd97706,
    fc: '#4338CA',
    fcHex: 0x4338ca,
    label: 'Cis Dual-TAA Co-Targeting',
    modeText: 'Co-engages two overexpressed receptors on the same tumor membrane',
  },
  'tumor-immunotherapy': {
    arm1: '#2563EB',
    arm2: '#E11D48',
    arm1Hex: 0x2563eb,
    arm2Hex: 0xe11d48,
    fc: '#0D9488',
    fcHex: 0x0d9488,
    label: 'Dual Immune Modulation',
    modeText: 'Simultaneously engages tumor-immune synapse checkpoints & TME targets',
  },
};

export function BispecificBindingAnimation({
  gene1,
  gene2,
  bias1 = 'target 1',
  bias2 = 'target 2',
  mechanismType = 'tumor-immunotherapy',
  bindingVerdict = 'strong-bind',
  score,
  loading = false,
  evalTrigger,
}: BispecificBindingAnimationProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const sceneHandleRef = useRef<BispecificSceneHandle | null>(null);
  const [replayCount, setReplayCount] = useState(0);
  const [webglFailed, setWebglFailed] = useState(false);

  const palette = MECH_PALETTE[mechanismType] ?? MECH_PALETTE['tumor-immunotherapy'];
  const dockVersion = evalTrigger * 100 + replayCount;

  const latestParamsRef = useRef({
    mechanismType,
    bindingVerdict,
    arm1Hex: palette.arm1Hex,
    arm2Hex: palette.arm2Hex,
    fcHex: palette.fcHex,
    dockVersion,
    reducedMotion: false,
  });

  latestParamsRef.current = {
    mechanismType,
    bindingVerdict,
    arm1Hex: palette.arm1Hex,
    arm2Hex: palette.arm2Hex,
    fcHex: palette.fcHex,
    dockVersion,
    reducedMotion:
      typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
  };

  // Mount WebGL canvas once and tear down cleanly on unmount
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    try {
      const handle = mountBispecific3DScene(el, latestParamsRef.current);
      sceneHandleRef.current = handle;
      return () => {
        handle.dispose();
        sceneHandleRef.current = null;
      };
    } catch {
      setWebglFailed(true);
    }
  }, []);

  // Push updated pair/mechanism/verdict/dock triggers seamlessly without resetting camera
  useEffect(() => {
    sceneHandleRef.current?.updateParams({
      mechanismType,
      bindingVerdict,
      arm1Hex: palette.arm1Hex,
      arm2Hex: palette.arm2Hex,
      fcHex: palette.fcHex,
      dockVersion,
      reducedMotion:
        typeof window !== 'undefined' &&
        window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
    });
  }, [
    gene1,
    gene2,
    mechanismType,
    bindingVerdict,
    palette.arm1Hex,
    palette.arm2Hex,
    palette.fcHex,
    dockVersion,
  ]);

  const isBlocked = bindingVerdict === 'blocked';
  const isConditional = bindingVerdict === 'conditional-bind';

  const headerTitle = isBlocked
    ? `3D Dual-Target Binding · Blocked (${palette.label})`
    : isConditional
      ? `3D Dual-Target Binding · Conditional Fit (${palette.label})`
      : `3D Dual-Target Binding · ${palette.label}`;

  const statusText = loading
    ? 'Evaluating pair...'
    : isBlocked
      ? 'Blocked · Antibody recoils'
      : isConditional
        ? `Conditional fit · ${mechanismType}`
        : `${mechanismType}${score !== undefined ? ` · ${score.toFixed(2)}` : ''}`;

  const footerText = isBlocked
    ? 'Steric crowding near the cell membrane prevents simultaneous dual-arm binding — the antibody approaches and recoils.'
    : isConditional
      ? `${palette.modeText} (tight membrane-loop clearance — linker spacing matters).`
      : palette.modeText;

  return (
    <div className="bg-[#FFFDF9] border border-[#E5D8C5] rounded-2xl overflow-hidden flex flex-col justify-between">
      {/* Top compact header */}
      <div className="px-5 py-4 bg-[#F5EDE1] border-b border-[#E5D8C5] flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="text-sm font-bold text-[#2B231C] truncate">
            {headerTitle}
          </div>
          <div className="text-xs font-mono text-[#786A5B] truncate">
            {gene1} ({bias1}) + {gene2} ({bias2})
          </div>
        </div>

        <button
          type="button"
          onClick={() => setReplayCount((c) => c + 1)}
          title="Replay 3D binding animation"
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#2B231C] bg-[#FFFDF9] hover:bg-[#EFE4D4] border border-[#DFCEB8] rounded-xl transition-colors shrink-0 cursor-pointer"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Replay
        </button>
      </div>

      {/* 3D WebGL viewport */}
      <div className="p-4">
        <div className="relative w-full h-[220px] rounded-xl overflow-hidden border border-[#E5D8C5] bg-[#FAF5EC]">
          {/* Isolated WebGL canvas container */}
          <div
            ref={mountRef}
            className="w-full h-full cursor-grab active:cursor-grabbing"
            aria-label={`3D bispecific antibody binding ${gene1} and ${gene2}`}
          />

          {webglFailed && (
            <div className="absolute inset-0 flex items-center justify-center text-xs font-mono text-[#6E6052] bg-[#FAF5EC]">
              3D WebGL context unavailable · {gene1} + {gene2} ({mechanismType})
            </div>
          )}

          {/* Semantic DOM labels anchored cleanly at edges */}
          <div className="pointer-events-none absolute top-2.5 inset-x-3.5 flex items-center justify-between text-[11px] font-mono font-semibold">
            <span style={{ color: palette.arm1 }}>
              {gene1} · {bias1}
            </span>
            <span
              className={
                isBlocked
                  ? 'text-[#B91C1C]'
                  : isConditional
                    ? 'text-[#B45309]'
                    : 'text-[#5E5245]'
              }
            >
              {statusText}
            </span>
            <span style={{ color: palette.arm2 }}>
              {gene2} · {bias2}
            </span>
          </div>

          <div className="pointer-events-none absolute bottom-2 right-3 text-[10px] font-mono text-[#786A5B]">
            Drag to orbit 3D
          </div>
        </div>
      </div>

      {/* Bottom concise mechanism summary */}
      <div className="px-5 py-3 bg-[#FAF6F0] border-t border-[#EDE2D3] text-xs text-[#5E5245] leading-snug">
        {footerText}
      </div>
    </div>
  );
}
