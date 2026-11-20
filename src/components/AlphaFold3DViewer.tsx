import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  RotateCw,
  ShieldAlert,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { bspaiClient } from '../api/bspaiClient.ts';
import type {
  AlphaFoldStructurePayload,
  MechanismType,
  PairGnnMetrics,
} from '../types.ts';
import {
  mountProtein3DScene,
  type AnimPhaseUpdate,
  type BindingAnimScenario,
  type CameraPreset,
  type HoveredResidueInfo,
  type ProteinColorMode,
  type ProteinRenderStyle,
  type ProteinSceneHandle,
  type ProteinViewTarget,
} from './three/proteinStructureScene.ts';

interface AlphaFold3DViewerProps {
  gene1: string;
  gene2: string;
  mechanismType?: MechanismType;
  pairGnn?: PairGnnMetrics;
}

export function AlphaFold3DViewer({
  gene1,
  gene2,
  mechanismType,
  pairGnn,
}: AlphaFold3DViewerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<ProteinSceneHandle | null>(null);

  const [struct1, setStruct1] = useState<AlphaFoldStructurePayload | null>(null);
  const [struct2, setStruct2] = useState<AlphaFoldStructurePayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [syncingLive, setSyncingLive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [webglFailed, setWebglFailed] = useState(false);

  const [viewTarget, setViewTarget] = useState<ProteinViewTarget>('dual');
  const [colorMode, setColorMode] = useState<ProteinColorMode>('salience');
  const [renderStyle, setRenderStyle] = useState<ProteinRenderStyle>('binding');
  const [playBindingAnim, setPlayBindingAnim] = useState(true);
  const [replayVersion, setReplayVersion] = useState(0);
  const [autoRotate, setAutoRotate] = useState(false);
  const [dockOpen, setDockOpen] = useState(true);

  const [hoveredResidue, setHoveredResidue] = useState<HoveredResidueInfo | null>(null);
  const [animPhase, setAnimPhase] = useState<AnimPhaseUpdate>({
    state: 'approaching',
    stepLabel: 'Approaching Pockets',
    detail: 'Aligning bispecific arms with exposed GNN binding pockets',
  });

  const fetchStructures = async (live = false) => {
    if (live) setSyncingLive(true);
    else setLoading(true);
    setError(null);
    try {
      const [r1, r2] = await Promise.all([
        bspaiClient.getAlphaFoldStructure(gene1, live),
        bspaiClient.getAlphaFoldStructure(gene2, live),
      ]);
      setStruct1(r1);
      setStruct2(r2);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load AlphaFold structures');
    } finally {
      setLoading(false);
      setSyncingLive(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([
      bspaiClient.getAlphaFoldStructure(gene1, false),
      bspaiClient.getAlphaFoldStructure(gene2, false),
    ])
      .then(([s1, s2]) => {
        if (cancelled) return;
        setStruct1(s1);
        setStruct2(s2);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load AlphaFold structures');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [gene1, gene2]);

  const reducedMotion =
    typeof window !== 'undefined' &&
    Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);

  const assessment = pairGnn?.bindingAssessment;

  const effectiveScenario: BindingAnimScenario =
    assessment?.verdict === 'blocked'
      ? 'blocked'
      : assessment?.verdict === 'conditional-bind'
        ? 'conditional'
        : 'bind';

  const blockedSide: 'target1' | 'target2' | 'both' =
    assessment && !assessment.target1.canBind && !assessment.target2.canBind
      ? 'both'
      : assessment && !assessment.target1.canBind
        ? 'target1'
        : assessment && !assessment.target2.canBind
          ? 'target2'
          : assessment?.target1.isMultiPass && assessment?.target2.isMultiPass
            ? 'both'
            : assessment?.target1.barrierReason
              ? 'target1'
              : 'target2';

  // Mount Three.js scene once
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;
    try {
      const handle = mountProtein3DScene(el, {
        structure1: struct1,
        structure2: struct2,
        viewTarget,
        colorMode,
        renderStyle,
        autoRotate,
        reducedMotion,
        playBindingAnim,
        bindingScenario: effectiveScenario,
        mechanismType,
        fitPercent: assessment?.fitPercent,
        blockedSide,
        replayVersion,
        onHoverResidue: setHoveredResidue,
        onAnimPhaseChange: setAnimPhase,
      });
      sceneRef.current = handle;
      return () => {
        handle.dispose();
        sceneRef.current = null;
      };
    } catch {
      setWebglFailed(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push structural / combination / animation updates seamlessly
  useEffect(() => {
    sceneRef.current?.updateParams({
      structure1: struct1,
      structure2: struct2,
      viewTarget,
      colorMode,
      renderStyle,
      autoRotate,
      reducedMotion,
      playBindingAnim,
      bindingScenario: effectiveScenario,
      mechanismType,
      fitPercent: assessment?.fitPercent,
      blockedSide,
      replayVersion,
      onHoverResidue: setHoveredResidue,
      onAnimPhaseChange: setAnimPhase,
    });
  }, [
    struct1,
    struct2,
    viewTarget,
    colorMode,
    renderStyle,
    autoRotate,
    reducedMotion,
    playBindingAnim,
    effectiveScenario,
    mechanismType,
    assessment?.fitPercent,
    blockedSide,
    replayVersion,
  ]);

  const handlePreset = (preset: CameraPreset) => {
    sceneRef.current?.setCameraPreset(preset);
  };

  const activeStruct =
    viewTarget === 'target2' ? struct2 ?? struct1 : struct1 ?? struct2;

  const verdictStyle =
    assessment?.verdict === 'blocked'
      ? {
          wrap: 'bg-rose-50/70 border-rose-200',
          title: 'text-rose-900',
          badge: 'text-rose-700',
          icon: <ShieldAlert className="w-5 h-5 text-rose-600 shrink-0" />,
          label: 'Binding Blocked',
        }
      : assessment?.verdict === 'conditional-bind'
        ? {
            wrap: 'bg-amber-50/70 border-amber-200',
            title: 'text-amber-950',
            badge: 'text-amber-800',
            icon: <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0" />,
            label: 'Conditional Fit',
          }
        : {
            wrap: 'bg-emerald-50/70 border-emerald-200',
            title: 'text-emerald-950',
            badge: 'text-emerald-700',
            icon: <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />,
            label: 'Can Bind Cleanly',
          };

  return (
    <div className="bg-[#FFFDF9] border border-[#E5D8C5] rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 bg-[#F5EDE1] border-b border-[#E5D8C5] flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
        <div>
          <h3 className="text-base font-bold text-[#2B231C]">
            3D Protein Binding &amp; GNN Structure Explorer
          </h3>
          <p className="text-xs text-[#6E6052] mt-0.5">
            Shows each target’s AlphaFold 3D fold, highlighted binding pockets, and a mechanism-aware binding simulation
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => fetchStructures(true)}
            disabled={syncingLive}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#5E5245] bg-[#FFFDF9] border border-[#DFCEB8] hover:bg-[#FAF4EA] disabled:opacity-50 rounded-xl transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${syncingLive ? 'animate-spin' : ''}`} />
            {syncingLive ? 'Querying AlphaFold DB...' : 'Sync AlphaFold DB'}
          </button>

          {activeStruct && (
            <a
              href={activeStruct.alphafoldUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#8B5E3C] bg-[#FFFDF9] hover:bg-[#EFE4D4] border border-[#DFCEB8] rounded-xl transition-colors whitespace-nowrap shrink-0"
            >
              <span>{activeStruct.entryId}</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12">
        {/* Left 7 cols: Simple Combination-Aware 3D Protein Animation */}
        <div className="xl:col-span-7 relative border-b xl:border-b-0 xl:border-r border-[#E5D8C5] bg-[#FCF7F3]">
          <div className="relative w-full h-[460px]">
            <div
              ref={mountRef}
              className="w-full h-full cursor-grab active:cursor-grabbing"
              aria-label={`Interactive 3D binding animation for ${gene1} and ${gene2}`}
            />

            {webglFailed && (
              <div className="absolute inset-0 flex items-center justify-center text-xs font-mono text-slate-600 bg-slate-50">
                WebGL 3D context unavailable · {struct1?.entryId} / {struct2?.entryId}
              </div>
            )}

            {loading && !struct1 && (
              <div className="absolute inset-0 flex items-center justify-center text-xs font-mono text-slate-600 bg-[#FCF7F3]/85">
                Loading AlphaFold 3D structures & binding animation...
              </div>
            )}

            {/* Top-Right Live Combination Animation Step or Residue Hover Readout */}
            <div className="pointer-events-none absolute top-3 right-3 bg-white/95 border border-slate-200 rounded-md px-3 py-2 text-[11px] text-right max-w-[270px]">
              {hoveredResidue ? (
                <div className="space-y-0.5 font-mono text-slate-800">
                  <div className="font-semibold text-[#2563EB]">
                    {hoveredResidue.gene} · {hoveredResidue.resName} {hoveredResidue.resSeq}
                  </div>
                  <div className="text-[10px] text-slate-600 truncate">
                    {hoveredResidue.domain}
                  </div>
                  <div className="text-[10px] text-slate-700">
                    pLDDT {hoveredResidue.plddt.toFixed(1)} · Pocket {(hoveredResidue.salience * 100).toFixed(0)}%
                  </div>
                </div>
              ) : (
                <div className="space-y-0.5">
                  <div
                    className={`font-semibold ${
                      animPhase.state === 'bound'
                        ? 'text-emerald-700'
                        : animPhase.state === 'blocked'
                          ? 'text-rose-700'
                          : animPhase.state === 'caution'
                            ? 'text-amber-700'
                            : 'text-[#2563EB]'
                    }`}
                  >
                    {animPhase.stepLabel}
                  </div>
                  <div className="text-[10px] text-slate-600 leading-snug">
                    {animPhase.detail}
                  </div>
                </div>
              )}
            </div>

            {/* Collapsible Bottom Control Dock */}
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10 w-[min(96%,680px)] flex flex-col items-center">
              <button
                type="button"
                onClick={() => setDockOpen((v) => !v)}
                className="mb-1 inline-flex items-center gap-1 px-2.5 py-0.5 bg-white border border-slate-300 rounded-t-md text-[10px] font-medium text-slate-700 hover:bg-slate-50 cursor-pointer"
              >
                {dockOpen ? (
                  <>
                    <ChevronDown className="w-3 h-3" /> Hide Controls
                  </>
                ) : (
                  <>
                    <ChevronUp className="w-3 h-3" /> Show Controls
                  </>
                )}
              </button>

              {dockOpen && (
                <div className="w-full bg-[#FFFDF9] border border-[#DFCEB8] rounded-xl px-3 py-2 shadow-sm flex flex-wrap items-center justify-between gap-2 text-[11px]">
                  {/* Playback Controls */}
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => setPlayBindingAnim((p) => !p)}
                      className="inline-flex items-center gap-1 px-2.5 py-1 bg-[#8B5E3C] hover:bg-[#734B2E] text-[#FFFDF9] rounded-lg font-medium cursor-pointer"
                    >
                      {playBindingAnim ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                      {playBindingAnim ? 'Pause' : 'Play'}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setPlayBindingAnim(true);
                        setReplayVersion((v) => v + 1);
                      }}
                      title="Replay binding animation"
                      className="p-1.5 border border-[#DFCEB8] hover:bg-[#F5EDE1] rounded-lg text-[#5E5245] cursor-pointer"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  {/* Protein View */}
                  <div className="flex items-center gap-0.5 p-0.5 bg-[#F3EADF] rounded-lg">
                    <button
                      type="button"
                      onClick={() => setViewTarget('dual')}
                      className={`px-2 py-1 rounded-md font-medium cursor-pointer ${
                        viewTarget === 'dual' ? 'bg-[#FFFDF9] text-[#2B231C] shadow-xs' : 'text-[#6E6052]'
                      }`}
                    >
                      Binding View
                    </button>
                    <button
                      type="button"
                      onClick={() => setViewTarget('target1')}
                      className={`px-2 py-1 rounded-md font-mono font-medium cursor-pointer ${
                        viewTarget === 'target1' ? 'bg-[#FFFDF9] text-[#8B5E3C] shadow-xs' : 'text-[#6E6052]'
                      }`}
                    >
                      {gene1}
                    </button>
                    <button
                      type="button"
                      onClick={() => setViewTarget('target2')}
                      className={`px-2 py-1 rounded-md font-mono font-medium cursor-pointer ${
                        viewTarget === 'target2' ? 'bg-[#FFFDF9] text-[#4F6D4A] shadow-xs' : 'text-[#6E6052]'
                      }`}
                    >
                      {gene2}
                    </button>
                  </div>

                  {/* Detail & Camera */}
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() =>
                        setRenderStyle((s) => (s === 'binding' ? 'ribbon-graph' : 'binding'))
                      }
                      className={`px-2 py-1 border rounded-lg cursor-pointer ${
                        renderStyle === 'ribbon-graph'
                          ? 'border-[#8B5E3C] bg-[#EFE4D4] text-[#8B5E3C]'
                          : 'border-[#DFCEB8] text-[#5E5245] hover:bg-[#F5EDE1]'
                      }`}
                    >
                      {renderStyle === 'ribbon-graph' ? 'GNN Graph: On' : 'GNN Graph: Off'}
                    </button>
                    <button
                      type="button"
                      onClick={() => handlePreset('front')}
                      className="px-2 py-1 border border-[#DFCEB8] hover:bg-[#F5EDE1] rounded-lg text-[#5E5245] cursor-pointer"
                    >
                      Reset
                    </button>
                    <button
                      type="button"
                      onClick={() => setAutoRotate((r) => !r)}
                      title="Rotate 3D view"
                      className={`p-1 border rounded-lg cursor-pointer ${
                        autoRotate
                          ? 'border-[#8B5E3C] bg-[#EFE4D4] text-[#8B5E3C]'
                          : 'border-[#DFCEB8] text-[#5E5245] hover:bg-[#F5EDE1]'
                      }`}
                    >
                      <RotateCw className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right 5 cols: Simple Binding Explanation */}
        <div className="xl:col-span-5 p-6 space-y-5 bg-[#FFFEFC]">
          {error && (
            <div className="p-3 bg-[#FDF4F0] border border-[#E5C1B5] rounded-lg text-xs text-[#853826]">
              {error}
            </div>
          )}

          {/* 1. Plain-English Binding Verdict Banner */}
          {assessment && (
            <div className={`p-4 border rounded-xl space-y-2.5 ${verdictStyle.wrap}`}>
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-2.5">
                  {verdictStyle.icon}
                  <div>
                    <div className={`text-xs font-semibold ${verdictStyle.badge}`}>
                      {verdictStyle.label}
                    </div>
                    <h3 className={`text-sm font-semibold ${verdictStyle.title}`}>
                      {assessment.headline}
                    </h3>
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <div className="text-xl font-mono tabular-nums font-semibold text-[#231D18]">
                    {assessment.fitPercent}%
                  </div>
                  <div className="text-[10px] text-[#6E6356]">3D Fit</div>
                </div>
              </div>

              <p className="text-xs text-[#4C4238] leading-relaxed">
                {assessment.summary}
              </p>

              <div className="pt-2 border-t border-black/10 text-xs text-[#231D18] font-medium leading-relaxed">
                {assessment.whyNote}
              </div>
            </div>
          )}

          {/* 2. 3 Simple Reasons: Can They Bind or Why Not? */}
          {assessment && (
            <div className="space-y-2.5">
              <h3 className="text-sm font-semibold text-[#231D18]">
                Why This Pair Can or Can’t Bind
              </h3>
              <div className="divide-y divide-[#EAE2D6] border border-[#E3D9CA] rounded-xl bg-[#FAF6F0]/45">
                {assessment.checks.map((chk) => {
                  const statusMeta =
                    chk.status === 'pass'
                      ? { text: 'Pass', cls: 'text-[#2F5A3B]', icon: <CheckCircle2 className="w-4 h-4 text-[#2F5A3B] shrink-0" /> }
                      : chk.status === 'caution'
                        ? { text: 'Caution', cls: 'text-[#9A5B13]', icon: <AlertTriangle className="w-4 h-4 text-[#B46918] shrink-0" /> }
                        : { text: 'Blocked', cls: 'text-[#A03025]', icon: <ShieldAlert className="w-4 h-4 text-[#B63A2E] shrink-0" /> };

                  return (
                    <div key={chk.id} className="p-3.5 space-y-1">
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 text-xs font-semibold text-[#231D18]">
                          {statusMeta.icon}
                          <span>{chk.label}</span>
                        </div>
                        <span className={`text-xs font-semibold ${statusMeta.cls}`}>
                          {statusMeta.text}
                        </span>
                      </div>
                      <p className="text-xs text-[#574D43] leading-relaxed pl-6">
                        {chk.detail}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
