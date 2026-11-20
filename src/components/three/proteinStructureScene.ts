import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import type { AlphaFoldStructurePayload, MechanismType } from '../../types.ts';
import { detectSoftwareWebGL } from './bispecificScene.ts';

export type ProteinColorMode = 'plddt' | 'ss' | 'salience';
export type ProteinRenderStyle = 'binding' | 'ribbon-graph';
export type ProteinViewTarget = 'target1' | 'target2' | 'dual';
export type CameraPreset = 'front' | 'pocket';
export type BindingAnimScenario = 'bind' | 'conditional' | 'blocked';

export interface AnimPhaseUpdate {
  state: 'approaching' | 'bound' | 'caution' | 'blocked' | 'single';
  stepLabel: string;
  detail: string;
}

export interface HoveredResidueInfo {
  gene: string;
  resSeq: number;
  resName: string;
  domain: string;
  plddt: number;
  ss: string;
  salience: number;
}

export interface ProteinSceneParams {
  structure1: AlphaFoldStructurePayload | null;
  structure2: AlphaFoldStructurePayload | null;
  viewTarget: ProteinViewTarget;
  colorMode: ProteinColorMode;
  renderStyle: ProteinRenderStyle;
  autoRotate: boolean;
  reducedMotion: boolean;
  playBindingAnim: boolean;
  bindingScenario: BindingAnimScenario;
  mechanismType?: MechanismType;
  fitPercent?: number;
  blockedSide?: 'target1' | 'target2' | 'both';
  replayVersion?: number;
  onHoverResidue?: (info: HoveredResidueInfo | null) => void;
  onAnimPhaseChange?: (phase: AnimPhaseUpdate) => void;
}

export interface ProteinSceneHandle {
  updateParams: (next: ProteinSceneParams) => void;
  setCameraPreset: (preset: CameraPreset) => void;
  dispose: () => void;
}

// Reusable scratch objects (zero GC in render loop)
const _color = new THREE.Color();
const _cLerped = new THREE.Color();
const _cLow = new THREE.Color(0x64748b);
const _cMid = new THREE.Color(0x0ea5e9);
const _cHigh = new THREE.Color(0xf59e0b);
const _cBound = new THREE.Color(0x10b981);
const _cCaution = new THREE.Color(0xf59e0b);
const _cBlocked = new THREE.Color(0xe11d48);
const _cNeutralRing = new THREE.Color(0x2563eb);

const _dummy = new THREE.Object3D();
const _desiredCamPos = new THREE.Vector3(0, 0.55, 6.5);
const _desiredTarget = new THREE.Vector3(0, 0.15, 0);
const _mouse = new THREE.Vector2();
const _leftTipWorld = new THREE.Vector3();
const _rightTipWorld = new THREE.Vector3();
const _p1World = new THREE.Vector3();
const _p2World = new THREE.Vector3();
const _dirVec = new THREE.Vector3();
const _midVec = new THREE.Vector3();
const _upAxis = new THREE.Vector3(0, 1, 0);
const _quat = new THREE.Quaternion();

function easeInOutCubic(x: number): number {
  const t = Math.max(0, Math.min(1, x));
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function colorForResidue(
  payload: AlphaFoldStructurePayload,
  idx: number,
  mode: ProteinColorMode,
  out: THREE.Color
): void {
  const res = payload.residues[idx];
  if (mode === 'plddt') {
    if (res.plddt >= 90) out.setHex(0x0053d6);
    else if (res.plddt >= 70) out.setHex(0x38bdf8);
    else if (res.plddt >= 50) out.setHex(0xfacc15);
    else out.setHex(0xf97316);
    return;
  }
  if (mode === 'ss') {
    if (res.ss === 'H') out.setHex(0xe11d48);
    else if (res.ss === 'E') out.setHex(0x2563eb);
    else out.setHex(0x0d9488);
    return;
  }
  const sal = payload.gnn.residueSalience[idx] ?? 0;
  if (sal < 0.5) {
    out.copy(_cLow).lerp(_cMid, sal * 2);
  } else {
    out.copy(_cMid).lerp(_cHigh, (sal - 0.5) * 2);
  }
}

interface BuiltProteinResult {
  group: THREE.Group;
  caMesh: THREE.InstancedMesh;
  pocketLocalCenter: THREE.Vector3;
  baseX: number;
  isSecreted: boolean;
  pocketRing: THREE.Mesh;
  pocketRingMat: THREE.MeshStandardMaterial;
  barrierGroup: THREE.Group;
  barrierMat: THREE.MeshBasicMaterial;
  disposables: Array<{ dispose: () => void }>;
}

function buildProteinAssembly(
  payload: AlphaFoldStructurePayload,
  colorMode: ProteinColorMode,
  renderStyle: ProteinRenderStyle,
  xOffset: number,
  targetScaleRadius: number
): BuiltProteinResult {
  const disposables: Array<{ dispose: () => void }> = [];
  const group = new THREE.Group();
  group.position.set(xOffset, -0.25, 0);
  const isSecreted = payload.foldClass.toLowerCase().includes('secreted');

  const residues = payload.residues;
  const n = residues.length;
  if (n < 2) {
    const emptyGeo = new THREE.SphereGeometry(0.01);
    const emptyMat = new THREE.MeshBasicMaterial();
    const ringGeo = new THREE.TorusGeometry(0.2, 0.02, 8, 24);
    const ringMat = new THREE.MeshStandardMaterial();
    const bGeo = new THREE.CircleGeometry(0.3, 24);
    const bMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0 });
    disposables.push(emptyGeo, emptyMat, ringGeo, ringMat, bGeo, bMat);
    const caMesh = new THREE.InstancedMesh(emptyGeo, emptyMat, 1);
    return {
      group,
      caMesh,
      pocketLocalCenter: new THREE.Vector3(0, 0, 0),
      baseX: xOffset,
      isSecreted,
      pocketRing: new THREE.Mesh(ringGeo, ringMat),
      pocketRingMat: ringMat,
      barrierGroup: new THREE.Group(),
      barrierMat: bMat,
      disposables,
    };
  }

  let maxDistSq = 1e-6;
  for (const r of residues) {
    const d2 = r.x * r.x + r.y * r.y + r.z * r.z;
    if (d2 > maxDistSq) maxDistSq = d2;
  }
  const scale = targetScaleRadius / Math.sqrt(maxDistSq);

  const points: THREE.Vector3[] = residues.map(
    (r) => new THREE.Vector3(r.x * scale, r.y * scale, r.z * scale)
  );

  // 1. Smooth Backbone Ribbon
  const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.45);
  const tubularSegments = Math.min(600, Math.max(120, n * 3));
  const tubeRadius = renderStyle === 'binding' ? 0.058 : 0.044;
  const tubeGeo = new THREE.TubeGeometry(curve, tubularSegments, tubeRadius, 10, false);

  const posAttr = tubeGeo.attributes.position;
  const radialSegments = 10;
  const ringVerts = radialSegments + 1;
  const colors = new Float32Array(posAttr.count * 3);

  for (let seg = 0; seg <= tubularSegments; seg++) {
    const t = seg / tubularSegments;
    const fIdx = t * (n - 1);
    const i0 = Math.min(n - 1, Math.floor(fIdx));
    const i1 = Math.min(n - 1, i0 + 1);
    const frac = fIdx - i0;

    colorForResidue(payload, i0, colorMode, _color);
    colorForResidue(payload, i1, colorMode, _cLerped);
    _color.lerp(_cLerped, frac);

    for (let rv = 0; rv < ringVerts; rv++) {
      const vIdx = seg * ringVerts + rv;
      if (vIdx < posAttr.count) {
        colors[vIdx * 3] = _color.r;
        colors[vIdx * 3 + 1] = _color.g;
        colors[vIdx * 3 + 2] = _color.b;
      }
    }
  }
  tubeGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const tubeMat = new THREE.MeshPhysicalMaterial({
    vertexColors: true,
    roughness: 0.28,
    metalness: 0.05,
    clearcoat: 0.72,
    clearcoatRoughness: 0.16,
  });
  disposables.push(tubeGeo, tubeMat);
  group.add(new THREE.Mesh(tubeGeo, tubeMat));

  // 2. Hoverable C-alpha Residue Nodes (subtle in binding mode, larger in graph mode)
  const atomRadius = renderStyle === 'ribbon-graph' ? 0.038 : 0.024;
  const caGeo = new THREE.SphereGeometry(atomRadius, 10, 8);
  const caMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.3,
    metalness: 0.08,
  });
  disposables.push(caGeo, caMat);

  const caMesh = new THREE.InstancedMesh(caGeo, caMat, n);
  caMesh.layers.set(1);
  caMesh.frustumCulled = false;
  caMesh.userData = { payload };

  for (let i = 0; i < n; i++) {
    const sal = payload.gnn.residueSalience[i] ?? 0;
    const s =
      renderStyle === 'binding'
        ? sal >= 0.8 ? 1.25 : 0.55
        : sal >= 0.78 ? 1.3 : 0.85;
    _dummy.position.copy(points[i]);
    _dummy.rotation.set(0, 0, 0);
    _dummy.scale.setScalar(s);
    _dummy.updateMatrix();
    caMesh.setMatrixAt(i, _dummy.matrix);

    colorForResidue(payload, i, colorMode, _color);
    caMesh.setColorAt(i, _color);
  }
  caMesh.instanceMatrix.needsUpdate = true;
  if (caMesh.instanceColor) caMesh.instanceColor.needsUpdate = true;
  caMesh.computeBoundingSphere();
  group.add(caMesh);

  // 3. Optional GNN 3D Spatial Contact Graph Edges
  if (renderStyle === 'ribbon-graph') {
    const spatialEdges = payload.gnn.contactEdges.filter((e) => e.isSpatialContact);
    if (spatialEdges.length > 0) {
      const lineVerts = new Float32Array(spatialEdges.length * 6);
      for (let i = 0; i < spatialEdges.length; i++) {
        const e = spatialEdges[i];
        const pA = points[e.source];
        const pB = points[e.target];
        lineVerts[i * 6] = pA.x;
        lineVerts[i * 6 + 1] = pA.y;
        lineVerts[i * 6 + 2] = pA.z;
        lineVerts[i * 6 + 3] = pB.x;
        lineVerts[i * 6 + 4] = pB.y;
        lineVerts[i * 6 + 5] = pB.z;
      }
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute('position', new THREE.BufferAttribute(lineVerts, 3));
      const lineMat = new THREE.LineBasicMaterial({
        color: colorMode === 'salience' ? 0xd97706 : 0x0f766e,
        transparent: true,
        opacity: 0.28,
      });
      disposables.push(lineGeo, lineMat);
      group.add(new THREE.LineSegments(lineGeo, lineMat));
    }
  }

  // 4. Top-GNN Binding Pocket Halo & Docking Reticle Ring
  const sortedSalIndices = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => (payload.gnn.residueSalience[b] ?? 0) - (payload.gnn.residueSalience[a] ?? 0))
    .slice(0, Math.min(14, n));

  const haloGeo = new THREE.SphereGeometry(0.08, 12, 10);
  const haloMat = new THREE.MeshStandardMaterial({
    color: 0xf59e0b,
    emissive: 0xd97706,
    emissiveIntensity: 0.35,
    roughness: 0.25,
    transparent: true,
    opacity: 0.32,
    depthWrite: false,
  });
  disposables.push(haloGeo, haloMat);

  const haloMesh = new THREE.InstancedMesh(haloGeo, haloMat, sortedSalIndices.length);
  haloMesh.frustumCulled = false;

  const pocketLocal = new THREE.Vector3();
  sortedSalIndices.forEach((rIdx, hi) => {
    const pt = points[rIdx];
    pocketLocal.add(pt);
    _dummy.position.copy(pt);
    _dummy.rotation.set(0, 0, 0);
    _dummy.scale.setScalar(1);
    _dummy.updateMatrix();
    haloMesh.setMatrixAt(hi, _dummy.matrix);
  });
  haloMesh.instanceMatrix.needsUpdate = true;
  haloMesh.computeBoundingSphere();
  group.add(haloMesh);

  if (sortedSalIndices.length > 0) {
    pocketLocal.divideScalar(sortedSalIndices.length);
  }

  // Pocket Docking Ring
  const ringGeo = new THREE.TorusGeometry(0.32, 0.022, 12, 36);
  const pocketRingMat = new THREE.MeshStandardMaterial({
    color: 0xf59e0b,
    emissive: 0xf59e0b,
    emissiveIntensity: 0.45,
    roughness: 0.2,
    metalness: 0.2,
    transparent: true,
    opacity: 0.9,
  });
  disposables.push(ringGeo, pocketRingMat);
  const pocketRing = new THREE.Mesh(ringGeo, pocketRingMat);
  pocketRing.position.copy(pocketLocal);
  pocketRing.rotation.x = Math.PI / 2.3;
  group.add(pocketRing);

  // Steric / Access Barrier Shield
  const barrierGroup = new THREE.Group();
  barrierGroup.position.copy(pocketLocal).add(new THREE.Vector3(0, 0.24, 0));
  barrierGroup.visible = false;

  const shieldGeo = new THREE.CircleGeometry(0.42, 32);
  const barrierMat = new THREE.MeshBasicMaterial({
    color: 0xe11d48,
    transparent: true,
    opacity: 0.25,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const shieldRimGeo = new THREE.TorusGeometry(0.42, 0.024, 10, 36);
  const shieldBarGeo = new THREE.BoxGeometry(0.76, 0.045, 0.025);
  const barrierSolidMat = new THREE.MeshStandardMaterial({
    color: 0xe11d48,
    emissive: 0xbe123c,
    emissiveIntensity: 0.4,
    roughness: 0.25,
  });
  disposables.push(shieldGeo, barrierMat, shieldRimGeo, shieldBarGeo, barrierSolidMat);

  const shieldDisc = new THREE.Mesh(shieldGeo, barrierMat);
  shieldDisc.rotation.x = Math.PI / 2.3;
  const shieldRim = new THREE.Mesh(shieldRimGeo, barrierSolidMat);
  shieldRim.rotation.x = Math.PI / 2.3;
  const shieldSlash = new THREE.Mesh(shieldBarGeo, barrierSolidMat);
  shieldSlash.rotation.set(Math.PI / 2.3, 0, Math.PI / 4);
  barrierGroup.add(shieldDisc, shieldRim, shieldSlash);
  group.add(barrierGroup);

  // 5. Transmembrane Bilayer Reference Ring (for membrane-anchored proteins)
  if (!isSecreted) {
    const tmYs: number[] = [];
    for (let i = 0; i < n; i++) {
      const dom = residues[i].domain.toLowerCase();
      if (dom.includes('transmembrane') || dom.includes('tm')) {
        tmYs.push(points[i].y);
      }
    }
    if (tmYs.length >= 3) {
      const meanTmY = tmYs.reduce((a, b) => a + b, 0) / tmYs.length;
      const discGeo = new THREE.CylinderGeometry(targetScaleRadius * 0.72, targetScaleRadius * 0.72, 0.026, 36);
      const discMat = new THREE.MeshStandardMaterial({
        color: 0xd97706,
        transparent: true,
        opacity: 0.1,
        roughness: 0.65,
        depthWrite: false,
      });
      const tmRingGeo = new THREE.TorusGeometry(targetScaleRadius * 0.72, 0.011, 8, 48);
      const tmRingMat = new THREE.MeshBasicMaterial({
        color: 0xb45309,
        transparent: true,
        opacity: 0.28,
      });
      disposables.push(discGeo, discMat, tmRingGeo, tmRingMat);
      const disc = new THREE.Mesh(discGeo, discMat);
      disc.position.y = meanTmY;
      const tmRing = new THREE.Mesh(tmRingGeo, tmRingMat);
      tmRing.rotation.x = Math.PI / 2;
      tmRing.position.y = meanTmY;
      group.add(disc, tmRing);
    }
  }

  return {
    group,
    caMesh,
    pocketLocalCenter: pocketLocal,
    baseX: xOffset,
    isSecreted,
    pocketRing,
    pocketRingMat,
    barrierGroup,
    barrierMat,
    disposables,
  };
}

interface AntibodyRig {
  group: THREE.Group;
  leftPivot: THREE.Group;
  rightPivot: THREE.Group;
  leftTipAnchor: THREE.Object3D;
  rightTipAnchor: THREE.Object3D;
  leftBeam: THREE.Mesh;
  leftBeamMat: THREE.MeshBasicMaterial;
  rightBeam: THREE.Mesh;
  rightBeamMat: THREE.MeshBasicMaterial;
  disposables: Array<{ dispose: () => void }>;
}

function buildBispecificAntibodyRig(): AntibodyRig {
  const disposables: Array<{ dispose: () => void }> = [];
  const group = new THREE.Group();

  const fcMat = new THREE.MeshPhysicalMaterial({
    color: 0x4f46e5,
    roughness: 0.25,
    metalness: 0.12,
    clearcoat: 0.8,
  });
  const collarMat = new THREE.MeshStandardMaterial({
    color: 0xfde68a,
    roughness: 0.22,
    metalness: 0.78,
  });
  const arm1Mat = new THREE.MeshPhysicalMaterial({
    color: 0x2563eb,
    roughness: 0.22,
    metalness: 0.1,
    clearcoat: 0.85,
  });
  const arm2Mat = new THREE.MeshPhysicalMaterial({
    color: 0x0d9488,
    roughness: 0.22,
    metalness: 0.1,
    clearcoat: 0.85,
  });
  const tipMat = new THREE.MeshStandardMaterial({
    color: 0xf59e0b,
    emissive: 0xd97706,
    emissiveIntensity: 0.3,
    roughness: 0.25,
  });

  const fcGeo = new THREE.CapsuleGeometry(0.11, 0.52, 8, 16);
  const hingeGeo = new THREE.SphereGeometry(0.14, 16, 14);
  const collarGeo = new THREE.TorusGeometry(0.135, 0.02, 8, 24);
  const fabGeo = new THREE.CapsuleGeometry(0.085, 0.88, 8, 16);
  const tipGeo = new THREE.SphereGeometry(0.105, 14, 12);

  const beamGeo = new THREE.CylinderGeometry(0.022, 0.022, 1.0, 10);
  const leftBeamMat = new THREE.MeshBasicMaterial({
    color: 0x10b981,
    transparent: true,
    opacity: 0,
  });
  const rightBeamMat = new THREE.MeshBasicMaterial({
    color: 0x10b981,
    transparent: true,
    opacity: 0,
  });

  disposables.push(
    fcMat,
    collarMat,
    arm1Mat,
    arm2Mat,
    tipMat,
    fcGeo,
    hingeGeo,
    collarGeo,
    fabGeo,
    tipGeo,
    beamGeo,
    leftBeamMat,
    rightBeamMat
  );

  const fcMesh = new THREE.Mesh(fcGeo, fcMat);
  fcMesh.position.y = 0.42;
  const hingeMesh = new THREE.Mesh(hingeGeo, collarMat);
  const collarMesh = new THREE.Mesh(collarGeo, collarMat);
  collarMesh.rotation.x = Math.PI / 2;
  collarMesh.position.y = 0.14;
  group.add(fcMesh, hingeMesh, collarMesh);

  const leftPivot = new THREE.Group();
  leftPivot.rotation.z = -0.78;
  const leftArmMesh = new THREE.Mesh(fabGeo, arm1Mat);
  leftArmMesh.position.y = -0.56;
  const leftTip = new THREE.Mesh(tipGeo, tipMat);
  leftTip.position.y = -1.06;
  const leftTipAnchor = new THREE.Object3D();
  leftTipAnchor.position.set(0, -1.12, 0);
  leftPivot.add(leftArmMesh, leftTip, leftTipAnchor);
  group.add(leftPivot);

  const rightPivot = new THREE.Group();
  rightPivot.rotation.z = 0.78;
  const rightArmMesh = new THREE.Mesh(fabGeo, arm2Mat);
  rightArmMesh.position.y = -0.56;
  const rightTip = new THREE.Mesh(tipGeo, tipMat);
  rightTip.position.y = -1.06;
  const rightTipAnchor = new THREE.Object3D();
  rightTipAnchor.position.set(0, -1.12, 0);
  rightPivot.add(rightArmMesh, rightTip, rightTipAnchor);
  group.add(rightPivot);

  const leftBeam = new THREE.Mesh(beamGeo, leftBeamMat);
  const rightBeam = new THREE.Mesh(beamGeo, rightBeamMat);

  return {
    group,
    leftPivot,
    rightPivot,
    leftTipAnchor,
    rightTipAnchor,
    leftBeam,
    leftBeamMat,
    rightBeam,
    rightBeamMat,
    disposables,
  };
}

function positionBeamBetween(
  beam: THREE.Mesh,
  a: THREE.Vector3,
  b: THREE.Vector3
): void {
  _dirVec.subVectors(b, a);
  const len = _dirVec.length();
  if (len < 1e-4) {
    beam.visible = false;
    return;
  }
  _midVec.addVectors(a, b).multiplyScalar(0.5);
  beam.position.copy(_midVec);
  _dirVec.normalize();
  _quat.setFromUnitVectors(_upAxis, _dirVec);
  beam.quaternion.copy(_quat);
  beam.scale.set(1, len, 1);
}

export function mountProtein3DScene(
  container: HTMLElement,
  initialParams: ProteinSceneParams
): ProteinSceneHandle {
  const isSoftwareGl = detectSoftwareWebGL();
  let params: ProteinSceneParams = { ...initialParams };
  let disposed = false;
  let frameId = 0;

  // Warm peach-cream canvas matching app theme
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xfcf7f3);
  scene.fog = new THREE.FogExp2(0xfcf7f3, 0.018);

  const width = Math.max(320, container.clientWidth);
  const height = Math.max(320, container.clientHeight);

  const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 80);
  camera.position.copy(_desiredCamPos);

  const renderer = new THREE.WebGLRenderer({
    antialias: !isSoftwareGl,
    powerPreference: 'high-performance',
  });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;
  renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  renderer.domElement.style.display = 'block';
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  container.appendChild(renderer.domElement);

  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  const envScene = new RoomEnvironment();
  const envTex = pmremGenerator.fromScene(envScene, 0.04).texture;
  scene.environment = envTex;
  envScene.dispose();
  pmremGenerator.dispose();

  const hemiLight = new THREE.HemisphereLight(0xfff7ed, 0xfed7aa, 0.92);
  scene.add(hemiLight);

  const keyLight = new THREE.DirectionalLight(0xfffbeb, 1.45);
  keyLight.position.set(5, 7, 6);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0x93c5fd, 0.8);
  rimLight.position.set(-5, 3, -5);
  scene.add(rimLight);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = !isSoftwareGl;
  controls.dampingFactor = 0.07;
  controls.minDistance = 2.2;
  controls.maxDistance = 12.0;
  controls.enablePan = true;
  controls.target.copy(_desiredTarget);

  let cameraTransitionActive = false;
  controls.addEventListener('start', () => {
    cameraTransitionActive = false;
  });

  const rootGroup = new THREE.Group();
  scene.add(rootGroup);

  const antibodyRig = buildBispecificAntibodyRig();
  rootGroup.add(antibodyRig.group, antibodyRig.leftBeam, antibodyRig.rightBeam);

  let activeDisposables: Array<{ dispose: () => void }> = [];
  let raycastMeshes: THREE.InstancedMesh[] = [];
  let pocketFocusPoint = new THREE.Vector3(0, 0.35, 0);
  let builtLeft: BuiltProteinResult | null = null;
  let builtRight: BuiltProteinResult | null = null;
  let sharedCisPlatform: THREE.Mesh | null = null;

  let animElapsed = 0;
  let lastPhaseKey = '';

  function emitPhase(phase: AnimPhaseUpdate) {
    if (!params.onAnimPhaseChange) return;
    const key = `${phase.state}::${phase.stepLabel}`;
    if (key !== lastPhaseKey) {
      lastPhaseKey = key;
      params.onAnimPhaseChange(phase);
    }
  }

  function rebuildProteins() {
    for (let i = rootGroup.children.length - 1; i >= 0; i--) {
      const ch = rootGroup.children[i];
      if (
        ch !== antibodyRig.group &&
        ch !== antibodyRig.leftBeam &&
        ch !== antibodyRig.rightBeam
      ) {
        rootGroup.remove(ch);
      }
    }
    for (const d of activeDisposables) {
      d.dispose();
    }
    activeDisposables = [];
    raycastMeshes = [];
    builtLeft = null;
    builtRight = null;
    sharedCisPlatform = null;

    const s1 = params.structure1;
    const s2 = params.structure2;

    if (params.viewTarget === 'dual' && s1 && s2) {
      const spreadX = params.mechanismType === 'bridge-two-cells' ? 1.72 : 1.50;
      builtLeft = buildProteinAssembly(s1, params.colorMode, params.renderStyle, -spreadX, 1.24);
      builtRight = buildProteinAssembly(s2, params.colorMode, params.renderStyle, spreadX, 1.24);
      rootGroup.add(builtLeft.group, builtRight.group);
      activeDisposables.push(...builtLeft.disposables, ...builtRight.disposables);
      raycastMeshes.push(builtLeft.caMesh, builtRight.caMesh);

      // Visual mechanism anchor: shared tumor base for TAA+TAA cis dual targeting
      if (params.mechanismType === 'TAA+TAA') {
        const baseGeo = new THREE.RingGeometry(1.9, 2.35, 48);
        const baseMat = new THREE.MeshBasicMaterial({
          color: 0x7c3aed,
          transparent: true,
          opacity: 0.12,
          side: THREE.DoubleSide,
          depthWrite: false,
        });
        disposablesPush(baseGeo, baseMat);
        sharedCisPlatform = new THREE.Mesh(baseGeo, baseMat);
        sharedCisPlatform.rotation.x = Math.PI / 2;
        sharedCisPlatform.position.y = -1.35;
        rootGroup.add(sharedCisPlatform);
      }

      antibodyRig.group.visible = true;
      pocketFocusPoint
        .copy(builtLeft.pocketLocalCenter)
        .add(builtLeft.group.position)
        .add(builtRight.pocketLocalCenter)
        .add(builtRight.group.position)
        .multiplyScalar(0.5);
    } else {
      antibodyRig.group.visible = false;
      antibodyRig.leftBeam.visible = false;
      antibodyRig.rightBeam.visible = false;
      const target = params.viewTarget === 'target2' ? s2 ?? s1 : s1 ?? s2;
      if (target) {
        const built = buildProteinAssembly(target, params.colorMode, params.renderStyle, 0, 1.55);
        rootGroup.add(built.group);
        activeDisposables.push(...built.disposables);
        raycastMeshes.push(built.caMesh);
        pocketFocusPoint.copy(built.pocketLocalCenter).add(built.group.position);
        builtLeft = built;
      }
      emitPhase({
        state: 'single',
        stepLabel: `Inspecting ${target?.gene ?? 'Target'}`,
        detail: 'Hover over any residue sphere to inspect AlphaFold pLDDT confidence',
      });
    }

    function disposablesPush(...items: Array<{ dispose: () => void }>) {
      activeDisposables.push(...items);
    }
  }

  rebuildProteins();

  // Throttled residue hover raycaster on layer 1
  const raycaster = new THREE.Raycaster();
  raycaster.layers.set(1);
  let lastRayMs = 0;

  const onPointerMove = (e: PointerEvent) => {
    if (!params.onHoverResidue || raycastMeshes.length === 0) return;
    const now = performance.now();
    if (now - lastRayMs < 55) return;
    lastRayMs = now;

    const rect = renderer.domElement.getBoundingClientRect();
    _mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    _mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(_mouse, camera);
    const hits = raycaster.intersectObjects(raycastMeshes, false);
    if (hits.length > 0 && hits[0].instanceId !== undefined) {
      const mesh = hits[0].object as THREE.InstancedMesh;
      const payload = mesh.userData.payload as AlphaFoldStructurePayload | undefined;
      const idx = hits[0].instanceId;
      if (payload && payload.residues[idx]) {
        const r = payload.residues[idx];
        params.onHoverResidue({
          gene: payload.gene,
          resSeq: r.resSeq,
          resName: r.resName,
          domain: r.domain,
          plddt: r.plddt,
          ss: r.ss,
          salience: payload.gnn.residueSalience[idx] ?? 0,
        });
        return;
      }
    }
    params.onHoverResidue(null);
  };

  const onPointerLeave = () => {
    params.onHoverResidue?.(null);
  };

  renderer.domElement.addEventListener('pointermove', onPointerMove);
  renderer.domElement.addEventListener('pointerleave', onPointerLeave);

  const resizeObserver = new ResizeObserver(() => {
    if (disposed) return;
    const w = Math.max(320, container.clientWidth);
    const h = Math.max(320, container.clientHeight);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
  resizeObserver.observe(container);

  const onContextLost = (e: Event) => {
    e.preventDefault();
  };
  renderer.domElement.addEventListener('webglcontextlost', onContextLost);

  const clock = new THREE.Clock();

  function afterDraw(cb: () => void) {
    const gl = renderer.getContext() as WebGL2RenderingContext;
    if (!gl || typeof gl.fenceSync !== 'function') {
      frameId = window.setTimeout(cb, 80);
      return;
    }
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    const poll = () => {
      if (disposed) {
        if (sync) gl.deleteSync(sync);
        return;
      }
      const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
      if (state === gl.UNSIGNALED) {
        frameId = window.setTimeout(poll, 5);
      } else {
        if (sync) gl.deleteSync(sync);
        frameId = window.setTimeout(cb, 80);
      }
    };
    frameId = window.setTimeout(poll, 5);
  }

  function updateBindingAnimation(dt: number, elapsedSec: number) {
    if (params.viewTarget !== 'dual' || !builtLeft || !builtRight) {
      return;
    }

    const cycleSec = 4.8;
    if (params.playBindingAnim && !params.reducedMotion) {
      animElapsed = (animElapsed + dt) % cycleSec;
    } else {
      animElapsed = 2.6;
    }
    const u = animElapsed / cycleSec;
    const mech = params.mechanismType ?? 'tumor-immunotherapy';
    const g1Name = params.structure1?.gene ?? 'Target 1';
    const g2Name = params.structure2?.gene ?? 'Target 2';

    // Smooth approach/hold/withdraw envelope
    let approach = 0;
    if (u < 0.34) {
      approach = easeInOutCubic(u / 0.34);
    } else if (u < 0.80) {
      approach = 1.0;
    } else {
      approach = 1.0 - easeInOutCubic((u - 0.80) / 0.20);
    }

    const engaged = u >= 0.30 && u <= 0.84;

    // 1. Combination-specific protein motion:
    // - bridge-two-cells: pulls both cell proteins gently toward center on engagement
    // - secreted ligand (e.g. VEGFA): floats gently until captured by Fab arm
    // - TAA+TAA: stays anchored on shared tumor membrane as antibody clamps down
    const pullFactor =
      mech === 'bridge-two-cells' && params.bindingScenario !== 'blocked'
        ? approach * 0.28
        : 0;
    builtLeft.group.position.x = builtLeft.baseX + pullFactor;
    builtRight.group.position.x = builtRight.baseX - pullFactor;

    const floatLeft = builtLeft.isSecreted ? Math.sin(elapsedSec * 1.9) * (0.14 * (1 - 0.65 * approach)) : 0;
    const floatRight = builtRight.isSecreted ? Math.sin(elapsedSec * 2.1 + 1.1) * (0.14 * (1 - 0.65 * approach)) : 0;
    builtLeft.group.position.y = -0.25 + floatLeft;
    builtRight.group.position.y = -0.25 + floatRight;

    // Compute live pocket world coordinates
    _p1World.copy(builtLeft.pocketLocalCenter).add(builtLeft.group.position);
    _p2World.copy(builtRight.pocketLocalCenter).add(builtRight.group.position);

    const midX = (_p1World.x + _p2World.x) * 0.5;
    const maxPocketY = Math.max(_p1World.y, _p2World.y);
    const dockY = maxPocketY + 1.02;
    const restY = dockY + 1.15;
    const blockedSide = params.blockedSide ?? 'target2';

    // Subtle Fab arm spread angle adapts to target separation
    const halfSpan = Math.abs(_p2World.x - _p1World.x) * 0.5;
    const armSpread = THREE.MathUtils.clamp(0.64 + (halfSpan - 1.35) * 0.22, 0.58, 0.96);
    antibodyRig.leftPivot.rotation.z = -armSpread;
    antibodyRig.rightPivot.rotation.z = armSpread;

    let bondAlpha = 0;

    if (params.bindingScenario === 'bind') {
      const breathe = engaged ? Math.sin((u - 0.30) * Math.PI * 6) * 0.024 : 0;
      antibodyRig.group.position.set(
        midX,
        THREE.MathUtils.lerp(restY, dockY, approach) + breathe,
        0
      );
      antibodyRig.group.rotation.z = 0;
      bondAlpha = engaged ? Math.min(0.88, Math.sin(((u - 0.30) / 0.54) * Math.PI) * 1.45) : 0;

      builtLeft.barrierGroup.visible = false;
      builtRight.barrierGroup.visible = false;
      builtLeft.pocketRingMat.color.copy(engaged ? _cBound : _cNeutralRing);
      builtLeft.pocketRingMat.emissive.copy(engaged ? _cBound : _cNeutralRing);
      builtLeft.pocketRing.scale.setScalar(engaged ? 0.86 : 1.05);
      builtRight.pocketRingMat.color.copy(engaged ? _cBound : _cNeutralRing);
      builtRight.pocketRingMat.emissive.copy(engaged ? _cBound : _cNeutralRing);
      builtRight.pocketRing.scale.setScalar(engaged ? 0.86 : 1.05);

      if (u < 0.32) {
        emitPhase({
          state: 'approaching',
          stepLabel: 'Approaching Pockets',
          detail:
            mech === 'bridge-two-cells'
              ? `Reaching across cells toward ${g1Name} and ${g2Name}`
              : mech === 'TAA+TAA'
                ? `Aligning both arms above ${g1Name} & ${g2Name} on the tumor surface`
                : `Aligning bispecific arms with ${g1Name} & ${g2Name} pockets`,
        });
      } else if (u < 0.82) {
        emitPhase({
          state: 'bound',
          stepLabel:
            mech === 'bridge-two-cells'
              ? 'Cell-to-Cell Bridge Locked'
              : mech === 'TAA+TAA'
                ? 'Dual Tumor Receptor Lock'
                : builtRight.isSecreted || builtLeft.isSecreted
                  ? 'Receptor Bound & Ligand Captured'
                  : 'Immune Synapse Dual Lock',
          detail:
            mech === 'bridge-two-cells'
              ? `Pulling ${g1Name} and ${g2Name} together across the two cells`
              : mech === 'TAA+TAA'
                ? `Clamping ${g1Name} + ${g2Name} simultaneously on the same cell`
                : builtRight.isSecreted || builtLeft.isSecreted
                  ? `Anchoring checkpoint while neutralizing soluble ligand`
                  : `Both arms lock cleanly onto exposed ${g1Name} & ${g2Name} pockets`,
        });
      } else {
        emitPhase({
          state: 'approaching',
          stepLabel: 'Replaying Binding Cycle',
          detail: 'Resetting positions to repeat docking animation',
        });
      }
    } else if (params.bindingScenario === 'conditional') {
      const wobble = engaged ? Math.sin((u - 0.30) * Math.PI * 10) * 0.065 : 0;
      const recoilY = engaged ? 0.2 + Math.abs(wobble) * 0.45 : 0;
      antibodyRig.group.position.set(
        midX,
        THREE.MathUtils.lerp(restY, dockY, approach) + recoilY,
        0
      );
      antibodyRig.group.rotation.z = engaged ? 0.12 + wobble : 0;
      bondAlpha = engaged ? 0.72 : 0;

      const tightSideLeft = blockedSide === 'target1';
      builtLeft.barrierGroup.visible = engaged && tightSideLeft;
      builtRight.barrierGroup.visible = engaged && !tightSideLeft;
      builtLeft.barrierMat.color.copy(_cCaution);
      builtRight.barrierMat.color.copy(_cCaution);
      builtLeft.barrierMat.opacity = 0.28;
      builtRight.barrierMat.opacity = 0.28;

      builtLeft.pocketRingMat.color.copy(engaged ? (tightSideLeft ? _cCaution : _cBound) : _cNeutralRing);
      builtLeft.pocketRingMat.emissive.copy(engaged ? (tightSideLeft ? _cCaution : _cBound) : _cNeutralRing);
      builtRight.pocketRingMat.color.copy(engaged ? (!tightSideLeft ? _cCaution : _cBound) : _cNeutralRing);
      builtRight.pocketRingMat.emissive.copy(engaged ? (!tightSideLeft ? _cCaution : _cBound) : _cNeutralRing);

      if (u < 0.32) {
        emitPhase({
          state: 'approaching',
          stepLabel: 'Testing Dual-Arm Reach',
          detail: `Checking membrane clearance for ${g1Name} + ${g2Name}`,
        });
      } else if (u < 0.82) {
        emitPhase({
          state: 'caution',
          stepLabel:
            mech === 'bridge-two-cells'
              ? 'Tight Membrane Bridge'
              : 'Partial Fit · Tight Loop',
          detail: `One arm locks easily; the other meets tight membrane-loop clearance`,
        });
      } else {
        emitPhase({
          state: 'approaching',
          stepLabel: 'Replaying Reach Check',
          detail: 'Resetting positions to repeat docking animation',
        });
      }
    } else {
      // Blocked: show barrier shield on the failing target(s) + recoil bounce
      const bounceProgress = engaged ? (u - 0.30) / 0.54 : 0;
      const bounceCurve = engaged ? Math.sin(bounceProgress * Math.PI) : 0;
      const shake = engaged ? Math.sin(bounceProgress * Math.PI * 10) * 0.045 : 0;
      const recoilY = bounceCurve * 0.62;
      const tiltZ =
        blockedSide === 'target1'
          ? -bounceCurve * 0.22 + shake
          : blockedSide === 'target2'
            ? bounceCurve * 0.22 + shake
            : shake * 1.3;

      antibodyRig.group.position.set(
        midX,
        THREE.MathUtils.lerp(restY, dockY + 0.08, approach) + recoilY,
        0
      );
      antibodyRig.group.rotation.z = tiltZ;
      bondAlpha = 0;

      const showLeftBlock = blockedSide === 'target1' || blockedSide === 'both';
      const showRightBlock = blockedSide === 'target2' || blockedSide === 'both';
      const shieldOpacity = engaged ? 0.34 + 0.14 * Math.sin(bounceProgress * Math.PI * 6) : 0;

      builtLeft.barrierGroup.visible = engaged && showLeftBlock;
      builtRight.barrierGroup.visible = engaged && showRightBlock;
      builtLeft.barrierMat.color.copy(_cBlocked);
      builtRight.barrierMat.color.copy(_cBlocked);
      builtLeft.barrierMat.opacity = shieldOpacity;
      builtRight.barrierMat.opacity = shieldOpacity;

      builtLeft.pocketRingMat.color.copy(engaged && showLeftBlock ? _cBlocked : _cCaution);
      builtLeft.pocketRingMat.emissive.copy(engaged && showLeftBlock ? _cBlocked : _cCaution);
      builtRight.pocketRingMat.color.copy(engaged && showRightBlock ? _cBlocked : _cCaution);
      builtRight.pocketRingMat.emissive.copy(engaged && showRightBlock ? _cBlocked : _cCaution);

      if (u < 0.32) {
        emitPhase({
          state: 'approaching',
          stepLabel: 'Attempting Pocket Dock',
          detail: `Lowering toward ${g1Name} and ${g2Name}`,
        });
      } else if (u < 0.82) {
        const blockerName =
          blockedSide === 'both'
            ? `${g1Name} & ${g2Name}`
            : blockedSide === 'target1'
              ? g1Name
              : g2Name;
        emitPhase({
          state: 'blocked',
          stepLabel: 'Blocked · Antibody Recoils',
          detail: `${blockerName} blocks dual-Fab docking (shielded / crowded pocket)`,
        });
      } else {
        emitPhase({
          state: 'approaching',
          stepLabel: 'Withdrawing After Block',
          detail: 'Resetting positions to repeat docking check',
        });
      }
    }

    // Update world matrices & bond beams cleanly
    antibodyRig.group.updateMatrixWorld(true);
    antibodyRig.leftTipAnchor.getWorldPosition(_leftTipWorld);
    antibodyRig.rightTipAnchor.getWorldPosition(_rightTipWorld);

    if (bondAlpha > 0.02) {
      const leftCaution = params.bindingScenario === 'conditional' && blockedSide === 'target1';
      const rightCaution = params.bindingScenario === 'conditional' && blockedSide !== 'target1';

      antibodyRig.leftBeam.visible = true;
      antibodyRig.leftBeamMat.color.copy(leftCaution ? _cCaution : _cBound);
      antibodyRig.leftBeamMat.opacity = leftCaution ? bondAlpha * 0.45 : bondAlpha;
      positionBeamBetween(antibodyRig.leftBeam, _leftTipWorld, _p1World);

      antibodyRig.rightBeam.visible = true;
      antibodyRig.rightBeamMat.color.copy(rightCaution ? _cCaution : _cBound);
      antibodyRig.rightBeamMat.opacity = rightCaution ? bondAlpha * 0.45 : bondAlpha;
      positionBeamBetween(antibodyRig.rightBeam, _rightTipWorld, _p2World);
    } else {
      antibodyRig.leftBeam.visible = false;
      antibodyRig.rightBeam.visible = false;
    }
  }

  let totalElapsed = 0;

  function tick() {
    if (disposed) return;
    const dt = Math.min(clock.getDelta(), 0.1);
    totalElapsed += dt;

    if (params.autoRotate && !params.reducedMotion) {
      rootGroup.rotation.y += dt * 0.3;
    }

    updateBindingAnimation(dt, totalElapsed);

    if (cameraTransitionActive) {
      camera.position.x = THREE.MathUtils.damp(camera.position.x, _desiredCamPos.x, 6.5, dt);
      camera.position.y = THREE.MathUtils.damp(camera.position.y, _desiredCamPos.y, 6.5, dt);
      camera.position.z = THREE.MathUtils.damp(camera.position.z, _desiredCamPos.z, 6.5, dt);
      controls.target.x = THREE.MathUtils.damp(controls.target.x, _desiredTarget.x, 6.5, dt);
      controls.target.y = THREE.MathUtils.damp(controls.target.y, _desiredTarget.y, 6.5, dt);
      controls.target.z = THREE.MathUtils.damp(controls.target.z, _desiredTarget.z, 6.5, dt);
      if (
        camera.position.distanceTo(_desiredCamPos) < 0.02 &&
        controls.target.distanceTo(_desiredTarget) < 0.02
      ) {
        cameraTransitionActive = false;
      }
    }

    controls.update();
    renderer.render(scene, camera);

    if (isSoftwareGl) {
      afterDraw(tick);
    } else {
      frameId = requestAnimationFrame(tick);
    }
  }

  if (!isSoftwareGl) {
    renderer.compile(scene, camera);
  }
  tick();

  return {
    updateParams(next: ProteinSceneParams) {
      const needsRebuild =
        next.structure1 !== params.structure1 ||
        next.structure2 !== params.structure2 ||
        next.viewTarget !== params.viewTarget ||
        next.colorMode !== params.colorMode ||
        next.renderStyle !== params.renderStyle ||
        next.mechanismType !== params.mechanismType;
      if (next.replayVersion !== params.replayVersion) {
        animElapsed = 0;
        lastPhaseKey = '';
      }
      params = { ...next };
      if (needsRebuild) {
        lastPhaseKey = '';
        rebuildProteins();
      }
    },
    setCameraPreset(preset: CameraPreset) {
      const isDual = params.viewTarget === 'dual';
      if (preset === 'front') {
        _desiredCamPos.set(0, 0.55, isDual ? 6.5 : 4.8);
        _desiredTarget.set(0, 0.15, 0);
      } else if (preset === 'pocket') {
        _desiredTarget.copy(pocketFocusPoint);
        _desiredCamPos.set(
          pocketFocusPoint.x + 1.0,
          pocketFocusPoint.y + 0.65,
          pocketFocusPoint.z + (isDual ? 4.4 : 3.1)
        );
      }
      cameraTransitionActive = true;
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('pointerleave', onPointerLeave);
      renderer.domElement.removeEventListener('webglcontextlost', onContextLost);
      controls.dispose();
      for (const d of activeDisposables) d.dispose();
      for (const d of antibodyRig.disposables) d.dispose();
      envTex.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    },
  };
}
