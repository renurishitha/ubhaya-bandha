import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import type { BindingVerdict, MechanismType } from '../../types.ts';

export interface SceneSimulationParams {
  mechanismType: MechanismType;
  bindingVerdict?: BindingVerdict;
  arm1Hex: number;
  arm2Hex: number;
  fcHex: number;
  dockVersion: number;
  reducedMotion: boolean;
}

// Reusable scratch vectors (zero GC in animation loop)
const _vA = new THREE.Vector3();
const _vB = new THREE.Vector3();
const _color = new THREE.Color();
const _dummy = new THREE.Object3D();

export function detectSoftwareWebGL(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param) ?? '');
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return true;
  }
}

function createHelixRibbonGeometry(
  start: THREE.Vector3,
  end: THREE.Vector3,
  turns: number,
  amp: number,
  phase: number,
  radius: number
): THREE.TubeGeometry {
  const dir = _vA.copy(end).sub(start);
  const len = dir.length();
  const forward = dir.clone().normalize();
  const upRef = Math.abs(forward.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  const sideA = new THREE.Vector3().crossVectors(forward, upRef).normalize();
  const sideB = new THREE.Vector3().crossVectors(forward, sideA).normalize();

  const pts: THREE.Vector3[] = [];
  const steps = 28;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const envelope = Math.sin(Math.PI * t);
    const angle = t * Math.PI * 2 * turns + phase;
    const p = start
      .clone()
      .addScaledVector(forward, len * t)
      .addScaledVector(sideA, Math.cos(angle) * amp * (0.45 + 0.55 * envelope))
      .addScaledVector(sideB, Math.sin(angle) * amp * (0.45 + 0.55 * envelope));
    pts.push(p);
  }
  const curve = new THREE.CatmullRomCurve3(pts);
  return new THREE.TubeGeometry(curve, 36, radius, 10, false);
}

function buildAntigenGroup(
  colorHex: number,
  epitopeHex: number,
  isLeft: boolean,
  disposables: Array<{ dispose: () => void }>
): { group: THREE.Group; matPrimary: THREE.MeshPhysicalMaterial; matEpitope: THREE.MeshStandardMaterial } {
  const group = new THREE.Group();

  const matPrimary = new THREE.MeshPhysicalMaterial({
    color: colorHex,
    roughness: 0.28,
    metalness: 0.05,
    clearcoat: 0.8,
    clearcoatRoughness: 0.16,
  });
  const matEpitope = new THREE.MeshStandardMaterial({
    color: epitopeHex,
    emissive: epitopeHex,
    emissiveIntensity: 0.28,
    roughness: 0.35,
    metalness: 0.15,
  });
  const matStalk = new THREE.MeshStandardMaterial({
    color: 0x64748b,
    roughness: 0.65,
    metalness: 0.12,
  });
  disposables.push(matPrimary, matEpitope, matStalk);

  // Transmembrane stalk helix leading into globular ectodomain
  const sign = isLeft ? -1 : 1;
  const stalkGeo = createHelixRibbonGeometry(
    new THREE.Vector3(sign * 1.15, -0.35, 0),
    new THREE.Vector3(sign * 0.3, -0.05, 0),
    2.2,
    0.09,
    0,
    0.055
  );
  disposables.push(stalkGeo);
  group.add(new THREE.Mesh(stalkGeo, matStalk));

  // Multi-lobed 4-part biological glycoprotein ectodomain cluster
  const lobeGeo = new THREE.SphereGeometry(0.26, 24, 20);
  const subLobeGeo = new THREE.SphereGeometry(0.19, 20, 16);
  const capGeo = new THREE.CapsuleGeometry(0.14, 0.18, 8, 16);
  disposables.push(lobeGeo, subLobeGeo, capGeo);

  const mainLobe = new THREE.Mesh(lobeGeo, matPrimary);
  mainLobe.position.set(sign * 0.24, 0.04, 0);
  group.add(mainLobe);

  const upperLobe = new THREE.Mesh(subLobeGeo, matPrimary);
  upperLobe.position.set(sign * 0.14, 0.24, 0.08);
  group.add(upperLobe);

  const lowerLobe = new THREE.Mesh(subLobeGeo, matPrimary);
  lowerLobe.position.set(sign * 0.16, -0.15, -0.07);
  group.add(lowerLobe);

  // Exposed binding epitope patch facing antibody paratope
  const epitopePatch = new THREE.Mesh(capGeo, matEpitope);
  epitopePatch.position.set(-sign * 0.04, 0.04, 0);
  epitopePatch.rotation.z = Math.PI / 2;
  group.add(epitopePatch);

  return { group, matPrimary, matEpitope };
}

function buildPhospholipidBilayer(
  mode: 'trans-left' | 'trans-right' | 'cis-bottom',
  headHex: number,
  disposables: Array<{ dispose: () => void }>
): THREE.Group {
  const group = new THREE.Group();
  const rows = mode === 'cis-bottom' ? 12 : 9;
  const cols = mode === 'cis-bottom' ? 7 : 6;
  const totalHeads = rows * cols * 2; // outer + inner leaflet

  const headGeo = new THREE.SphereGeometry(0.085, 14, 12);
  const tailGeo = new THREE.CylinderGeometry(0.022, 0.022, 0.22, 8);
  const headMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.35,
    metalness: 0.05,
  });
  const tailMat = new THREE.MeshStandardMaterial({
    color: 0xcbd5e1,
    roughness: 0.72,
    metalness: 0.02,
  });
  disposables.push(headGeo, tailGeo, headMat, tailMat);

  const heads = new THREE.InstancedMesh(headGeo, headMat, totalHeads);
  const tails = new THREE.InstancedMesh(tailGeo, tailMat, totalHeads);
  heads.frustumCulled = false;
  tails.frustumCulled = false;

  const baseColor = new THREE.Color(headHex);
  let idx = 0;

  for (let r = 0; r < rows; r++) {
    const u = rows > 1 ? r / (rows - 1) - 0.5 : 0;
    for (let c = 0; c < cols; c++) {
      const v = cols > 1 ? c / (cols - 1) - 0.5 : 0;
      for (let leaflet = 0; leaflet < 2; leaflet++) {
        const offset = leaflet === 0 ? 0.16 : -0.16;

        if (mode === 'cis-bottom') {
          const x = u * 5.2;
          const z = v * 1.9;
          const archY = -1.62 + Math.cos(u * 1.4) * 0.24 - Math.abs(v) * 0.08;
          _dummy.position.set(x, archY + offset, z);
          _dummy.rotation.set(0, 0, -u * 0.24);
        } else if (mode === 'trans-left') {
          const y = u * 2.7;
          const z = v * 1.7;
          const archX = -2.85 + Math.cos(u * 1.4) * 0.28;
          _dummy.position.set(archX + offset, y, z);
          _dummy.rotation.set(0, 0, Math.PI / 2 + u * 0.2);
        } else {
          const y = u * 2.7;
          const z = v * 1.7;
          const archX = 2.85 - Math.cos(u * 1.4) * 0.28;
          _dummy.position.set(archX - offset, y, z);
          _dummy.rotation.set(0, 0, -Math.PI / 2 - u * 0.2);
        }

        const scaleJitter = 0.9 + (((r * 7 + c * 13 + leaflet) % 5) / 5) * 0.22;
        _dummy.scale.setScalar(scaleJitter);
        _dummy.updateMatrix();
        heads.setMatrixAt(idx, _dummy.matrix);

        // Position hydrophobic tail inward toward bilayer core
        _vB.set(0, leaflet === 0 ? -0.11 : 0.11, 0).applyEuler(_dummy.rotation);
        _dummy.position.add(_vB);
        _dummy.scale.set(1, 1, 1);
        _dummy.updateMatrix();
        tails.setMatrixAt(idx, _dummy.matrix);

        _color.copy(baseColor).offsetHSL(0, 0, leaflet === 0 ? 0.04 : -0.06);
        heads.setColorAt(idx, _color);
        idx++;
      }
    }
  }

  heads.instanceMatrix.needsUpdate = true;
  if (heads.instanceColor) heads.instanceColor.needsUpdate = true;
  tails.instanceMatrix.needsUpdate = true;
  heads.computeBoundingSphere();
  tails.computeBoundingSphere();

  group.add(heads, tails);
  return group;
}

export interface BispecificSceneHandle {
  updateParams: (next: SceneSimulationParams) => void;
  dispose: () => void;
}

export function mountBispecific3DScene(
  container: HTMLElement,
  initialParams: SceneSimulationParams
): BispecificSceneHandle {
  const isSoftwareGl = detectSoftwareWebGL();
  let params: SceneSimulationParams = { ...initialParams };

  const disposables: Array<{ dispose: () => void }> = [];

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xfaf5ec);
  scene.fog = new THREE.FogExp2(0xfaf5ec, 0.035);

  const width = Math.max(240, container.clientWidth);
  const height = Math.max(176, container.clientHeight);

  const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 60);
  camera.position.set(0, 0.45, 5.4);

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
  disposables.push(envTex);

  // Studio 3-point + cool biological ambient lighting
  const hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0xc7d2fe, 0.95);
  scene.add(hemiLight);

  const keyLight = new THREE.DirectionalLight(0xfffbeb, 1.55);
  keyLight.position.set(4, 6, 5);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0x93c5fd, 0.85);
  rimLight.position.set(-5, 2, -4);
  scene.add(rimLight);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.enableZoom = true;
  controls.minDistance = 3.6;
  controls.maxDistance = 8.2;
  controls.minPolarAngle = Math.PI * 0.22;
  controls.maxPolarAngle = Math.PI * 0.72;
  controls.enableDamping = !isSoftwareGl;
  controls.dampingFactor = 0.07;

  // Root assembly group so we can gently breathe / float cleanly
  const rootGroup = new THREE.Group();
  scene.add(rootGroup);

  // --- Bispecific Antibody Assembly (10-part IgG ribbon macromolecule) ---
  const antibodyGroup = new THREE.Group();
  rootGroup.add(antibodyGroup);

  const matArm1 = new THREE.MeshPhysicalMaterial({
    color: params.arm1Hex,
    roughness: 0.26,
    metalness: 0.08,
    clearcoat: 0.85,
    clearcoatRoughness: 0.14,
  });
  const matArm2 = new THREE.MeshPhysicalMaterial({
    color: params.arm2Hex,
    roughness: 0.26,
    metalness: 0.08,
    clearcoat: 0.85,
    clearcoatRoughness: 0.14,
  });
  const matLightChain = new THREE.MeshPhysicalMaterial({
    color: 0xe2e8f0,
    roughness: 0.35,
    metalness: 0.15,
    clearcoat: 0.5,
  });
  const matHinge = new THREE.MeshStandardMaterial({
    color: 0x334155,
    roughness: 0.25,
    metalness: 0.75,
  });
  disposables.push(matArm1, matArm2, matLightChain, matHinge);

  // Fc Heterodimeric Heavy Chains (2 intertwined helical domains)
  const fcLeftGeo = createHelixRibbonGeometry(
    new THREE.Vector3(-0.11, -1.25, 0),
    new THREE.Vector3(-0.08, -0.12, 0),
    1.8,
    0.075,
    0,
    0.085
  );
  const fcRightGeo = createHelixRibbonGeometry(
    new THREE.Vector3(0.11, -1.25, 0),
    new THREE.Vector3(0.08, -0.12, 0),
    1.8,
    0.075,
    Math.PI,
    0.085
  );
  const ch3DomainGeo = new THREE.SphereGeometry(0.15, 20, 16);
  const hingeRingGeo = new THREE.TorusGeometry(0.16, 0.032, 12, 28);
  disposables.push(fcLeftGeo, fcRightGeo, ch3DomainGeo, hingeRingGeo);

  antibodyGroup.add(new THREE.Mesh(fcLeftGeo, matArm1));
  antibodyGroup.add(new THREE.Mesh(fcRightGeo, matArm2));

  const fcBallL = new THREE.Mesh(ch3DomainGeo, matArm1);
  fcBallL.position.set(-0.1, -1.16, 0.04);
  const fcBallR = new THREE.Mesh(ch3DomainGeo, matArm2);
  fcBallR.position.set(0.1, -1.16, -0.04);
  antibodyGroup.add(fcBallL, fcBallR);

  const hinge1 = new THREE.Mesh(hingeRingGeo, matHinge);
  hinge1.rotation.x = Math.PI / 2;
  hinge1.position.set(0, -0.12, 0);
  const hinge2 = new THREE.Mesh(hingeRingGeo, matHinge);
  hinge2.rotation.x = Math.PI / 2;
  hinge2.position.set(0, -0.22, 0);
  antibodyGroup.add(hinge1, hinge2);

  // Left Fab Arm Group (pivoted at hinge)
  const leftFabGroup = new THREE.Group();
  leftFabGroup.position.set(-0.08, -0.1, 0);
  antibodyGroup.add(leftFabGroup);

  const fabLHeavyGeo = createHelixRibbonGeometry(
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(-0.98, 0.62, 0),
    2.1,
    0.075,
    0.2,
    0.082
  );
  const fabLLightGeo = createHelixRibbonGeometry(
    new THREE.Vector3(-0.22, 0.14, 0.14),
    new THREE.Vector3(-1.02, 0.48, 0.12),
    1.8,
    0.06,
    1.7,
    0.062
  );
  const paratopeGeo = new THREE.TorusGeometry(0.17, 0.045, 14, 28, Math.PI * 1.45);
  disposables.push(fabLHeavyGeo, fabLLightGeo, paratopeGeo);

  leftFabGroup.add(new THREE.Mesh(fabLHeavyGeo, matArm1));
  leftFabGroup.add(new THREE.Mesh(fabLLightGeo, matLightChain));
  const leftParatope = new THREE.Mesh(paratopeGeo, matArm1);
  leftParatope.position.set(-1.02, 0.56, 0);
  leftParatope.rotation.z = Math.PI * 0.72;
  leftFabGroup.add(leftParatope);

  // Right Fab Arm Group (pivoted at hinge)
  const rightFabGroup = new THREE.Group();
  rightFabGroup.position.set(0.08, -0.1, 0);
  antibodyGroup.add(rightFabGroup);

  const fabRHeavyGeo = createHelixRibbonGeometry(
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0.98, 0.62, 0),
    2.1,
    0.075,
    1.2,
    0.082
  );
  const fabRLightGeo = createHelixRibbonGeometry(
    new THREE.Vector3(0.22, 0.14, -0.14),
    new THREE.Vector3(1.02, 0.48, -0.12),
    1.8,
    0.06,
    2.8,
    0.062
  );
  disposables.push(fabRHeavyGeo, fabRLightGeo);

  rightFabGroup.add(new THREE.Mesh(fabRHeavyGeo, matArm2));
  rightFabGroup.add(new THREE.Mesh(fabRLightGeo, matLightChain));
  const rightParatope = new THREE.Mesh(paratopeGeo, matArm2);
  rightParatope.position.set(1.02, 0.56, 0);
  rightParatope.rotation.z = -Math.PI * 0.22;
  rightFabGroup.add(rightParatope);

  // --- Antigens 1 & 2 ---
  const antigen1 = buildAntigenGroup(params.arm1Hex, 0x38bdf8, true, disposables);
  const antigen2 = buildAntigenGroup(params.arm2Hex, 0xfb7185, false, disposables);
  rootGroup.add(antigen1.group, antigen2.group);

  // --- Binding Confirmation Rings ---
  const ringGeo = new THREE.TorusGeometry(0.24, 0.024, 12, 32);
  const ringMat1 = new THREE.MeshBasicMaterial({
    color: params.arm1Hex,
    transparent: true,
    opacity: 0,
  });
  const ringMat2 = new THREE.MeshBasicMaterial({
    color: params.arm2Hex,
    transparent: true,
    opacity: 0,
  });
  disposables.push(ringGeo, ringMat1, ringMat2);

  const dockRing1 = new THREE.Mesh(ringGeo, ringMat1);
  dockRing1.position.set(-1.12, 0.5, 0);
  dockRing1.rotation.y = Math.PI / 3;
  const dockRing2 = new THREE.Mesh(ringGeo, ringMat2);
  dockRing2.position.set(1.12, 0.5, 0);
  dockRing2.rotation.y = -Math.PI / 3;
  rootGroup.add(dockRing1, dockRing2);

  // --- Lipid Bilayer Membranes (Trans vs Cis) ---
  const transLeftMembrane = buildPhospholipidBilayer('trans-left', 0x3b82f6, disposables);
  const transRightMembrane = buildPhospholipidBilayer('trans-right', 0x10b981, disposables);
  const cisBottomMembrane = buildPhospholipidBilayer('cis-bottom', 0x6366f1, disposables);
  rootGroup.add(transLeftMembrane, transRightMembrane, cisBottomMembrane);

  // --- Ambient Microenvironment Cytokine Particles (InstancedMesh) ---
  const particleCount = 28;
  const particleGeo = new THREE.SphereGeometry(0.036, 10, 8);
  const particleMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.3,
    metalness: 0.1,
    transparent: true,
    opacity: 0.65,
  });
  disposables.push(particleGeo, particleMat);
  const particles = new THREE.InstancedMesh(particleGeo, particleMat, particleCount);
  particles.frustumCulled = false;

  const particleSeeds = Array.from({ length: particleCount }, (_, i) => {
    const angle = (i / particleCount) * Math.PI * 2;
    return {
      x: Math.cos(angle * 1.7) * 2.3 + ((i % 3) - 1) * 0.45,
      y: Math.sin(angle * 2.3) * 1.15,
      z: Math.sin(angle * 1.1) * 1.2 - 0.3,
      speed: 0.5 + (i % 5) * 0.18,
      phase: i * 0.7,
      scale: 0.65 + (i % 4) * 0.25,
    };
  });
  const pColors = [0x3b82f6, 0x0d9488, 0x8b5cf6, 0xf59e0b, 0xf43f5e];
  for (let i = 0; i < particleCount; i++) {
    _color.setHex(pColors[i % pColors.length]);
    particles.setColorAt(i, _color);
  }
  if (particles.instanceColor) particles.instanceColor.needsUpdate = true;
  rootGroup.add(particles);

  let dockElapsed = 0;
  let lastDockVersion = params.dockVersion;
  let lastMechanism = params.mechanismType;
  let lastVerdict = params.bindingVerdict ?? 'strong-bind';
  let disposed = false;
  let frameId = 0;

  const applyColorsAndVisibility = () => {
    const verdict = params.bindingVerdict ?? 'strong-bind';
    matArm1.color.setHex(params.arm1Hex);
    matArm2.color.setHex(params.arm2Hex);
    antigen1.matPrimary.color.setHex(params.arm1Hex);
    antigen2.matPrimary.color.setHex(params.arm2Hex);
    antigen1.matEpitope.color.setHex(verdict === 'blocked' ? 0xdc2626 : params.arm1Hex);
    antigen1.matEpitope.emissive.setHex(verdict === 'blocked' ? 0xdc2626 : params.arm1Hex);
    antigen2.matEpitope.color.setHex(verdict === 'blocked' ? 0xdc2626 : params.arm2Hex);
    antigen2.matEpitope.emissive.setHex(verdict === 'blocked' ? 0xdc2626 : params.arm2Hex);
    ringMat1.color.setHex(
      verdict === 'blocked' ? 0xdc2626 : verdict === 'conditional-bind' ? 0xd97706 : params.arm1Hex
    );
    ringMat2.color.setHex(
      verdict === 'blocked' ? 0xdc2626 : verdict === 'conditional-bind' ? 0xd97706 : params.arm2Hex
    );

    const isCis = params.mechanismType === 'TAA+TAA';
    transLeftMembrane.visible = !isCis;
    transRightMembrane.visible = !isCis;
    cisBottomMembrane.visible = isCis;
  };

  applyColorsAndVisibility();

  // Handle ResizeObserver for responsive card width
  const resizeObserver = new ResizeObserver(() => {
    if (disposed) return;
    const w = Math.max(240, container.clientWidth);
    const h = Math.max(176, container.clientHeight);
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

  function updateKinematics(dt: number, totalTime: number) {
    const verdict = params.bindingVerdict ?? 'strong-bind';
    if (
      params.dockVersion !== lastDockVersion ||
      params.mechanismType !== lastMechanism ||
      verdict !== lastVerdict
    ) {
      lastDockVersion = params.dockVersion;
      lastMechanism = params.mechanismType;
      lastVerdict = verdict;
      dockElapsed = 0;
      applyColorsAndVisibility();
    }

    dockElapsed += dt;
    const isCis = params.mechanismType === 'TAA+TAA';
    const dockX = 1.18;
    const dockY = 0.48;
    const startOffset = isCis ? { x: 0.45, y: -0.65 } : { x: 1.05, y: 0.18 };

    if (verdict === 'blocked') {
      // Blocked: antibody approaches partway, hits red steric barrier rings, shakes and recoils back
      const cycle = params.reducedMotion ? 0.55 : (dockElapsed % 3.2) / 3.2;
      const attempt =
        cycle < 0.35
          ? 1 - Math.pow(1 - cycle / 0.35, 2)
          : cycle < 0.72
            ? 1 - ((cycle - 0.35) / 0.37) * 0.72
            : 0.28 * (1 - (cycle - 0.72) / 0.28);

      const impactWindow = cycle >= 0.3 && cycle <= 0.72;
      const shake =
        !params.reducedMotion && impactWindow
          ? Math.sin((cycle - 0.3) * Math.PI * 12) * 0.08
          : 0;
      const recoilDrop = impactWindow ? Math.sin(((cycle - 0.3) / 0.42) * Math.PI) * 0.32 : 0;

      // Never finishes docking — stops short and recoils downward
      antibodyGroup.position.y = THREE.MathUtils.lerp(-0.55, -0.18, attempt) - recoilDrop;
      antibodyGroup.rotation.y = params.reducedMotion ? 0.1 : Math.sin(totalTime * 0.5) * 0.1;
      antibodyGroup.rotation.z = shake;

      // Fab arms stay splayed open (cannot clasp targets)
      const splayedArm = 0.26 + (impactWindow ? 0.12 : 0);
      leftFabGroup.rotation.z = splayedArm;
      rightFabGroup.rotation.z = -splayedArm;

      // Antigens stay separated (cannot close onto Fab paratopes)
      const partialApproach = attempt * 0.32;
      antigen1.group.position.set(
        THREE.MathUtils.lerp(-dockX - startOffset.x, -dockX - 0.38, partialApproach),
        THREE.MathUtils.lerp(dockY + startOffset.y, dockY + 0.08, partialApproach),
        0
      );
      antigen1.group.rotation.z = isCis ? 0.42 : -0.18;

      antigen2.group.position.set(
        THREE.MathUtils.lerp(dockX + startOffset.x, dockX + 0.38, partialApproach),
        THREE.MathUtils.lerp(dockY + startOffset.y, dockY + 0.08, partialApproach),
        0
      );
      antigen2.group.rotation.z = isCis ? -0.42 : 0.18;

      if (!isCis) {
        transLeftMembrane.position.x = -0.35;
        transRightMembrane.position.x = 0.35;
      } else {
        cisBottomMembrane.position.y = -0.22;
      }

      // Red pulsing repulsion rings at impact
      const pulse = impactWindow
        ? 0.45 + 0.4 * Math.sin(((cycle - 0.3) / 0.42) * Math.PI * 3)
        : 0.18;
      ringMat1.opacity = pulse;
      ringMat2.opacity = pulse;
      dockRing1.scale.setScalar(impactWindow ? 1.25 : 0.95);
      dockRing2.scale.setScalar(impactWindow ? 1.25 : 0.95);
    } else if (verdict === 'conditional-bind') {
      // Conditional / partial fit: docks with slight asymmetric tilt and amber caution halo
      const rawProgress = params.reducedMotion ? 1 : Math.min(1, dockElapsed / 1.25);
      const ease = 1 - Math.pow(1 - rawProgress, 3);

      antibodyGroup.position.y = THREE.MathUtils.lerp(-0.45, -0.04, ease);
      antibodyGroup.rotation.y = params.reducedMotion ? 0.12 : Math.sin(totalTime * 0.55) * 0.14;
      const wobble = !params.reducedMotion && rawProgress >= 1 ? Math.sin(totalTime * 3.4) * 0.035 : 0;
      antibodyGroup.rotation.z = THREE.MathUtils.lerp(0, 0.08, ease) + wobble;

      const armSpread = THREE.MathUtils.lerp(0.24, 0.06, ease);
      leftFabGroup.rotation.z = armSpread;
      rightFabGroup.rotation.z = -armSpread * 1.45;

      antigen1.group.position.set(
        THREE.MathUtils.lerp(-dockX - startOffset.x, -dockX, ease),
        THREE.MathUtils.lerp(dockY + startOffset.y, dockY, ease),
        0
      );
      antigen1.group.rotation.z = isCis ? THREE.MathUtils.lerp(0.45, 0.22, ease) : THREE.MathUtils.lerp(-0.2, 0.05, ease);

      antigen2.group.position.set(
        THREE.MathUtils.lerp(dockX + startOffset.x, dockX + 0.12, ease),
        THREE.MathUtils.lerp(dockY + startOffset.y, dockY - 0.06, ease),
        0
      );
      antigen2.group.rotation.z = isCis ? THREE.MathUtils.lerp(-0.45, -0.28, ease) : THREE.MathUtils.lerp(0.2, -0.08, ease);

      if (!isCis) {
        transLeftMembrane.position.x = THREE.MathUtils.lerp(-0.45, -0.08, ease);
        transRightMembrane.position.x = THREE.MathUtils.lerp(0.45, 0.12, ease);
      } else {
        cisBottomMembrane.position.y = THREE.MathUtils.lerp(-0.28, -0.05, ease);
      }

      const cautionPulse = rawProgress >= 0.85 ? 0.32 + 0.25 * Math.sin(totalTime * 3.2) : 0;
      ringMat1.opacity = cautionPulse * 0.5;
      ringMat2.opacity = cautionPulse;
      dockRing1.scale.setScalar(1.0);
      dockRing2.scale.setScalar(1.12);
    } else {
      // Strong clean bind
      const rawProgress = params.reducedMotion ? 1 : Math.min(1, dockElapsed / 1.15);
      const ease = 1 - Math.pow(1 - rawProgress, 3);

      antibodyGroup.position.y = THREE.MathUtils.lerp(-0.45, 0.02, ease);
      antibodyGroup.rotation.y = params.reducedMotion ? 0.12 : Math.sin(totalTime * 0.55) * 0.14;
      antibodyGroup.rotation.z = 0;
      if (!params.reducedMotion && rawProgress >= 1) {
        antibodyGroup.position.y += Math.sin(totalTime * 1.6) * 0.025;
      }

      const armSpread = THREE.MathUtils.lerp(0.24, 0, ease);
      leftFabGroup.rotation.z = armSpread;
      rightFabGroup.rotation.z = -armSpread;

      antigen1.group.position.set(
        THREE.MathUtils.lerp(-dockX - startOffset.x, -dockX, ease),
        THREE.MathUtils.lerp(dockY + startOffset.y, dockY, ease),
        0
      );
      antigen1.group.rotation.z = isCis ? THREE.MathUtils.lerp(0.45, 0.22, ease) : THREE.MathUtils.lerp(-0.2, 0.05, ease);

      antigen2.group.position.set(
        THREE.MathUtils.lerp(dockX + startOffset.x, dockX, ease),
        THREE.MathUtils.lerp(dockY + startOffset.y, dockY, ease),
        0
      );
      antigen2.group.rotation.z = isCis ? THREE.MathUtils.lerp(-0.45, -0.22, ease) : THREE.MathUtils.lerp(0.2, -0.05, ease);

      if (!isCis) {
        transLeftMembrane.position.x = THREE.MathUtils.lerp(-0.45, 0, ease);
        transRightMembrane.position.x = THREE.MathUtils.lerp(0.45, 0, ease);
      } else {
        cisBottomMembrane.position.y = THREE.MathUtils.lerp(-0.28, 0, ease);
      }

      const ringWindow = Math.max(0, Math.min(1, (dockElapsed - 0.72) / 0.85));
      const ringAlpha = !params.reducedMotion && ringWindow > 0 && ringWindow < 1 ? Math.sin(ringWindow * Math.PI) * 0.85 : 0;
      const ringScale = 0.85 + ringWindow * 1.15;
      ringMat1.opacity = ringAlpha;
      ringMat2.opacity = ringAlpha;
      dockRing1.scale.setScalar(ringScale);
      dockRing2.scale.setScalar(ringScale);
    }

    // Ambient cytokine particle drift
    for (let i = 0; i < particleCount; i++) {
      const s = particleSeeds[i];
      const bob = params.reducedMotion ? 0 : Math.sin(totalTime * s.speed + s.phase) * 0.14;
      _dummy.position.set(s.x, s.y + bob, s.z);
      _dummy.rotation.set(0, 0, 0);
      _dummy.scale.setScalar(s.scale);
      _dummy.updateMatrix();
      particles.setMatrixAt(i, _dummy.matrix);
    }
    particles.instanceMatrix.needsUpdate = true;
  }

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

  function tick() {
    if (disposed) return;
    const dt = Math.min(clock.getDelta(), 0.1);
    const total = clock.elapsedTime;
    updateKinematics(dt, total);
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
    updateParams(next: SceneSimulationParams) {
      params = { ...next };
      applyColorsAndVisibility();
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener('webglcontextlost', onContextLost);
      controls.dispose();
      for (const item of disposables) {
        item.dispose();
      }
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    },
  };
}
