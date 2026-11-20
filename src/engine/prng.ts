/**
 * Deterministic 32-bit PRNG (Mulberry32) shared between TypeScript and Python
 * so identical seeds produce identical synthetic scRNA-seq matrices and CV folds.
 */
export class Mulberry32 {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  nextUint32(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  }

  random(): number {
    return this.nextUint32() / 4294967296;
  }

  randint(low: number, highExclusive: number): number {
    return low + Math.floor(this.random() * (highExclusive - low));
  }

  normal(mean = 0, std = 1): number {
    const u1 = Math.max(1e-12, this.random());
    const u2 = this.random();
    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + std * z;
  }
}

export function seededShuffle<T>(items: T[], rng: Mulberry32): T[] {
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = rng.randint(0, i + 1);
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}
