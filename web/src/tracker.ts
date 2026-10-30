import { Box, Detection, iou } from './yolo';

export interface Track extends Box { id: number; cls: number; score: number; hits: number; missed: number; target: Box }

export class IouTracker {
  tracks: Track[] = [];
  private nextId = 1;
  constructor(private matchIou = 0.3, private maxMissed = 4, private tauMs = 90) {}

  /** Call after each inference. */
  update(dets: Detection[]): void {
    const usedD = new Set<number>();
    const usedT = new Set<number>();
    const pairs: { t: number; d: number; v: number }[] = [];
    this.tracks.forEach((t, ti) => dets.forEach((d, di) => {
      if (t.cls !== d.cls) return;
      const v = iou(t.target, d);
      if (v >= this.matchIou) pairs.push({ t: ti, d: di, v });
    }));
    pairs.sort((a, b) => b.v - a.v);
    for (const p of pairs) {
      if (usedT.has(p.t) || usedD.has(p.d)) continue;
      usedT.add(p.t); usedD.add(p.d);
      const t = this.tracks[p.t], d = dets[p.d];
      t.target = { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 };
      t.score = d.score; t.hits++; t.missed = 0;
    }
    this.tracks.forEach((t, ti) => { if (!usedT.has(ti)) t.missed++; });
    dets.forEach((d, di) => {
      if (usedD.has(di)) return;
      this.tracks.push({ id: this.nextId++, cls: d.cls, score: d.score, hits: 1, missed: 0,
        x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2, target: { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 } });
    });
    this.tracks = this.tracks.filter(t => t.missed <= this.maxMissed);
  }

  /** Call every animation frame: time-based exponential smoothing toward targets. */
  step(dtMs: number): void {
    const a = 1 - Math.exp(-dtMs / this.tauMs);
    for (const t of this.tracks) {
      t.x1 += (t.target.x1 - t.x1) * a; t.y1 += (t.target.y1 - t.y1) * a;
      t.x2 += (t.target.x2 - t.x2) * a; t.y2 += (t.target.y2 - t.y2) * a;
    }
  }

  /** Tracks worth drawing: confirmed (seen twice or strong) and not stale. */
  visible(): Track[] {
    return this.tracks.filter(t => t.missed <= 2 && (t.hits >= 2 || t.score > 0.6));
  }
}
