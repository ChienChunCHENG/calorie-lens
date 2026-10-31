// Track -> item assignment with short-term re-attachment. Pure (no DOM).
import type { Item } from './meal.ts';

export interface Rect { x1: number; y1: number; x2: number; y2: number }
export interface TrackRef { id: number; box: Rect; label?: string }

export const REATTACH_MS = 1500; // a lost track keeps its assignment this long and may re-attach to a new overlapping track
export const MEAL_GRACE_MS = 5000; // lost items still count in the meal total this long
export const REATTACH_IOU = 0.3;

export function rectIou(a: Rect, b: Rect): number {
  const iw = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const ih = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

interface Entry { item: Item | null; box: Rect; label?: string }
interface Ghost { item: Item; box: Rect; lostAt: number }

export class Assignments {
  private entries = new Map<number, Entry>();
  private ghosts: Ghost[] = [];

  get(id: number): Item | null { return this.entries.get(id)?.item ?? null; }

  /** Call after each tracker update. autoFor(label) maps a track's voted label to an item (null: not a known dish). */
  sync(tracks: TrackRef[], now: number, autoFor?: (label: string | undefined) => Item | null): void {
    const present = new Set(tracks.map(t => t.id));
    for (const [id, e] of this.entries) {
      if (present.has(id)) continue;
      if (e.item) this.ghosts.push({ item: e.item, box: e.box, lostAt: now });
      this.entries.delete(id);
    }
    this.ghosts = this.ghosts.filter(g => now - g.lostAt <= MEAL_GRACE_MS);
    for (const t of tracks) {
      const e = this.entries.get(t.id);
      if (e) {
        e.box = t.box;
        if (autoFor && t.label !== e.label) { e.label = t.label; e.item = autoFor(t.label); } // track's vote changed
        continue;
      }
      let item: Item | null = null;
      let best = -1, bi = REATTACH_IOU;
      this.ghosts.forEach((g, i) => {
        if (now - g.lostAt > REATTACH_MS) return;
        const v = rectIou(g.box, t.box);
        if (v > bi) { bi = v; best = i; }
      });
      if (best >= 0) item = this.ghosts.splice(best, 1)[0].item;
      else item = autoFor?.(t.label) ?? null;
      this.entries.set(t.id, { item, box: t.box, label: t.label });
    }
  }

  /** Items on screen now plus recently lost ones (for the meal total). */
  activeItems(): Item[] {
    const out: Item[] = [];
    for (const e of this.entries.values()) if (e.item) out.push(e.item);
    for (const g of this.ghosts) out.push(g.item);
    return out;
  }

  clear(): void { this.entries.clear(); this.ghosts = []; }
}
