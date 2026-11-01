// Tiny IndexedDB wrapper for the meal log (no dependencies). Every call can reject; callers show a message.
import type { Backup, Meal } from './log.ts';
import { BACKUP_VERSION, parseBackup } from './log.ts';

const DB = 'calorie-lens', STORE = 'meals';

function open(): Promise<IDBDatabase> {
  return new Promise((res, rej) => {
    if (typeof indexedDB === 'undefined') return rej(new Error('IndexedDB unavailable'));
    let rq: IDBOpenDBRequest;
    try { rq = indexedDB.open(DB, 1); } catch (e) { return rej(e); }
    rq.onupgradeneeded = () => { rq.result.createObjectStore(STORE, { keyPath: 'id' }).createIndex('ts', 'ts'); };
    rq.onsuccess = () => res(rq.result);
    rq.onerror = () => rej(rq.error ?? new Error('IndexedDB open failed'));
    rq.onblocked = () => rej(new Error('IndexedDB blocked'));
  });
}

async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  try {
    return await new Promise<T>((res, rej) => {
      const t = db.transaction(STORE, mode), rq = fn(t.objectStore(STORE));
      t.oncomplete = () => res(rq.result);
      t.onerror = t.onabort = () => rej(t.error ?? new Error('IndexedDB transaction failed'));
    });
  } finally { db.close(); }
}

export const getAllMeals = (): Promise<Meal[]> => tx('readonly', s => s.getAll() as IDBRequest<Meal[]>);
export const putMeal = (m: Meal): Promise<unknown> => tx('readwrite', s => s.put(m));
export const deleteMeal = (id: string): Promise<unknown> => tx('readwrite', s => s.delete(id));

/** Probe: can we open the DB at all (private mode / blocked storage)? */
export async function storageOk(): Promise<string | null> {
  try { await getAllMeals(); return null; } catch (e: any) { return String(e?.message ?? e); }
}

function blobToDataUrl(b: Blob): Promise<string> {
  return new Promise((res, rej) => { const f = new FileReader(); f.onload = () => res(String(f.result)); f.onerror = () => rej(f.error); f.readAsDataURL(b); });
}
async function dataUrlToBlob(u: string): Promise<Blob> { return (await fetch(u)).blob(); }

export async function exportBackup(meals: Meal[]): Promise<string> {
  const out: Backup = {
    app: 'calorie-lens', version: BACKUP_VERSION, exported: new Date().toISOString(),
    meals: await Promise.all(meals.map(async m => ({ ...m, thumb: m.thumb ? await blobToDataUrl(m.thumb) : null }))),
  };
  return JSON.stringify(out);
}

/** Merge a backup into the DB (same meal id = overwritten). Returns the number of meals imported. */
export async function importBackup(text: string): Promise<number> {
  const meals = parseBackup(text);
  for (const m of meals) await putMeal({ ...m, thumb: m.thumb ? await dataUrlToBlob(m.thumb) : null });
  return meals.length;
}
