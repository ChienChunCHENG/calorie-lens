// IEEE half-precision helpers (no Float16Array dependency: iOS Safari lacks it).
export function f16ToF32(h: number): number {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export function f16ArrayToF32(src: ArrayLike<number>): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = f16ToF32(src[i]);
  return out;
}

const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
export function f32ToF16(v: number): number {
  f32[0] = v;
  const x = u32[0], sign = (x >>> 16) & 0x8000;
  let e = ((x >>> 23) & 0xff) - 127 + 15;
  const m = x & 0x7fffff;
  if (((x >>> 23) & 0xff) === 255) return sign | 0x7c00 | (m ? 0x200 : 0);
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    const mm = (m | 0x800000) >> (1 - e);
    return sign | ((mm + 0x1000) >> 13);
  }
  const r = sign | (e << 10) | (m >> 13);
  return r + ((m & 0x1000) ? 1 : 0); // round to nearest (carry propagates into exponent correctly)
}

export function f32ArrayToF16(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = f32ToF16(src[i]);
  return out;
}
