"""Azimuth of the plate about its normal between frames, from the plate-plane rectified images (no IMU, no poses).
rectify(frame) -> RS x RS top-down image of the plate (cm/px = 2*R_CM/RS) from the ellipse pose; rel_angle(a, b) -> degrees (plate content of b = a rotated by +angle)."""
import numpy as np, cv2
import geom

R_CM = 14.0; RS = 128


def rectify(img, n, c, K):
    A, B = geom.plane_basis(n)
    # plane coords (x, y) in cm -> camera point c + x A + y B -> pixel
    M = K @ np.stack([A, B, c], 1)  # 3x3 maps (x, y, 1)
    s = 2 * R_CM / RS
    T = np.array([[s, 0, -R_CM], [0, s, -R_CM], [0, 0, 1.0]])  # rectified px -> plane cm
    Hm = M @ T  # rectified px -> image px
    return cv2.warpPerspective(img, Hm, (RS, RS), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def polar(im, nang=180, nr=40, r0=0.15, r1=0.95):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) if im.ndim == 3 else im.astype(np.float32)
    c = RS / 2
    th = np.linspace(0, 2 * np.pi, nang, endpoint=False); rr = np.linspace(r0, r1, nr) * c
    X = c + rr[None, :] * np.cos(th)[:, None]; Y = c + rr[None, :] * np.sin(th)[:, None]
    p = cv2.remap(g, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR)
    p = p - p.mean(0, keepdims=True); p /= (p.std() + 1e-6)
    return p


def rel_angle(ra, rb, nang=180):
    """angle (deg, in [-180, 180)) by which the content of rb is rotated relative to ra. Circular cross-correlation over the angular axis."""
    pa, pb = polar(ra, nang), polar(rb, nang)
    F = np.fft.fft(pa, axis=0) * np.conj(np.fft.fft(pb, axis=0))
    cc = np.fft.ifft(F, axis=0).real.sum(1)
    k = int(np.argmax(cc)); 
    # parabolic refinement
    y0, y1, y2 = cc[(k - 1) % nang], cc[k], cc[(k + 1) % nang]
    off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-9)
    sh = (k + off); sh = sh - nang if sh > nang / 2 else sh
    return float(-sh * 360.0 / nang), float(cc[k] / (cc.std() + 1e-9))
