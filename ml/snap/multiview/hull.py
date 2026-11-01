"""Visual-hull carving over the plate disc from several views with plate-ellipse poses + registered azimuths (rotation.py).
Voxel grid in the REFERENCE view's plate frame: x,y in [-R,R] (step VS cm), z in [0,ZMAX]. A voxel is kept when, in at least `vote` of the views in which it projects inside the
image, it lies in the (dilated) food silhouette. Returns volume (mL) and the occupancy array."""
import numpy as np, cv2
import geom

VS = 0.5; R = 14.0; ZMAX = 10.0
gx = np.arange(-R, R + 1e-6, VS); gz = np.arange(VS / 2, ZMAX, VS)
X, Y, Z = np.meshgrid(gx, gx, gz, indexing="ij")
PTS = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
DISC = (X ** 2 + Y ** 2 <= R ** 2).ravel()


def project(f, phi_deg):
    """pixel coords (N,2) and validity of PTS for view f whose plate frame is rotated by phi (content angle) relative to the reference."""
    t = np.radians(phi_deg); c, s = np.cos(t), np.sin(t)
    qx = c * PTS[:, 0] - s * PTS[:, 1]; qy = s * PTS[:, 0] + c * PTS[:, 1]
    A, B = geom.plane_basis(f.n)
    P = f.c[None, :] + qx[:, None] * A[None, :] + qy[:, None] * B[None, :] + PTS[:, 2:3] * f.n[None, :]
    uv = (f.K @ P.T).T
    ok = uv[:, 2] > 1
    uv = uv[:, :2] / np.maximum(uv[:, 2:], 1e-6)
    return uv, ok


def carve(views, vote=0.75, dil=2, min_views=2):
    """views: list of (Frame, phi_deg, food_silhouette bool HxW). -> volume_ml, occ (bool N), nviews (N)"""
    n = len(PTS); seen = np.zeros(n, int); inside = np.zeros(n, int)
    for f, phi, sil in views:
        S = cv2.dilate(sil.astype(np.uint8), np.ones((2 * dil + 1, 2 * dil + 1), np.uint8)) if dil else sil.astype(np.uint8)
        uv, ok = project(f, phi)
        h, w = S.shape
        u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
        inimg = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        val = np.zeros(n, bool); val[inimg] = S[v[inimg], u[inimg]] > 0
        seen += inimg; inside += val
    occ = DISC & (seen >= min_views) & (inside >= vote * np.maximum(seen, 1))
    return float(occ.sum() * VS ** 3), occ, seen
