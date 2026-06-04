"""Data-derived typical (Uitz-style) relative vertical Chl-a profile.

The "base" shape rel_base(z; surface Chl) is the empirical median of the
surface-normalized profile rel(z) = Chl(z)/Chl_surf, binned by surface Chl
(trophic state). It is built only from RATIOS (rel), so the BGC-Argo
multiplicative fluorescence bias cancels (consistent with the relative-only
design). It already contains the typical deep-chlorophyll maximum per trophic
class and decays toward 0 at depth, so a model of the form

    rel_pred(z) = rel_base(z; C_surf) * a(z, env),   0 <= a <= A_max

forces rel -> 0 in the deep ocean by construction (bounded amplitude times a
decaying base) while leaving the subsurface free through a(z).

Reference: Uitz et al. (2006, JGR) -- trophic-state normalized profiles; here
instantiated from our own GLODAP+BGC-Argo data in absolute depth (no z/Ze
normalization, avoiding the Morel-Ze underestimate in productive water).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Surface-Chl class edges (mg/m3) and depth nodes (m); surface-dense to ~300 m.
DEFAULT_SURF_EDGES = np.array([0.0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.7, 1.2, 2.5, 1e3])
DEFAULT_DEPTH_GRID = np.array(
    [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130, 150, 175, 200, 250, 300],
    dtype=float)
MIN_COUNT = 20  # minimum rows in a (surf-bin, depth-node) cell to trust the median


@dataclass
class BaseProfile:
    surf_edges: np.ndarray            # (nb+1,) bin boundaries used to build the table
    depth_grid: np.ndarray            # (nd,)
    table: np.ndarray                 # (nb, nd) median rel; row 0 (surface) forced to 1
    counts: np.ndarray                # (nb, nd) sample counts
    surf_centers: np.ndarray | None = None  # (nb,) log10 surf at each bin; enables
    #                                         continuous (smooth) interpolation over
    #                                         surface Chl instead of nearest-bin

    def to_dict(self) -> dict:
        d = dict(surf_edges=self.surf_edges.tolist(), depth_grid=self.depth_grid.tolist(),
                 table=self.table.tolist(), counts=self.counts.tolist())
        if self.surf_centers is not None:
            d["surf_centers"] = self.surf_centers.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseProfile":
        sc = d.get("surf_centers")
        return cls(np.asarray(d["surf_edges"], float), np.asarray(d["depth_grid"], float),
                   np.asarray(d["table"], float), np.asarray(d["counts"], float),
                   None if sc is None else np.asarray(sc, float))

    def eval(self, surf_chl, depth) -> np.ndarray:
        """rel_base at each (surface Chl, depth). With `surf_centers` set, the base
        is interpolated SMOOTHLY over log10(surface Chl) and depth (no bin jumps);
        otherwise the nearest surf bin is used. 1 at the surface, 0 below the
        deepest grid node (hard deep->0) in both modes."""
        surf_chl = np.asarray(surf_chl, float); depth = np.asarray(depth, float)
        if self.surf_centers is not None:
            from scipy.interpolate import RegularGridInterpolator
            c = self.surf_centers
            interp = RegularGridInterpolator((c, self.depth_grid), self.table,
                                             bounds_error=False, fill_value=None)
            ls = np.clip(np.log10(np.clip(surf_chl, 1e-3, None)), c[0], c[-1])
            z = np.clip(depth, self.depth_grid[0], self.depth_grid[-1])
            out = interp(np.column_stack([ls, z]))
            out = np.where(depth > self.depth_grid[-1], 0.0, out)
            return np.clip(out, 0.0, None)
        nb = len(self.surf_edges) - 1
        sb = np.clip(np.digitize(surf_chl, self.surf_edges) - 1, 0, nb - 1)
        out = np.empty(len(depth), float)
        for b in range(nb):
            m = sb == b
            if m.any():
                out[m] = np.interp(depth[m], self.depth_grid, self.table[b],
                                   left=self.table[b, 0], right=0.0)
        return np.clip(out, 0.0, None)


def build_base_profile(surf_chl, depth, rel, *, surf_edges=DEFAULT_SURF_EDGES,
                       depth_grid=DEFAULT_DEPTH_GRID, min_count=MIN_COUNT,
                       quantile=False, n_bins=9, continuous=False) -> BaseProfile:
    """Median rel per (surface-Chl bin, nearest depth node), gap-filled in depth.

    `quantile=True` replaces the fixed surf_edges with equal-count (quantile)
    bins of the surface Chl (stable medians, finer where data is dense).
    `continuous=True` also stores per-bin log10 surf centers so BaseProfile.eval
    interpolates SMOOTHLY over surface Chl (no bin jumps).
    """
    surf_chl = np.asarray(surf_chl, float); depth = np.asarray(depth, float)
    rel = np.asarray(rel, float)
    if quantile:
        s = surf_chl[np.isfinite(surf_chl) & (surf_chl > 0)]
        edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n_bins + 1)))
        edges[0] = 0.0; edges[-1] = max(float(edges[-1]) * 10, 1e3)  # cover tails
        surf_edges = edges
    nb = len(surf_edges) - 1; nd = len(depth_grid)
    sb = np.clip(np.digitize(surf_chl, surf_edges) - 1, 0, nb - 1)
    mids = (depth_grid[:-1] + depth_grid[1:]) / 2.0
    di = np.digitize(depth, mids)  # nearest node index 0..nd-1
    table = np.full((nb, nd), np.nan); counts = np.zeros((nb, nd), int)
    for b in range(nb):
        for d in range(nd):
            cell = (sb == b) & (di == d) & np.isfinite(rel)
            counts[b, d] = int(cell.sum())
            if counts[b, d] >= min_count:
                table[b, d] = float(np.median(rel[cell]))
    # gap-fill each bin: linear interp over trusted nodes; surface=1; deep->0 tail
    for b in range(nb):
        ok = np.isfinite(table[b])
        if ok.sum() >= 2:
            table[b] = np.interp(depth_grid, depth_grid[ok], table[b, ok],
                                 left=1.0, right=table[b, ok][-1])
        else:
            table[b] = np.exp(-depth_grid / 50.0)  # fallback typical decay
        table[b, 0] = 1.0
    surf_centers = None
    if continuous:
        centers = np.full(nb, np.nan)
        for b in range(nb):
            s = surf_chl[(sb == b) & np.isfinite(surf_chl) & (surf_chl > 0)]
            if s.size:
                centers[b] = float(np.median(np.log10(s)))
        for b in range(nb):                                   # fill empty bins
            if not np.isfinite(centers[b]):
                lo = max(surf_edges[b], 1e-3); hi = min(surf_edges[b + 1], 1e3)
                centers[b] = 0.5 * (np.log10(lo) + np.log10(hi))
        for b in range(1, nb):                                # strictly increasing
            centers[b] = max(centers[b], centers[b - 1] + 1e-3)
        surf_centers = centers
    return BaseProfile(np.asarray(surf_edges, float), np.asarray(depth_grid, float),
                       table, counts, surf_centers)
