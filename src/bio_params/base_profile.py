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
    surf_edges: np.ndarray   # (nb+1,)
    depth_grid: np.ndarray   # (nd,)
    table: np.ndarray        # (nb, nd) median rel; row 0 (surface) forced to 1
    counts: np.ndarray       # (nb, nd) sample counts

    def to_dict(self) -> dict:
        return dict(surf_edges=self.surf_edges.tolist(),
                    depth_grid=self.depth_grid.tolist(),
                    table=self.table.tolist(), counts=self.counts.tolist())

    @classmethod
    def from_dict(cls, d: dict) -> "BaseProfile":
        return cls(np.asarray(d["surf_edges"], float), np.asarray(d["depth_grid"], float),
                   np.asarray(d["table"], float), np.asarray(d["counts"], float))

    def eval(self, surf_chl, depth) -> np.ndarray:
        """rel_base at each (surface Chl, depth) row: nearest surf bin, linear in
        depth; 1 at the surface, 0 below the deepest grid node (hard deep->0)."""
        surf_chl = np.asarray(surf_chl, float); depth = np.asarray(depth, float)
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
                       depth_grid=DEFAULT_DEPTH_GRID, min_count=MIN_COUNT) -> BaseProfile:
    """Median rel per (surface-Chl bin, nearest depth node), gap-filled in depth."""
    surf_chl = np.asarray(surf_chl, float); depth = np.asarray(depth, float)
    rel = np.asarray(rel, float)
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
    return BaseProfile(np.asarray(surf_edges, float), np.asarray(depth_grid, float),
                       table, counts)
