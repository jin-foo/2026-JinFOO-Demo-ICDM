#!/usr/bin/env python3
"""Together-by-isolation schematic for the ICDM demo paper.

A 2-D *feature plane* (not a projection of Phi): isolation trees cut the plane;
leafmates are the points a query's leaves failed to exclude. Every quantity
drawn -- cut positions, leaf cells, path depths, shared-leaf counts, and both
background fields -- comes from one fitted ``sklearn.ensemble.IsolationForest``.

Palette follows the light booth tokens (paper / dust / amber query / jade leaves).

  uv run python figures/together_by_isolation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import IsolationForest

OUT = Path(__file__).resolve().parent
SEED = 42

PAPER = "#F1F3F4"
PANEL = "#FFFFFF"
HAIRLINE = "#CBD5DA"
DUST = "#8FA0A8"
STRUCTURE = "#5C6F79"
INK = "#0E1A20"
INK_DIM = "#42555F"
QUERY = "#C47500"
QUERY_INK = "#8F5300"
JADE = "#009878"
JADE_INK = "#006E58"
SIGNAL = "#1E7FE0"
VOID = PAPER  # savefig / vignette alias

N_TREES = 120
PSI = 256  # isolation-kernel subsample size; also caps tree depth at log2(PSI)
GRID = 320  # background-field resolution
CUT_DEPTH = 4  # drawn depth of the *context* mesh in (b); the tree itself is deeper
N_POCKET, N_SECOND, N_SAT, N_DUST = 230, 180, 70, 95


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def gaussian(rng, centre, sd, rot_deg, n) -> np.ndarray:
    """Rotated anisotropic Gaussian blob."""
    th = np.deg2rad(rot_deg)
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = rng.normal(0.0, 1.0, size=(n, 2)) * np.asarray(sd)
    return pts @ r.T + np.asarray(centre)


def make_cloud(rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Two dense pockets, one loose satellite, and uniform dust, plus the query index."""
    pocket = gaussian(rng, (0.30, 0.615), (0.075, 0.048), 28.0, N_POCKET)
    second = gaussian(rng, (0.715, 0.325), (0.058, 0.070), -18.0, N_SECOND)
    satellite = gaussian(rng, (0.60, 0.825), (0.078, 0.040), 8.0, N_SAT)
    dust = rng.uniform([0.03, 0.03], [0.97, 0.97], size=(N_DUST, 2))
    X = np.clip(np.vstack([pocket, second, satellite, dust]), 0.025, 0.975)

    # Query = deepest point of the left pocket.
    q = int(np.argmin(np.sum((X[:N_POCKET] - X[:N_POCKET].mean(0)) ** 2, axis=1)))
    return X, q


def pick_sparse(X: np.ndarray, depth: np.ndarray) -> int:
    """Fastest-isolating point in the upper-left, where the label has room."""
    room = (X[:, 0] < 0.40) & (X[:, 1] > 0.70)
    return int(np.argmin(np.where(room, depth, np.inf)))


# --------------------------------------------------------------------------
# forest readouts (all genuine)
# --------------------------------------------------------------------------


def mean_cut_depth(forest: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Mean number of splits traversed before each point lands in a leaf."""
    total = np.zeros(len(X))
    for est in forest.estimators_:
        total += np.asarray(est.decision_path(X).sum(axis=1)).ravel() - 1.0
    return total / len(forest.estimators_)


def shared_leaf_fraction(forest: IsolationForest, X: np.ndarray, q_xy: np.ndarray) -> np.ndarray:
    """Fraction of trees in which each row shares the query's leaf.

    This is the isolation-kernel similarity to the query: Jaccard on leaf sets,
    which for a single query reduces to the shared-leaf count over trees.
    """
    q = np.asarray(q_xy, dtype=np.float64).reshape(1, 2)
    hits = np.zeros(len(X))
    for est in forest.estimators_:
        hits += est.apply(X) == est.apply(q)[0]
    return hits / len(forest.estimators_)


def leaf_cell(tree, xy, xlim, ylim) -> tuple[float, float, float, float]:
    """Axis-aligned cell containing ``xy`` in this tree, clipped to the panel."""
    t = tree.tree_
    x0, x1 = xlim
    y0, y1 = ylim
    node = 0
    while t.feature[node] >= 0:
        feat = int(t.feature[node])
        thr = float(t.threshold[node])
        left = xy[feat] <= thr
        if feat == 0:
            x1, x0 = (min(x1, thr), x0) if left else (x1, max(x0, thr))
        else:
            y1, y0 = (min(y1, thr), y0) if left else (y1, max(y0, thr))
        node = int(t.children_left[node] if left else t.children_right[node])
    return x0, x1, y0, y1


def query_path(tree, xy) -> list[tuple[int, int, float, tuple[float, float, float, float]]]:
    """Root-to-leaf splits for ``xy``: (node, feature, threshold, cell before split).

    These are literally "the cuts that failed" -- every split the query survived
    while its leafmates survived with it.
    """
    t = tree.tree_
    node, x0, x1, y0, y1 = 0, 0.0, 1.0, 0.0, 1.0
    out = []
    while t.feature[node] >= 0:
        feat = int(t.feature[node])
        thr = float(t.threshold[node])
        out.append((node, feat, thr, (x0, x1, y0, y1)))
        left = xy[feat] <= thr
        if feat == 0:
            x1, x0 = (min(x1, thr), x0) if left else (x1, max(x0, thr))
        else:
            y1, y0 = (min(y1, thr), y0) if left else (y1, max(y0, thr))
        node = int(t.children_left[node] if left else t.children_right[node])
    return out


def pick_tree(forest: IsolationForest, X: np.ndarray, q: int) -> int:
    """First tree whose query cell is interior, near-square, and legibly populated.

    Deterministic and stated, so panel (b) shows a representative tree rather
    than a cherry-picked extreme. Falls back to the closest match if the
    admissible band is empty.
    """
    mates, ratio, interior = [], [], []
    for est in forest.estimators_:
        leaves = est.apply(X)
        mates.append(int((leaves == leaves[q]).sum()) - 1)
        x0, x1, y0, y1 = leaf_cell(est, X[q], (0.0, 1.0), (0.0, 1.0))
        ratio.append((x1 - x0) / max(y1 - y0, 1e-9))
        interior.append(x0 > 0.002 and x1 < 0.998 and y0 > 0.002 and y1 < 0.998)
    mates = np.asarray(mates)
    ratio = np.asarray(ratio)
    ok = np.asarray(interior) & (ratio > 0.4) & (ratio < 2.5) & (mates >= 30) & (mates <= 55)
    if ok.any():
        return int(np.flatnonzero(ok)[0])
    return int(np.argmin(np.abs(mates - 40)))


def grid_points(n: int = GRID) -> tuple[np.ndarray, tuple[int, int]]:
    """Flattened ``n x n`` sample of the unit plane, with the 2-D shape to fold back to."""
    g = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()]), xx.shape


# --------------------------------------------------------------------------
# drawing primitives
# --------------------------------------------------------------------------


def wash(ax, values, color, max_alpha, gamma=1.0, zorder=1.0):
    """Quiet continuous field: one hue, alpha carries the magnitude."""
    rgba = np.zeros(values.shape + (4,))
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = np.clip(values, 0.0, 1.0) ** gamma * max_alpha
    ax.imshow(
        rgba,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        interpolation="bilinear",
        aspect="auto",
        zorder=zorder,
        rasterized=True,
    )


def vignette(ax, strength=0.10):
    """Quiet corner settle. On paper this is a hairline-cool wash, not a blackout."""
    n = 192
    g = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(g, g)
    r = np.hypot(xx, yy) / np.sqrt(2.0)
    rgba = np.zeros((n, n, 4))
    rgba[..., :3] = to_rgb(STRUCTURE)
    rgba[..., 3] = np.clip((r - 0.55) / 0.50, 0.0, 1.0) ** 1.8 * strength
    ax.imshow(
        rgba,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        interpolation="bilinear",
        aspect="auto",
        zorder=7.5,
        rasterized=True,
    )


def glow(ax, xy, color, base=26.0, layers=5, peak=0.11, zorder=4.0):
    """Soft halo from stacked low-alpha scatters.

    Alpha falls faster than area grows, so the outermost ring has no visible
    edge -- a hard-edged disc is what makes a stacked-scatter glow look cheap.
    """
    for i in range(layers, 0, -1):
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=base * i**2.0,
            c=color,
            alpha=peak / i**1.9,
            linewidths=0,
            zorder=zorder,
            rasterized=True,
        )


def _segment(ax, feat, thr, cell, color, lw, alpha, zorder):
    """Draw one axis-aligned split, clipped to the cell it divides."""
    x0, x1, y0, y1 = cell
    if feat == 0:
        if not x0 < thr < x1:
            return
        ax.plot(
            [thr, thr],
            [y0, y1],
            color=color,
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=zorder,
        )
    else:
        if not y0 < thr < y1:
            return
        ax.plot(
            [x0, x1],
            [thr, thr],
            color=color,
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=zorder,
        )


def draw_context_cuts(ax, tree, skip: set[int], max_depth=CUT_DEPTH):
    """The tree's upper splits: a quiet scaffold, brightest at the root."""
    t = tree.tree_

    def walk(node, x0, x1, y0, y1, depth):
        feat = int(t.feature[node])
        if feat < 0 or depth >= max_depth:
            return
        thr = float(t.threshold[node])
        f = depth / max(1, max_depth - 1)
        if node not in skip:
            _segment(
                ax,
                feat,
                thr,
                (x0, x1, y0, y1),
                STRUCTURE,
                lw=1.15 * (1.0 - f) ** 1.1 + 0.30,
                alpha=0.52 * (1.0 - f) ** 1.2 + 0.09,
                zorder=2.0 + f,
            )
        if feat == 0:
            walk(int(t.children_left[node]), x0, thr, y0, y1, depth + 1)
            walk(int(t.children_right[node]), thr, x1, y0, y1, depth + 1)
        else:
            walk(int(t.children_left[node]), x0, x1, y0, thr, depth + 1)
            walk(int(t.children_right[node]), x0, x1, thr, y1, depth + 1)

    walk(0, 0.0, 1.0, 0.0, 1.0, 0)


def draw_query_cuts(ax, path):
    """The query's own root-to-leaf splits, tightening from structure into jade."""
    struct = np.array(to_rgb(STRUCTURE))
    jade = np.array(to_rgb(JADE))
    n = max(len(path) - 1, 1)
    for d, (_node, feat, thr, cell) in enumerate(path):
        f = d / n
        _segment(
            ax,
            feat,
            thr,
            cell,
            color=tuple(struct + (jade - struct) * f**1.6),
            lw=1.30 - 0.72 * f,
            alpha=0.42 + 0.40 * f,
            zorder=3.4 + f,
        )


def mark_query(ax, xy):
    """Filled amber disk plus an ink outline. The only warm mark in the frame."""
    glow(ax, xy.reshape(1, 2), QUERY, base=18.0, layers=4, peak=0.07, zorder=8.0)
    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=150,
        facecolors="none",
        edgecolors=QUERY,
        linewidths=0.9,
        alpha=0.95,
        zorder=9,
    )
    ax.scatter([xy[0]], [xy[1]], s=26, facecolors=QUERY, edgecolors=INK, linewidths=0.85, zorder=10)


def style_ax(ax, letter, title):
    """Square panel, no ticks, hairline spines, left-aligned letter and title above."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(HAIRLINE)
        spine.set_linewidth(0.7)
        spine.set_zorder(11)
    ax.text(
        0.0,
        1.036,
        letter,
        transform=ax.transAxes,
        color=INK,
        fontsize=8.6,
        fontweight="bold",
        va="baseline",
        ha="left",
    )
    ax.text(
        0.046,
        1.036,
        title,
        transform=ax.transAxes,
        color=INK,
        fontsize=8.4,
        va="baseline",
        ha="left",
    )


def note(ax, text, xy, xytext, color=INK_DIM, size=6.3, **kw):
    """Small annotation with a hairline leader, no arrowhead."""
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        color=color,
        fontsize=size,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.75, shrinkA=1.5, shrinkB=4.0),
        zorder=12,
        **kw,
    )


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------


def main() -> None:
    """Fit the forest once, then draw the three-act panel and write PDF + PNG."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": INK,
            "axes.linewidth": 0.7,
        }
    )

    X, q = make_cloud(np.random.default_rng(SEED))

    forest = IsolationForest(
        n_estimators=N_TREES,
        max_samples=PSI,
        random_state=SEED,
        n_jobs=1,
    ).fit(X)

    depth = mean_cut_depth(forest, X)
    share = shared_leaf_fraction(forest, X, X[q])
    sparse = pick_sparse(X, depth)
    gpts, gshape = grid_points()

    # (a) field: isolation resistance -- the forest's own normality score.
    # Lightly smoothed: the raw score is piecewise constant on tree cells, and
    # the staircase reads as noise at print size.
    resist = gaussian_filter(forest.score_samples(gpts).reshape(gshape), 4.5)
    lo, hi = np.quantile(resist, [0.55, 0.998])
    resist01 = np.clip((resist - lo) / (hi - lo), 0.0, 1.0)

    # (c) field: isolation-kernel similarity to the query, on the same grid.
    # Blocky and non-circular by construction -- this is a leaf-set overlap,
    # not a Euclidean radius.
    ksim = gaussian_filter(shared_leaf_fraction(forest, gpts, X[q]).reshape(gshape), 0.9)

    tree_i = pick_tree(forest, X, q)
    tree = forest.estimators_[tree_i]
    leaves = tree.apply(X)
    mates = leaves == leaves[q]
    mates[q] = False
    cell = leaf_cell(tree, X[q], (0.0, 1.0), (0.0, 1.0))
    path = query_path(tree, X[q])

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.02), dpi=400)
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.885, bottom=0.028, wspace=0.052)

    gx = np.linspace(0.0, 1.0, gshape[1])
    gy = np.linspace(0.0, 1.0, gshape[0])

    # ---------------------------------------------------------------- (a)
    ax = axes[0]
    style_ax(ax, "a", "Dense pockets resist isolation")
    wash(ax, resist01, SIGNAL, max_alpha=0.22, gamma=1.7, zorder=1.0)
    ax.contour(
        gx,
        gy,
        resist01,
        levels=[0.30, 0.62, 0.88],
        colors=SIGNAL,
        linewidths=0.38,
        alpha=0.28,
        zorder=1.4,
    )
    vignette(ax)
    ax.scatter(X[:, 0], X[:, 1], s=8.5, c=DUST, alpha=0.88, linewidths=0, zorder=3, rasterized=True)
    mark_query(ax, X[q])
    note(
        ax,
        f"dense pocket\n{depth[q]:.1f} cuts to isolate",
        xy=(X[q, 0] - 0.075, X[q, 1] - 0.055),
        xytext=(0.055, 0.30),
        ha="left",
        va="top",
    )
    note(
        ax,
        f"lone point\n{depth[sparse]:.1f} cuts",
        xy=X[sparse],
        xytext=(min(X[sparse, 0] + 0.075, 0.55), X[sparse, 1] - 0.075),
        ha="left",
        va="top",
    )
    ax.annotate(
        "query",
        xy=X[q],
        xytext=(X[q, 0] + 0.105, X[q, 1] + 0.125),
        color=QUERY_INK,
        fontsize=6.8,
        arrowprops=dict(
            arrowstyle="-", color=QUERY_INK, lw=0.55, alpha=0.95, shrinkA=1.5, shrinkB=6.5
        ),
        zorder=12,
    )

    # ---------------------------------------------------------------- (b)
    ax = axes[1]
    style_ax(ax, "b", "One tree: the cuts that failed")
    x0, x1, y0, y1 = cell
    ax.add_patch(
        Rectangle(
            (x0, y0), x1 - x0, y1 - y0, facecolor=JADE, edgecolor="none", alpha=0.12, zorder=1.2
        )
    )
    draw_context_cuts(ax, tree, skip={n for n, *_ in path})
    draw_query_cuts(ax, path)
    vignette(ax, strength=0.08)
    # Leafmates carry JADE_INK, not JADE: in greyscale print jade (~0.46 luma)
    # and dust (~0.62) are too close to separate at 8pt scatter size.
    ax.scatter(
        X[~mates, 0],
        X[~mates, 1],
        s=6.4,
        c=DUST,
        alpha=0.50,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )
    ax.scatter(
        X[mates, 0],
        X[mates, 1],
        s=8.8,
        c=JADE_INK,
        alpha=0.96,
        linewidths=0,
        zorder=6,
        rasterized=True,
    )
    for pad, alpha, lw in [(0.030, 0.14, 2.4), (0.014, 0.28, 1.4), (0.0, 0.95, 0.85)]:
        ax.add_patch(
            Rectangle(
                (x0 - pad, y0 - pad),
                (x1 - x0) + 2 * pad,
                (y1 - y0) + 2 * pad,
                facecolor="none",
                edgecolor=JADE,
                alpha=alpha,
                lw=lw,
                zorder=5.5,
            )
        )
    mark_query(ax, X[q])
    note(
        ax,
        f"the query's leaf\n{int(mates.sum())} points survive together",
        xy=(x1, y1),
        xytext=(min(x1 + 0.10, 0.60), min(y1 + 0.16, 0.94)),
        color=JADE_INK,
        ha="left",
        va="bottom",
    )

    # ---------------------------------------------------------------- (c)
    ax = axes[2]
    style_ax(ax, "c", "Many trees: leafmates assemble")
    # Floor the field so the low-similarity axis-aligned bands drop out and only
    # the query's actual neighbourhood carries the wash.
    ksim01 = np.clip((ksim - 0.20) / 0.62, 0.0, 1.0)
    wash(ax, ksim01, JADE, max_alpha=0.22, gamma=1.15, zorder=1.0)
    vignette(ax)
    order = np.argsort(share)
    f = share[order]
    # Monotone darkening dust -> jade -> jade-ink, so the ramp still reads as a
    # ramp when the page is printed in greyscale.
    cmap = mpl.colors.LinearSegmentedColormap.from_list("dust_jade", [DUST, JADE, JADE_INK])
    ax.scatter(
        X[order, 0],
        X[order, 1],
        s=5.0 + 32.0 * f**1.7,
        c=cmap(np.clip(f * 1.30, 0.0, 1.0) ** 0.8),
        alpha=np.clip(0.44 + 0.56 * f, 0.42, 1.0),
        linewidths=0,
        zorder=4,
        rasterized=True,
    )
    mark_query(ax, X[q])
    ax.text(
        0.042,
        0.048,
        f"size · tone  =  shared leaves with query, over {N_TREES} trees",
        color=INK_DIM,
        fontsize=6.0,
        ha="left",
        va="bottom",
        zorder=12,
    )

    pdf = OUT / "together-by-isolation.pdf"
    png = OUT / "together-by-isolation.png"
    # Rasterized layers (scatters and fields) bake at this dpi; cuts, the query
    # mark, and all type stay vector. CreationDate=None keeps the PDF byte-identical
    # across runs.
    fig.savefig(pdf, dpi=300, facecolor=PAPER, edgecolor=PAPER, metadata={"CreationDate": None})
    fig.savefig(png, dpi=320, facecolor=PAPER, edgecolor=PAPER)
    plt.close(fig)
    print(
        f"tree {tree_i} | leafmates {int(mates.sum())} | cell {cell} | "
        f"query depth {depth[q]:.2f} | sparse depth {depth[sparse]:.2f} | "
        f"share max(non-query) {np.sort(share)[-2]:.2f}"
    )
    print(f"wrote {pdf} and {png}")


if __name__ == "__main__":
    main()
