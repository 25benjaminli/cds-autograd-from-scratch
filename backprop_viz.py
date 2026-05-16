import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button
from vectorized_engine import Tensor

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
})

BG       = "#f5f5f0"
NODE_DEF = "#ffffff"
NODE_ACT = "#c8d8e8"
NODE_UPD = "#c8e6c8"
NODE_PAR = "#fdf6e3"
EDGE_COL = "#888888"

BOX_W, BOX_H = 1.8, 1.3
SHRINK = 28

FORMULAS = {
    "@":   r"$\nabla_X += \nabla_Y W^T,\quad \nabla_W += X^T \nabla_Y$",
    "+":   r"$\nabla_A += \nabla_Z,\ \ \nabla_B += \nabla_Z$",
    "neg": r"$\nabla_X += -\nabla_Y$",
    "sum": r"$\nabla_X += \mathbf{1} \cdot \nabla_Y$",
}
POW_FORMULA = r"$\nabla_X += n\,X^{n-1} \odot \nabla_Y$"

LABELS = {
    "x": "$x$", "W": "$W$", "b": "$b$", "y": "$y$",
    "xW": "$xW$", "y_hat": r"$\hat{y}$", "neg_yhat": r"$-\hat{y}$",
    "err": r"$y-\hat{y}$", "sq_err": r"$(y-\hat{y})^2$", "loss": r"$\mathcal{L}$",
}

LAYOUT = {
    "x":       (-4.5, 3.2), "W":       (-2.5, 3.2), "xW":      (-3.5, 1.6),
    "b":       ( 0.0, 3.2), "y_hat":   ( 0.0, 1.6), "neg_yhat":( 0.0, 0.0),
    "y":       ( 3.5, 3.2), "err":     ( 3.5, 1.6), "sq_err":  ( 3.5, 0.0),
    "loss":    ( 3.5,-1.6),
}


def build_graph():
    x     = Tensor(np.array([[1., 0., 1.]]),           _name="x")
    W     = Tensor(np.array([[1.,0.],[0.,1.],[1.,0.]]), _name="W")
    b     = Tensor(np.array([[0., 0.]]),                _name="b")
    y     = Tensor(np.array([[3., 1.]]),                _name="y")
    xW        = x @ W;      xW._name      = "xW"
    y_hat     = xW + b;     y_hat._name   = "y_hat"
    neg       = -y_hat;     neg._name     = "neg_yhat"
    err       = y + neg;    err._name     = "err"
    sq        = err ** 2;   sq._name      = "sq_err"
    loss      = sq.sum();   loss._name    = "loss"
    return loss, [W, b]


def topo_order(root):
    visited, order = set(), []
    def visit(n):
        if n in visited: return
        visited.add(n)
        for p in n._prev: visit(p)
        order.append(n)
    visit(root)
    return order


def fmt_arr(arr):
    flat = arr.flatten()
    if flat.size <= 6:
        if arr.ndim == 2 and arr.shape[0] > 1:
            rows = ["  ".join(f"{v:+.2f}" for v in r) for r in arr]
            return "\n".join(f"[{r}]" for r in rows)
        return "[" + "  ".join(f"{v:+.2f}" for v in flat) + "]"
    return f"[{flat[0]:+.2f} … {flat[-1]:+.2f}]  ({arr.shape})"


def build_steps(loss, params, lr):
    topo  = topo_order(loss)
    snap  = lambda: {n: (n.data.copy(), n.grad.copy()) for n in topo}
    steps = []

    for n in topo:
        steps.append(("Forward: " + n._name, {n}, set(), snap(), None))

    loss.grad = np.ones_like(loss.data)
    steps.append(("Backward init: $\\partial\\mathcal{L}/\\partial\\mathcal{L}=1$",
                  {loss}, {loss}, snap(),
                  r"$\partial\mathcal{L}/\partial\mathcal{L}=1$"))

    for n in reversed(topo):
        before = {x: x.grad.copy() for x in topo}
        n._backward()
        changed = {x for x in topo if not np.allclose(before[x], x.grad)}
        formula = POW_FORMULA if n._op.startswith("pow(") else FORMULAS.get(n._op, "")
        steps.append(("Backward: " + n._name, {n}, changed, snap(), formula))

    for p in params:
        old, gn = np.linalg.norm(p.data), np.linalg.norm(p.grad)
        p.data -= lr * p.grad
        steps.append((f"Update {p._name}: |p|={old:.3f}→{np.linalg.norm(p.data):.3f}  |g|={gn:.3f}",
                      {p}, {p}, snap(),
                      rf"$\theta \leftarrow \theta - {lr}\,\nabla_\theta$"))

    return topo, steps


class Viewer:
    def __init__(self, loss, params, lr=0.1):
        self.topo, self.steps = build_steps(loss, params, lr)
        self.params = set(params)
        self.idx    = 0

        by_name  = {n._name: n for n in self.topo if n._name}
        self.pos = {by_name[k]: v for k, v in LAYOUT.items() if k in by_name}
        self.edges = [(p, n) for n in self.topo for p in n._prev]

        self.fig = plt.figure(figsize=(14, 8.5))
        self.fig.patch.set_facecolor(BG)
        self.ax_title = self.fig.add_axes([0.02, 0.88, 0.96, 0.10])
        self.ax_title.set_facecolor(BG)
        self.ax_title.axis("off")
        self.ax = self.fig.add_axes([0.02, 0.18, 0.96, 0.68])
        self.ax.set_facecolor(BG)
        self.ax.axis("off")

        xs, ys = zip(*self.pos.values())
        self.ax.set_xlim(min(xs) - 1.4, max(xs) + 1.4)
        self.ax.set_ylim(min(ys) - 1.4, max(ys) + 0.8)

        self._build_artists()
        self._build_buttons()
        self._update()

    def _build_artists(self):
        self._edges = {}
        for src, dst in self.edges:
            if src not in self.pos or dst not in self.pos: continue
            x1, y1 = self.pos[src]
            x2, y2 = self.pos[dst]
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                linewidth=1.2, color=EDGE_COL, shrinkA=SHRINK, shrinkB=SHRINK,
                connectionstyle="arc3,rad=0.0", zorder=1, clip_on=False, visible=False)
            self.ax.add_patch(arrow)
            mx, my = 0.5*(x1+x2), 0.5*(y1+y2)
            lbl = self.ax.text(mx, my, dst._op or "input", ha="center", va="center",
                fontsize=7.5, color="#555",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, edgecolor="none"),
                zorder=2, visible=False)
            self._edges[(src, dst)] = (arrow, lbl)

        self._patches, self._name_texts, self._val_texts = {}, {}, {}
        for n, (x, y) in self.pos.items():
            self._patches[n] = self.ax.add_patch(mpatches.FancyBboxPatch(
                (x - BOX_W/2, y - BOX_H/2), BOX_W, BOX_H,
                boxstyle="round,pad=0.08", facecolor=NODE_DEF, edgecolor="#444444",
                linewidth=1.2, zorder=3, clip_on=False, visible=False))
            self._name_texts[n] = self.ax.text(
                x, y + BOX_H*0.22, LABELS.get(n._name, n._name or "?"),
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="#111", zorder=5, visible=False)
            self._val_texts[n] = self.ax.text(
                x, y - BOX_H*0.2, "", ha="center", va="center",
                fontsize=7.5, color="#333", family="monospace", zorder=5, visible=False)

        self._title_txt   = self.ax_title.text(0.5, 0.65, "", transform=self.ax_title.transAxes,
                                ha="center", va="top", fontsize=12, color="#222222")
        self._counter_txt = self.ax_title.text(0.5, 0.15, "", transform=self.ax_title.transAxes,
                                ha="center", va="bottom", fontsize=9, color="#666")
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        self._formula_txt = self.ax.text(
            xl[0] + 0.1, yl[0] + 0.1, "", ha="left", va="bottom",
            fontsize=13, color="#1a1a1a", zorder=6, visible=False,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#aaaaaa", linewidth=0.9))

    def _build_buttons(self):
        self._btns = []
        for ax_b, label, delta in [
            (self.fig.add_axes([0.38, 0.05, 0.10, 0.07]), "◀  Prev", -1),
            (self.fig.add_axes([0.52, 0.05, 0.10, 0.07]), "Next  ▶", +1),
        ]:
            btn = Button(ax_b, label, color="#dde3ea", hovercolor="#bcc8d8")
            btn.label.set_fontsize(11)
            btn.on_clicked(lambda _, d=delta: self._step(d))
            self._btns.append(btn)

    def _step(self, d):
        self.idx = max(0, min(self.idx + d, len(self.steps) - 1))
        self._update()

    def _update(self):
        title, active, updated, sp, formula = self.steps[self.idx]
        n_fwd    = len(self.topo)
        revealed = set(self.topo[:self.idx + 1]) if self.idx < n_fwd else set(self.topo)

        for (src, dst), (arrow, lbl) in self._edges.items():
            vis = src in revealed and dst in revealed
            arrow.set_visible(vis)
            lbl.set_visible(vis)

        for n in self.topo:
            if n not in self.pos: continue
            vis = n in revealed
            self._patches[n].set_visible(vis)
            self._name_texts[n].set_visible(vis)
            self._val_texts[n].set_visible(vis)
            if not vis: continue
            self._patches[n].set_facecolor(
                NODE_ACT if n in active else
                NODE_UPD if n in updated else
                NODE_PAR if n in self.params else NODE_DEF)
            d, g = sp[n]
            self._val_texts[n].set_text(f"v={fmt_arr(d)}\ng={fmt_arr(g)}")

        self._title_txt.set_text(title)
        self._counter_txt.set_text(f"step {self.idx+1} / {len(self.steps)}")
        self._formula_txt.set_text(formula or "")
        self._formula_txt.set_visible(bool(formula))
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    loss, params = build_graph()
    Viewer(loss, params, lr=0.1)
    plt.show()