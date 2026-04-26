import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button
from vectorized_engine import Tensor

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
})

FORMULAS = {
    "@":   r"$\nabla_X \mathrel{+}= \nabla_Y W^\top,\quad \nabla_W \mathrel{+}= X^\top \nabla_Y$",
    "+":   r"$\nabla_A \mathrel{+}= \nabla_Z,\quad \nabla_B \mathrel{+}= \nabla_Z$",
    "neg": r"$\nabla_X \mathrel{+}= -\nabla_Y$",
    "sum": r"$\nabla_X \mathrel{+}= \mathbf{1}\,\nabla_Y$",
    "":    r"$\text{Leaf -- no backward rule}$",
}
POW_FORMULA = r"$\nabla_X \mathrel{+}= n\,X^{n-1} \odot \nabla_Y$"

LABELS = {
    "x": r"$x$", "W": r"$W$", "b": r"$b$", "y": r"$y$",
    "xW": r"$xW$", "y_hat": r"$\hat{y}$", "neg_yhat": r"$-\hat{y}$",
    "err": r"$y-\hat{y}$", "sq_err": r"$(y-\hat{y})^2$", "loss": r"$\mathcal{L}$",
}


def build_graph():
    x = Tensor(np.array([[1., 0., 1.]]), _name="x")
    W = Tensor(np.array([[1.,0.],[0.,1.],[1.,0.]]), _name="W")
    b = Tensor(np.array([[0., 0.]]), _name="b")
    y = Tensor(np.array([[3., 1.]]), _name="y")
    xW = x @ W
    xW._name = "xW"
    y_hat = xW + b
    y_hat._name= "y_hat"
    neg_yhat = -y_hat
    neg_yhat._name = "neg_yhat"
    err = y + neg_yhat
    err._name = "err"
    sq_err = err ** 2
    sq_err._name = "sq_err"
    loss = sq_err.sum()
    loss._name = "loss"
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


def auto_layout(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    by_name = {n._name: n for n in nodes if n._name is not None}
    pos = {}

    if "x" in by_name and "W" in by_name and "xW" in by_name:
        pos[by_name["x"]] = (-4.0, 3.2)
        pos[by_name["W"]] = (-2.0, 3.2)
        pos[by_name["xW"]] = (-3.0, 1.55)

    if "b" in by_name and "y_hat" in by_name:
        pos[by_name["b"]] = (0.0, 3.2)
        pos[by_name["y_hat"]] = (0.0, 1.55)

    if "neg_yhat" in by_name:
        pos[by_name["neg_yhat"]] = (0.0, -0.1)

    if "y" in by_name and "err" in by_name:
        pos[by_name["y"]] = (3.0, 3.2)
        pos[by_name["err"]] = (3.0, 1.55)

    if "sq_err" in by_name and "loss" in by_name:
        pos[by_name["sq_err"]] = (3.0, -0.1)
        pos[by_name["loss"]] = (3.0, -1.75)

    remaining = [n for n in nodes if n not in pos]
    if remaining:
        for n in nx.topological_sort(G):
            preds = list(G.predecessors(n))
            G.nodes[n]["layer"] = max((G.nodes[p]["layer"] for p in preds), default=-1) + 1
        raw = nx.multipartite_layout(G.subgraph(remaining), subset_key="layer", scale=1.4)
        xs, ys = [p[0] for p in raw.values()], [p[1] for p in raw.values()]

        def remap(v, s0, s1, d0, d1):
            return d0 if s0 == s1 else d0 + (v - s0) / (s1 - s0) * (d1 - d0)

        pos.update({
            n: (
                remap(p[0], min(xs), max(xs), -6.0, 4.0),
                remap(p[1], min(ys), max(ys), -3.6, 3.6),
            )
            for n, p in raw.items()
        })

    return pos


def snap(nodes):
    return {n: (n.data.copy(), n.grad.copy()) for n in nodes}


def build_steps(loss, params, lr):
    topo  = topo_order(loss)
    steps = []

    for n in topo:
        steps.append(("Forward: " + n._name, {n}, set(), snap(topo), None))

    loss.grad = np.ones_like(loss.data)
    steps.append(("Backward init: dloss/dloss=1", {loss}, {loss}, snap(topo),
                  r"$\partial\mathcal{L}/\partial\mathcal{L}=1$"))

    for n in reversed(topo):
        before = {x: x.grad.copy() for x in topo}
        n._backward()
        changed = {x for x in topo if not np.allclose(before[x], x.grad)}
        formula = POW_FORMULA if n._op.startswith("pow(") else FORMULAS.get(n._op, "")
        steps.append(("Backward: " + n._name, {n}, changed, snap(topo), formula))

    for p in params:
        old, gn = np.linalg.norm(p.data), np.linalg.norm(p.grad)
        p.data -= lr * p.grad
        steps.append((f"Update {p._name}: |p| {old:.3f}->{np.linalg.norm(p.data):.3f}, |g|={gn:.3f}",
                      {p}, {p}, snap(topo), rf"$\theta \leftarrow \theta - {lr}\,\nabla_\theta$"))

    return topo, steps


def mat_latex(arr):
    rows = [" & ".join(f"{v:.2f}" for v in r) for r in arr]
    return r"$\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}$"


class Viewer:
    COLORS = {"active": "#e8e8e8", "updated": "#d4edda", "param": "#fdf6e3", "default": "white"}

    def __init__(self, loss, params, lr=0.1):
        self.topo, self.steps = build_steps(loss, params, lr)
        self.edges = [(p, n) for n in self.topo for p in n._prev]
        self.pos = auto_layout(self.topo, self.edges)
        self.params = set(params)
        self.idx = 0

        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        self.fig.subplots_adjust(bottom=0.2)
        self._btn = []
        for label, rect, delta in [("Prev", [0.35,0.06,0.12,0.075], -1),
                                    ("Next", [0.53,0.06,0.12,0.075], +1)]:
            btn = Button(self.fig.add_axes(rect), label)
            btn.on_clicked(lambda _, d=delta: self._step(d))
            self._btn.append(btn)

        self.render()

    def _step(self, d):
        self.idx = max(0, min(self.idx + d, len(self.steps) - 1))
        self.render()

    def render(self):
        self.ax.clear()
        self.ax.axis("off")
        self.ax.set_facecolor("white")

        title, active, updated, sp, formula = self.steps[self.idx]

        xs, ys = zip(*self.pos.values())
        self.ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        self.ax.set_ylim(min(ys) - 1.8, max(ys) + 0.6)

        for src, dst in self.edges:
            x1, y1 = self.pos[src]
            x2, y2 = self.pos[dst]
            self.ax.add_patch(FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="->", mutation_scale=11, linewidth=1.2, color="#888888",
                shrinkA=18, shrinkB=18, connectionstyle="arc3,rad=0.0", clip_on=False))
            self.ax.text(0.5 * (x1 + x2), 0.5 * (y1 + y2), dst._op or "input",
                         ha="center", va="center", fontsize=8, color="#555",
                         bbox={"boxstyle":"round,pad=0.15","facecolor":"white","edgecolor":"none"})

        for n, (x, y) in self.pos.items():
            color = (self.COLORS["active"]  if n in active  else
                     self.COLORS["updated"] if n in updated else
                     self.COLORS["param"]   if n in self.params else
                     self.COLORS["default"])
            d, g = sp[n]
            self.ax.text(x, y,
                         f"{LABELS.get(n._name, n._name)}\nv={mat_latex(d)}\ng={mat_latex(g)}",
                         ha="center", va="center", fontsize=10,
                         bbox={"boxstyle":"round,pad=0.4","facecolor":color,
                               "edgecolor":"#1a1a1a","linewidth":0.8})

        self.ax.set_title(
            f"Step {self.idx+1}/{len(self.steps)} -- {title}\n"
            r"\text{grey=active, green=updated}", fontsize=11, pad=14)

        if formula:
            xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
            self.ax.text(xl[0]+0.05, yl[0]+0.05, formula, ha="left", va="bottom", fontsize=16,
                         bbox={"boxstyle":"round,pad=0.25","facecolor":"white",
                               "edgecolor":"#1a1a1a","linewidth":0.8})

        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    loss, params = build_graph()
    Viewer(loss, params, lr=0.1)
    plt.show()