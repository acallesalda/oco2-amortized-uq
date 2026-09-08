#!/usr/bin/env python3
"""why_profiles.py -- why the amortized retrieval beats ACOS on XCO2 and loses near
the surface.

    python3 experiments/scoring/why_profiles.py \
        --profiles experiments/scoring/profiles_bo.npz \
        --xco2     experiments/scoring/xco2_bo.npz \
        --last4    experiments/scoring/last4_bo.npz \
        --split-csv group_split_v1_cov.csv \
        --out experiments/scoring/tables --fig latex/figures

Like `score.py` and `figures.py`, this needs nothing but the artifacts: no GPU, no
checkpoint, no spectra. Everything it prints is recomputed from `y_std`, the three
predictives and the split table, so any number quoted in the revision can be regenerated
in twenty seconds.

THE QUESTION. On XCO2 the normalizing flow beats the operational retrieval by 27 %
(2.11 against 2.87 ppm). On the near-surface layers it loses by 17 % (12.48 against
10.66). Both numbers come from the same models, the same soundings and the same split,
so one of them is not about the models at all -- and it is worth knowing which.

THE ANSWER, in the order the sections below establish it.

  1  Per level, the flow wins at 16 of the 20 levels, ties at level 3, and loses only at
     18, 19 and 20 -- where the deficit grows monotonically toward the surface.
  2  86 % of the truth's near-surface variance is WITHIN a reference sounding: for fixed
     geometry and surface, the 100 replicates differ by 17.8 ppm at level 20 and only
     7.1 ppm between scenes. The near-surface task is almost entirely a per-sounding
     question, where XCO2 is largely a per-scene one.
  3  That within-scene near-surface variation is orthogonal to the column: the truth's
     level-20 deviation correlates -0.05 with the truth's column deviation. Being
     excellent at XCO2 buys nothing at level 20 -- they are separate pieces of
     information, the retrieval's first and second degrees of freedom.
  4  In the natural basis, the leading EOF of the within-sounding truth anomaly carries
     80 % of that variance, is essentially orthogonal to the pressure weighting, and is a
     boundary-layer/free-troposphere seesaw at constant column: the second of the roughly
     two CO2 degrees of freedom. ACOS recovers it at r = 0.70 and the flow's posterior
     mean at r = 0.03.
  5  But EVERY method reads the individual spectrum -- what changes with height is how
     much usable information each extracts, and the ordering REVERSES. Switch each
     method's per-sounding response off (replace it by its own scene mean) and: on XCO2
     the learned models lose 9-10 % while ACOS *gains* 4.7 %, because ACOS over-responds
     by a factor of two relative to its own skill; at level 20 ACOS loses 17.7 %, the
     deterministic network 3.1 % and the flow's mean 0 %. So the near-surface deficit is
     under-extraction plus heavy shrinkage, not blindness, and the XCO2 advantage is
     mostly the shrinkage that a per-sounding optimal estimation cannot perform.
  6  Against the floor -- the best any predictor that knows only the scene can do -- the
     flow is at 1.06x near the surface and ACOS at 0.91x, i.e. below it; on XCO2 the flow
     is at 1.16x and ACOS at 1.58x. And this is not a calibration failure: the flow's
     near-surface predictive is calibrated (z-RMS 1.04, 94 % coverage at a nominal 95)
     where ACOS's intervals are 25-35 % too wide.
  7  Clusters 8 and 9 are a second, separable effect: per-cluster error tracks the number
     of training reference soundings for every learned method and for none of ACOS's, and
     the two smallest clusters carry 66 % of the near-surface gap and all of the 20-level
     gap. Removing them flips the 20-level task to a flow win and leaves 7 % of the
     near-surface deficit standing -- which is the part sections 3-6 explain.
  8  Optional (--group-by): the same breakdown by any covariate rather than by cluster.
     Built for the March and July artifacts, where SOM labels do not carry across months
     and the regime has to be named by surface pressure or airmass instead.

Sections 1-6 are a property of the method; section 7 is a property of this training set.
They want different remedies and the revision should not conflate them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True,
    "grid.alpha": 0.25,
})
_LAYOUT = {"layout": "constrained"}
COLORS = {"ACOS": "#d95f02", "Laplace": "#1b9e77", "flow": "#7570b3", "truth": "#444444"}

_ID = re.compile(r"clst(\d+)_ref(\d+)_rep(\d+)")

# The XCO2 pressure weighting function, RetrievalResults/xco2_pressure_weighting_function.
# It varies by 1.3e-4 across a scene's 100 replicates, i.e. not at all at the precision
# anything here needs, so it is taken as a constant. Levels 1 and 20 are half-layers,
# which is why the surface level carries only 2.6 % of the column.
H = np.full(20, 0.0527)
H[0], H[19] = 0.0265, 0.0261
H = H / H.sum()


# ---------------------------------------------------------------------------------
class Art:
    """One scored artifact, un-standardized to ppm, with scene labels attached."""

    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        meta = json.loads(bytes(d["__meta__"]).decode("utf-8"))
        mu = np.asarray(meta["y_mu"])
        sd = np.asarray(meta["y_sigma"])
        scale = float(meta["ppm_scale"])

        def un(a):
            return (a * sd + mu) * scale

        self.path = path
        self.meta = meta
        self.ids = d["ids"]
        parts = [_ID.match(i).groups() for i in self.ids]
        self.clst = np.array([int(a) for a, _, _ in parts])
        self.ref = np.array([int(b) for _, b, _ in parts])
        self.scene = np.array([f"{a}/{b}" for a, b, _ in parts])
        self.y = un(d["y_std"])
        self.flow = un(d["nf_samples_std"].mean(axis=1))
        self.lap = un(d["lap_mu_std"])
        self.acos = un(d["acos_mu_std"])
        self.samples = (d["nf_samples_std"] * sd + mu) * scale
        self.flow_sd = d["nf_samples_std"].std(axis=1) * sd * scale
        self.acos_sd = np.sqrt(np.einsum("nii->ni", d["acos_cov_std"])) * sd * scale
        self.lap_sd = np.sqrt(np.einsum("nii->ni", d["lap_cov_std"])) * sd * scale
        self.levels = list(meta.get("levels") or [])

        # Whatever per-sounding covariates predict.py chose to store, for --group-by.
        # Taken generically rather than from a fixed list: the month artifacts carry more
        # than the February ones (dof_co2 and mixture_component above all), and a hardcoded
        # list would silently drop exactly the column this analysis wants.
        _not_aux = {"y_std", "lap_mu_std", "lap_cov_epi_std", "nf_samples_std",
                    "nf_logprob_std", "nf_latent_std", "y_retrieved_std", "lap_cov_std",
                    "lap_aleatoric_var_std", "acos_mu_std", "acos_cov_std", "acos_valid",
                    "ids", "__meta__"}
        self.aux = {k: d[k] for k in d.files
                    if k not in _not_aux and np.asarray(d[k]).shape == (len(self.ids),)}

        # Non-physical rows. July 2020 carries one sounding, clst0002_ref0011_rep0010,
        # whose ACOS retrieved state is NaN at level 20 and 65-253 ppm at levels 17-19 --
        # a failed retrieval rather than the -1e19 fill sentinel. One row in
        # 44,327 is enough to make every pooled ACOS statistic NaN, so it is screened here
        # rather than left to poison a table. The mask is the INTERSECTION over methods, so
        # all four columns of a comparison are computed on identical soundings.
        ok = np.ones(len(self.ids), dtype=bool)
        for name, M in (("truth", self.y), ("flow", self.flow), ("Laplace", self.lap),
                        ("ACOS", self.acos)):
            bad = ~np.isfinite(M).all(axis=1) | (np.abs(M) > 1e4).any(axis=1)
            if bad.any():
                print("    screened %d non-physical %s row(s): %s"
                      % (bad.sum(), name, ", ".join(map(str, self.ids[bad][:3]))))
            ok &= ~bad
        self.ok = ok
        if not ok.all():
            print("    %d of %d soundings retained (%.4f %%)"
                  % (ok.sum(), len(ok), 100 * ok.mean()))
            for attr in ("y", "flow", "lap", "acos", "samples", "flow_sd", "acos_sd",
                         "lap_sd"):
                setattr(self, attr, getattr(self, attr)[ok])
            self.ids = self.ids[ok]
            self.clst = self.clst[ok]
            self.ref = self.ref[ok]
            self.scene = self.scene[ok]
            self.aux = {k: v[ok] for k, v in self.aux.items()}

        self.scenes = np.unique(self.scene)
        self._idx = {s: np.where(self.scene == s)[0] for s in self.scenes}

    def level_of(self, k):
        """The pressure-level label of output column k, from the artifact's own metadata.

        A last4 artifact's four columns are levels 17-20, not 1-4; without this every
        table built from one would be silently mislabelled."""
        return self.levels[k] if k < len(self.levels) else k + 1

    def methods(self):
        return (("flow", self.flow), ("Laplace", self.lap), ("ACOS", self.acos))

    def dev(self, M):
        """Deviation from the scene mean: the part of a quantity that a fixed reference
        sounding does not explain. Accepts 1-D or 2-D."""
        M = np.asarray(M, dtype=float)
        out = np.empty_like(M)
        for s in self.scenes:
            i = self._idx[s]
            out[i] = M[i] - M[i].mean(axis=0)
        return out

    def scene_mean(self, M):
        M = np.asarray(M, dtype=float)
        out = np.empty_like(M)
        for s in self.scenes:
            i = self._idx[s]
            out[i] = M[i].mean(axis=0)
        return out


def align(arts):
    """Restrict every artifact to the soundings all of them retained.

    Screening happens per artifact, and a sounding whose ACOS profile is unusable can have
    a perfectly good ACOS column -- July's failed retrieval is exactly that. Without this
    the XCO2 and near-surface artifacts end up with different row counts and every
    cross-task statement in sections 5 and 6 is comparing two different populations.
    """
    live = [a for a in arts if a is not None]
    if len(live) < 2:
        return
    common = set(live[0].ids)
    for a in live[1:]:
        common &= set(a.ids)
    for a in live:
        if len(a.ids) == len(common):
            continue
        keep = np.array([i in common for i in a.ids])
        print("    aligned %s to the %d soundings every artifact retained (dropped %d)"
              % (Path(a.path).name, keep.sum(), (~keep).sum()))
        for attr in ("y", "flow", "lap", "acos", "samples", "flow_sd", "acos_sd", "lap_sd"):
            setattr(a, attr, getattr(a, attr)[keep])
        a.ids = a.ids[keep]
        a.clst = a.clst[keep]
        a.ref = a.ref[keep]
        a.scene = a.scene[keep]
        a.aux = {k: v[keep] for k, v in a.aux.items()}
        a.scenes = np.unique(a.scene)
        a._idx = {s: np.where(a.scene == s)[0] for s in a.scenes}


def rmse(a, b, mask=None):
    e = np.asarray(a) - np.asarray(b)
    if mask is not None:
        e = e[mask]
    return float(np.sqrt(np.mean(e ** 2)))


def corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _emit(df, name, out_dir, floats="%.4f"):
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name + ".csv")
    df.to_csv(path, index=False, float_format=floats)
    print("    wrote " + str(path))


# ---------------------------------------------------------------------------------
def section_1_per_level(pro, out_dir):
    print("=" * 100)
    print("1. PER-LEVEL RMSE [ppm].  Level 20 is nearest the surface.")
    print("=" * 100)
    rows = []
    for k in range(pro.y.shape[1]):
        rows.append(dict(
            level=pro.level_of(k),
            truth_sd=float(pro.y[:, k].std()),
            flow=rmse(pro.flow[:, k], pro.y[:, k]),
            Laplace=rmse(pro.lap[:, k], pro.y[:, k]),
            ACOS=rmse(pro.acos[:, k], pro.y[:, k]),
        ))
    t = pd.DataFrame(rows)
    t["flow_minus_ACOS"] = t.flow - t.ACOS
    t["flow_over_ACOS"] = t.flow / t.ACOS
    print(t.round(3).to_string(index=False))
    win = t[t.flow_over_ACOS < 1].level.tolist()
    lost = sorted(set(t.level) - set(win))
    print()
    print("    the flow wins at %d of %d levels; it loses at %s, and the deficit grows"
          % (len(win), len(t), lost))
    print("    monotonically toward the surface: "
          + ", ".join("L%d %.3f" % (r.level, r.flow_over_ACOS)
                      for r in t.tail(3).itertuples()))
    _emit(t, "why_profiles_per_level", out_dir)
    return t


def section_2_variance(pro, out_dir):
    print()
    print("=" * 100)
    print("2. WHERE THE TRUTH'S VARIANCE LIVES: between reference soundings, or within one")
    print("=" * 100)
    rows = []
    for k in range(pro.y.shape[1]):
        y = pro.y[:, k]
        between = pro.scene_mean(y)
        within = pro.dev(y)
        rows.append(dict(level=pro.level_of(k), truth_sd=float(y.std()),
                         sd_between=float(between.std()),
                         sd_within=float(np.sqrt(np.mean(within ** 2))),
                         pct_within=100 * np.mean(within ** 2)
                         / np.mean((y - y.mean()) ** 2)))
    t = pd.DataFrame(rows)
    print(t.round(2).to_string(index=False))
    print()
    print("    at level 20 the within-scene share is %.0f %%: for one fixed reference"
          % t.pct_within.iloc[-1])
    print("    sounding the 100 replicates differ near the surface by %.1f ppm, against"
          % t.sd_within.iloc[-1])
    print("    %.1f ppm between scenes." % t.sd_between.iloc[-1])
    _emit(t, "why_profiles_variance", out_dir)
    return t


def section_3_orthogonality(pro, xc, out_dir):
    print()
    print("=" * 100)
    print("3. IS THE NEAR-SURFACE SIGNAL IN THE COLUMN?  All correlations are within-scene.")
    print("=" * 100)
    if pro.y.shape[1] != 20:
        print("    (skipped: this artifact has %d targets, and the column functional needs"
              " all 20)" % pro.y.shape[1])
        return None
    dcol_truth = pro.dev(pro.y @ H)
    rows = []
    for k in range(20):
        dk = pro.dev(pro.y[:, k])
        rows.append(dict(level=k + 1,
                         truth_level_vs_truth_column=corr(dk, dcol_truth),
                         flow_vs_truth=corr(pro.dev(pro.flow[:, k]), dk),
                         Laplace_vs_truth=corr(pro.dev(pro.lap[:, k]), dk),
                         ACOS_vs_truth=corr(pro.dev(pro.acos[:, k]), dk)))
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    print()
    print("    level 20's within-scene variation is orthogonal to the column (r = %+.3f),"
          % t.truth_level_vs_truth_column.iloc[-1])
    print("    so column skill does not transfer to it. ACOS reads it at r = %.3f; the"
          % t.ACOS_vs_truth.iloc[-1])
    print("    flow at %.3f." % t.flow_vs_truth.iloc[-1])
    if xc is not None and (xc.ids == pro.ids).all():
        print("    For reference, on the dedicated XCO2 task the flow's within-scene")
        print("    correlation with the truth column is %.3f and ACOS's is %.3f."
              % (corr(xc.dev(xc.flow[:, 0]), xc.dev(xc.y[:, 0])),
                 corr(xc.dev(xc.acos[:, 0]), xc.dev(xc.y[:, 0]))))
    _emit(t, "why_profiles_within_scene", out_dir)
    return t


def section_4_eofs(pro, out_dir, n_modes=4):
    print()
    print("=" * 100)
    print("4. THE TWO DEGREES OF FREEDOM.  EOFs of the within-scene truth profile anomaly.")
    print("=" * 100)
    if pro.y.shape[1] != 20:
        print("    (skipped: this artifact has %d targets, and the EOF basis is defined on"
              " the full 20-level profile)" % pro.y.shape[1])
        return None, None, None
    A = pro.dev(pro.y)
    w, V = np.linalg.eigh(np.cov(A.T))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    hn = H / np.linalg.norm(H)
    print("    variance carried: "
          + ", ".join("EOF%d %.1f %%" % (m + 1, 100 * w[m] / w.sum())
                      for m in range(5)))
    print("    |cos angle| between EOF1 and the pressure-weighting direction: %.3f"
          % abs(V[:, 0] @ hn))
    print("      -- i.e. EOF1 is a constant-column mode, invisible to XCO2.")
    print("    EOF1 loadings, level 1..20:")
    print("      " + np.array2string(np.round(V[:, 0], 3), max_line_width=200))

    rows = []
    for name, P in pro.methods():
        D = pro.dev(P)
        rec = {"method": name}
        a, p = A @ H, D @ H
        rec["column_r"] = corr(p, a)
        rec["column_amp"] = float(p.std() / a.std())
        for m in range(n_modes):
            a, p = A @ V[:, m], D @ V[:, m]
            rec["EOF%d_r" % (m + 1)] = corr(p, a)
            rec["EOF%d_amp" % (m + 1)] = float(p.std() / a.std())
        rows.append(rec)
    t = pd.DataFrame(rows)
    print()
    print(t.round(3).to_string(index=False))
    print()
    print("    `amp` is the ratio of the method's within-scene spread to the truth's. The")
    print("    flow shrinks EOF1 to a twelfth of its amplitude, which is the MSE-optimal")
    print("    thing to do with a direction you cannot read; ACOS reproduces it at full")
    print("    amplitude and 0.70 correlation.")
    _emit(t, "why_profiles_eof_skill", out_dir)
    loadings = {"level": np.arange(1, 21)}
    for m in range(n_modes):
        loadings["EOF%d" % (m + 1)] = V[:, m]
    loadings["var_share"] = np.r_[w[:n_modes] / w.sum(),
                                  np.full(20 - n_modes, np.nan)]
    _emit(pd.DataFrame(loadings), "why_profiles_eof_loadings", out_dir)
    return t, V, w


def section_5_response(pro, xc, l4, out_dir):
    """What does reading THIS spectrum, rather than knowing which scene it is, buy?

    This is the section that keeps the diagnosis honest. A small within-scene correlation
    is easy to over-read as "the network ignores the spectrum". The test here is a
    self-comparison: replace every sounding's prediction by that method's own scene-mean
    prediction and see how much worse the method gets. What is left is exactly the value
    of the per-sounding response, in the method's own units, with no reference to any
    other method.
    """
    print()
    print("=" * 108)
    print("5. WHAT DOES READING THIS PARTICULAR SPECTRUM BUY EACH METHOD?")
    print("   Each method against ITSELF with the per-sounding response switched off --")
    print("   every sounding replaced by that method's own scene-mean prediction.")
    print("   `gain` is the RMSE reduction that reading the individual spectrum earns.")
    print("=" * 108)

    def gain_table(art, cols, labels, task):
        rows = []
        for j, lab in zip(cols, labels):
            y = art.y[:, j]
            rec = {"task": task, "target": lab}
            for m, P in art.methods():
                flat = rmse(art.scene_mean(P)[:, j], y)
                full = rmse(P[:, j], y)
                d_t = art.dev(y)
                d_p = art.dev(P[:, j])
                a = float(np.sqrt(np.mean(d_p ** 2)) / np.sqrt(np.mean(d_t ** 2)))
                r = corr(d_p, d_t)
                rec[m + "_gain%"] = 100 * (flat - full) / flat
                rec[m + "_amp"] = a
                rec[m + "_r"] = r
                # what it would reach if that same response were optimally rescaled
                rec[m + "_best%"] = 100 * (1 - np.sqrt(max(0.0, 1 - r ** 2)))
            rows.append(rec)
        return pd.DataFrame(rows)

    if pro.y.shape[1] == 20:
        cols = [0, 4, 9, 14, 16, 17, 18, 19]
    else:
        cols = list(range(pro.y.shape[1]))
    t = gain_table(pro, cols, ["L%d" % pro.level_of(c) for c in cols],
                   "%d-level" % pro.y.shape[1])
    parts = [t]
    if xc is not None:
        parts.append(gain_table(xc, [0], ["XCO2"], "XCO2"))
    if l4 is not None:
        parts.append(gain_table(l4, [0, 1, 2, 3], ["L17", "L18", "L19", "L20"], "last4"))
    full = pd.concat(parts, ignore_index=True)

    show = ["task", "target"] + [m + s for m in ("flow", "Laplace", "ACOS")
                                 for s in ("_gain%", "_r", "_amp", "_best%")]
    print(full[show].round(2).to_string(index=False))
    print()
    print("    gain%   RMSE reduction the per-sounding response actually earns")
    print("    r       correlation of that response with the truth's per-sounding deviation")
    print("    amp     size of that response as a fraction of the truth's")
    print("    best%   the reduction the SAME response would earn if optimally rescaled,")
    print("            i.e. 1 - sqrt(1 - r^2): the ceiling on what this method extracts")
    print()
    # Read the conclusions off the table rather than reciting February's. Running this on
    # a March or July artifact and printing February's numbers underneath it is exactly the
    # kind of error this whole section exists to avoid.
    def row(task, target):
        m = full[(full.task == task) & (full.target == target)]
        return None if m.empty else m.iloc[0]

    col = row("XCO2", "XCO2")
    surf_task = "last4" if l4 is not None else full.task.iloc[0]
    srf = row(surf_task, "L20")
    if srf is None:
        srf = row(surf_task, full[full.task == surf_task].target.iloc[-1])

    print("    Read off the table above, and note that none of it says 'the network")
    print("    ignores the spectrum':")
    if col is not None:
        print("      * On XCO2 the flow gains %+.1f %% and the deterministic network %+.1f %%"
              % (col["flow_gain%"], col["Laplace_gain%"]))
        print("        from the individual spectrum; ACOS gains %+.1f %%, with a per-sounding"
              % col["ACOS_gain%"])
        print("        response %.2fx the truth's amplitude against a skill of r = %.2f --"
              % (col["ACOS_amp"], col["ACOS_r"]))
        print("        over-responding by a factor of %.1f, so reading the spectrum leaves it"
              % (col["ACOS_amp"] / max(col["ACOS_r"], 1e-6)))
        print("        worse than its own scene mean.")
    if srf is not None:
        print("      * At the surface the ordering flips: ACOS %+.1f %%, the deterministic"
              % srf["ACOS_gain%"])
        print("        network %+.1f %%, the flow %+.1f %%. Their skills there are r = %.2f,"
              % (srf["Laplace_gain%"], srf["flow_gain%"], srf["ACOS_r"]))
        print("        %.2f and %.2f, and the two learned responses run at %.2f and %.2f of"
              % (srf["Laplace_r"], srf["flow_r"], srf["Laplace_amp"], srf["flow_amp"]))
        print("        the truth's amplitude.")
        if srf["Laplace_gain%"] < 0 and srf["Laplace_amp"] > 0.2:
            print("        NOTE: the deterministic network's surface response is NEGATIVE")
            print("        here -- it keeps responding at %.0f %% amplitude while its skill has"
                  % (100 * srf["Laplace_amp"]))
            print("        fallen to r = %.2f, so what it reads is noise and it would do"
                  % srf["Laplace_r"])
            print("        better to ignore the spectrum. That is the signature of a")
            print("        per-sounding response that has not transferred.")
    print("      * `best%` minus `gain%` is what each method is throwing away by responding")
    print("        at the wrong amplitude. It is largest for ACOS everywhere, which is the")
    print("        mechanism of the amortized advantage: learning the right amount of")
    print("        shrinkage from a training population is what a per-sounding optimal")
    print("        estimation cannot do. The same arithmetic says an ORACLE rescaling of")
    print("        ACOS's own response would beat the flow on XCO2, so state the mechanism")
    print("        in the paper rather than leaving a reader to find the oracle number.")
    _emit(full, "why_profiles_spectral_response", out_dir)
    return full


def section_6_floor(pro, xc, l4, out_dir):
    print()
    print("=" * 100)
    print("6. THE FLOOR: the best RMSE any predictor that knows only the SCENE can reach,")
    print("   and each method against it.  Below 1.00 means genuine per-sounding skill.")
    print("=" * 100)
    rows = []
    dpro = list(range(pro.y.shape[1]))
    tasks = [("XCO2", xc, [0]),
             ("profile (%d level)" % pro.y.shape[1], pro, dpro),
             ("near surface (17-20)", l4, [0, 1, 2, 3])]
    for name, art, ks in tasks:
        if art is None:
            continue
        Y = art.y[:, ks]
        floor = rmse(art.scene_mean(Y), Y)
        rec = {"task": name, "scene_oracle_floor": floor}
        for m, P in art.methods():
            rec[m] = rmse(P[:, ks], Y)
            rec[m + "_over_floor"] = rec[m] / floor
        rows.append(rec)
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))

    print()
    print("   The same thing from the error side: MSE = bias^2 + var(scene offsets) +")
    print("   within-scene MSE.  Amortization removes the first two and not the third.")
    rows = []
    # The surface level, wherever it sits in this artifact's output vector: column 19 of a
    # 20-level model, column 3 of a last4 one. Taking it positionally would silently report
    # level 4 as if it were the surface.
    surf = pro.levels.index(20) if 20 in pro.levels else pro.y.shape[1] - 1
    budget = [("XCO2", xc, 0), ("profile L%d (surface)" % pro.level_of(surf), pro, surf)]
    for name, art, k in budget:
        if art is None or k >= art.y.shape[1]:
            continue
        for m, P in art.methods():
            e = P[:, k] - art.y[:, k]
            offs = np.array([e[art._idx[s]].mean() for s in art.scenes])
            wi = float(np.sqrt(np.mean(art.dev(e) ** 2)))
            tot = float(np.sqrt(np.mean(e ** 2)))
            rows.append(dict(task=name, method=m, bias=float(e.mean()),
                             sd_scene_offset=float(offs.std()), within_rms=wi, RMSE=tot,
                             pct_bias=100 * e.mean() ** 2 / tot ** 2,
                             pct_offset=100 * offs.var() / tot ** 2,
                             pct_within=100 * wi ** 2 / tot ** 2))
    b = pd.DataFrame(rows)
    print(b.round(2).to_string(index=False))
    print()
    ac = b[(b.method == "ACOS")]
    if len(ac) >= 1:
        for _, r in ac.iterrows():
            print("    ACOS on %s: %.0f %% of its mean squared error is a %+.1f ppm bias and"
                  % (r["task"], r["pct_bias"], r["bias"]))
            print("      %.0f %% is per-scene offsets, so %.0f %% of it is systematic and"
                  % (r["pct_offset"], r["pct_bias"] + r["pct_offset"]))
            print("      removable by an amortized retrieval fitted to a matched population.")
    print("    Where that systematic share is large the learned models win; where it is")
    print("    small only per-sounding skill decides the comparison.")
    _emit(b, "why_profiles_error_budget", out_dir)

    print()
    print("   Calibration of the near-surface predictive -- is the flow honest about not")
    print("   knowing?  z is the RMS error in units of each method's own predictive sd.")
    rows = []
    levels = l4.levels or [17, 18, 19, 20]
    for j, k in enumerate(levels):
        y = l4.y[:, j]
        rec = {"level": k}
        for m, P, S in (("flow", l4.flow, l4.flow_sd), ("Laplace", l4.lap, l4.lap_sd),
                        ("ACOS", l4.acos, l4.acos_sd)):
            rec[m + "_rmse"] = rmse(P[:, j], y)
            rec[m + "_sd"] = float(S[:, j].mean())
            rec[m + "_z"] = float(np.sqrt(np.mean(((P[:, j] - y) / S[:, j]) ** 2)))
        lo = np.percentile(l4.samples[:, :, j], 2.5, axis=1)
        hi = np.percentile(l4.samples[:, :, j], 97.5, axis=1)
        rec["flow_cov95"] = float(100 * np.mean((y >= lo) & (y <= hi)))
        rec["ACOS_cov95"] = float(100 * np.mean(
            np.abs(l4.acos[:, j] - y) <= 1.96 * l4.acos_sd[:, j]))
        rows.append(rec)
    c = pd.DataFrame(rows)
    print(c.round(2).to_string(index=False))
    print()
    zf = c[[k for k in c.columns if k == "flow_z"]].to_numpy().ravel()
    za = c[[k for k in c.columns if k == "ACOS_z"]].to_numpy().ravel()
    cv = c["flow_cov95"].to_numpy()
    print("    flow: z = %.2f-%.2f, %.1f-%.1f %% coverage at a nominal 95. A z above 1 means"
          % (np.nanmin(zf), np.nanmax(zf), np.nanmin(cv), np.nanmax(cv)))
    print("    the predictive is too NARROW for its own error; below 1, too wide. ACOS runs")
    print("    at z = %.2f-%.2f here." % (np.nanmin(za), np.nanmax(za)))
    print("    RMSE rewards an informative but over-dispersed point estimate and a proper")
    print("    scoring rule does not, which is why the two disagree on this task.")
    _emit(t, "why_profiles_floor", out_dir)
    _emit(c, "why_profiles_near_surface_calibration", out_dir)
    return t, c


def section_7_clusters(pro, xc, l4, split_csv, out_dir):
    print()
    print("=" * 100)
    print("7. THE SECOND, SEPARABLE EFFECT: cluster sampling.")
    print("=" * 100)
    if split_csv is None or not Path(split_csv).exists():
        print("    (skipped: --split-csv not found, so training sizes are unknown)")
        return None
    sp = pd.read_csv(split_csv, usecols=["clst", "ref", "split"])
    n_train = sp[sp.split == "train"].groupby("clst")["ref"].nunique().to_dict()
    n_test = sp[sp.split == "test"].groupby("clst")["ref"].nunique().to_dict()

    tasks = [("xco2", xc, [0]), ("last4", l4, [0, 1, 2, 3]),
             ("prof", pro, list(range(pro.y.shape[1])))]
    rows = []
    for c in sorted(np.unique(pro.clst)):
        m = pro.clst == c
        rec = dict(cluster=int(c), n_train=n_train.get(c, 0),
                   n_test_scenes=n_test.get(c, 0), n=int(m.sum()))
        for nm, art, ks in tasks:
            if art is None:
                continue
            f = rmse(art.flow[m][:, ks], art.y[m][:, ks])
            a = rmse(art.acos[m][:, ks], art.y[m][:, ks])
            rec[nm + "_flow"] = f
            rec[nm + "_ACOS"] = a
            rec[nm + "_ratio"] = f / a
        rows.append(rec)
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))

    try:
        from scipy.stats import spearmanr
        print()
        print("    Spearman against n_train -- ACOS is the control, it does not train:")
        for nm, art, _ in tasks:
            if art is None:
                continue
            for who in ("flow", "ACOS", "ratio"):
                col = nm + "_" + who
                if col not in t:
                    continue
                r, p = spearmanr(t.n_train, t[col])
                print("      %-14s rho = %+.3f   p = %.3f" % (col, r, p))
    except ImportError:
        print("    (scipy not available; skipping the rank correlations)")

    print()
    print("    With clusters 8 and 9 removed entirely:")
    keep = ~np.isin(pro.clst, [8, 9])
    for nm, art, ks in tasks:
        if art is None:
            continue
        f = rmse(art.flow[keep][:, ks], art.y[keep][:, ks])
        a = rmse(art.acos[keep][:, ks], art.y[keep][:, ks])
        lp = rmse(art.lap[keep][:, ks], art.y[keep][:, ks])
        allr = (rmse(art.flow[:, ks], art.y[:, ks])
                / rmse(art.acos[:, ks], art.y[:, ks]))
        print("      %-6s flow %7.3f  Laplace %7.3f  ACOS %7.3f   flow/ACOS %.3f"
              "  (all clusters: %.3f)" % (nm, f, lp, a, f / a, allr))

    print()
    print("    Share of the flow-minus-ACOS mean squared error gap carried by 8 and 9,")
    print("    which are %.1f %% of the test soundings:" % (100 * np.mean(~keep)))
    for nm, art, ks in tasks:
        if art is None:
            continue
        gap = ((art.flow[:, ks] - art.y[:, ks]) ** 2
               - (art.acos[:, ks] - art.y[:, ks]) ** 2)
        tot = gap.mean()
        part = gap[~keep].mean() * np.mean(~keep)
        share = 100 * part / tot if tot != 0 else float("nan")
        print("      %-6s gap %+9.3f ppm^2   from 8+9 %+9.3f  (%5.1f %%)"
              % (nm, tot, part, share))

    print()
    print("    An ORACLE per-cluster debias (subtract each cluster's own mean test error;")
    print("    an upper bound on what any per-cluster correction could ever buy):")
    for nm, art, ks in tasks:
        if art is None:
            continue
        out = {}
        for who, P in (("flow", art.flow), ("ACOS", art.acos)):
            e = P[:, ks] - art.y[:, ks]
            d = e.copy()
            for c in np.unique(art.clst):
                mm = art.clst == c
                d[mm] = e[mm] - e[mm].mean(axis=0)
            out[who] = (float(np.sqrt(np.mean(e ** 2))),
                        float(np.sqrt(np.mean(d ** 2))))
        print("      %-6s flow %6.3f -> %6.3f   ACOS %6.3f -> %6.3f   "
              "ratio %.3f -> %.3f"
              % (nm, out["flow"][0], out["flow"][1], out["ACOS"][0], out["ACOS"][1],
                 out["flow"][0] / out["ACOS"][0], out["flow"][1] / out["ACOS"][1]))
    print()
    print("    Read that last block carefully. On the profile tasks the ratio gets WORSE")
    print("    once cluster intercepts are removed by fiat, because much of what the flow")
    print("    wins there is the removal of ACOS's own cluster-level bias. The")
    print("    near-surface deficit is not an intercept problem, which is why sections")
    print("    3-6 and not section 7 are the explanation for it.")
    _emit(t, "why_profiles_per_cluster", out_dir)
    return t


# ---------------------------------------------------------------------------------
def _resolve_group(art, name, cov_csv):
    """Per-sounding values of `name`, from the artifact or from a joined covariate CSV.

    The artifact stores whatever predict.py chose to keep (SlrZen, Footprint, the two
    aerosol types, latitude, longitude). Anything else -- `prior_psfc_pa` and `airmass`
    above all, which are what separate the high-altitude regime -- has to come from the
    split table, joined on `ID`, or from a per-scene table joined on (clst, ref).
    """
    if name in art.aux:
        return np.asarray(art.aux[name])
    if name == "clst":
        return art.clst
    if cov_csv is None:
        raise KeyError(
            f"'{name}' is not stored in the artifact (it has {sorted(art.aux)}); "
            "pass --covariate-csv with the split table or a per-scene table")
    df = pd.read_csv(cov_csv)
    if name not in df.columns:
        raise KeyError(f"'{name}' is not a column of {cov_csv}")
    if "ID" in df.columns and len(df) >= len(art.ids):
        s = df.set_index("ID")[name]
        return s.reindex(art.ids).to_numpy()
    if {"clst", "ref"} <= set(df.columns):          # a per-scene table, e.g. scenes_202003.csv
        s = df.set_index([df.clst.astype(int), df.ref.astype(int)])[name]
        return s.reindex(list(zip(art.clst, art.ref))).to_numpy()
    raise KeyError(f"{cov_csv} has neither an ID column nor (clst, ref) to join on")


def group_figure(tables, name, fig_dir, stem):
    """Skill against the scene-oracle floor, regime by regime.

    One panel per task. Below 1.0 is genuine per-sounding skill; at or above 1.0 the
    method has recovered the scene and nothing more. The point of the figure is the
    contrast between a flat line and a sloping one.
    """
    if fig_dir is None or not tables:
        return
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    n = len(tables)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.0), squeeze=False, **_LAYOUT)
    for ax, (task, t) in zip(axes[0], tables):
        x = np.arange(len(t))
        ax.axhline(1.0, color="0.45", lw=1.4)
        ax.plot(x, t["ACOS_over_floor"], marker="o", ms=5, color=COLORS["ACOS"],
                label="ACOS L2FP (v10)")
        ax.plot(x, t["Laplace"] / t["floor"], marker="o", ms=5, color=COLORS["Laplace"],
                label="Laplace")
        ax.plot(x, t["flow_over_floor"], marker="o", ms=5, color=COLORS["flow"],
                label="Normalizing flow")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in t[name]], rotation=30, ha="right", fontsize=9)
        ax.set_xlabel(name + "  (quantile bins, increasing)")
        ax.set_ylabel("RMSE / scene-oracle floor")
        ax.set_title(task)
        ax.legend(loc="best")
    fig.suptitle("below 1.0 = genuine per-sounding skill; at 1.0 = the scene and nothing more")
    path = fig_dir / (stem + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("    wrote " + str(path))


def section_8_groups(arts, name, cov_csv, bins, out_dir, fig_dir=None, stem=None):
    """Per-regime RMSE, for any artifact and any grouping covariate.

    The near-surface deficit on March and July is best split by
    physical regime -- surface pressure and airmass -- rather than by SOM label, because
    the labels are a per-delivery stratification and do not carry across months. This is
    the function that does it, and it works on the February artifacts too.
    """
    print()
    print("=" * 108)
    print(f"8. PER-REGIME BREAKDOWN, grouped by '{name}'")
    print("=" * 108)
    drawn = []
    for task, art, ks in arts:
        if art is None:
            continue
        try:
            g = _resolve_group(art, name, cov_csv)
        except KeyError as e:
            print(f"    {task}: {e}")
            return None
        if np.issubdtype(np.asarray(g).dtype, np.number) and len(np.unique(g)) > bins:
            edges = np.nanquantile(g.astype(float), np.linspace(0, 1, bins + 1))
            edges[-1] += 1e-9
            code = np.clip(np.digitize(g.astype(float), edges[1:-1]), 0, bins - 1)
            labels = [f"[{edges[i]:.4g}, {edges[i+1]:.4g})" for i in range(bins)]
            key = np.array([labels[c] for c in code])
        else:
            key = np.asarray(g).astype(str)
        rows = []
        for grp in pd.unique(key):
            m = key == grp
            if m.sum() < 2:
                continue
            Y = art.y[m][:, ks]
            floor = rmse(art.scene_mean(art.y)[m][:, ks], Y)
            rec = {"task": task, name: grp, "n": int(m.sum()), "floor": floor}
            for meth, P in art.methods():
                rec[meth] = rmse(P[m][:, ks], Y)
            rec["flow_over_ACOS"] = rec["flow"] / rec["ACOS"]
            rec["flow_over_floor"] = rec["flow"] / floor
            rec["ACOS_over_floor"] = rec["ACOS"] / floor
            rows.append(rec)
        t = pd.DataFrame(rows).sort_values(name)
        print(t.round(3).to_string(index=False))
        print()
        _emit(t, f"why_profiles_by_{name}_{task}", out_dir)
        drawn.append((task, t))
    print("    A deficit that is concentrated in one bin is a sampling story; one that is")
    print("    flat across bins is the structural story of sections 3-6. Read the two")
    print("    `*_over_floor` columns rather than `flow_over_ACOS`: a flat one means the")
    print("    method has the same per-sounding skill in every regime, and a sloping one")
    print("    means its skill tracks how much information that regime carries.")
    # profiles duplicates last4 when the artifact is a 4-target one; do not draw it twice
    seen, uniq = set(), []
    for task, t in drawn:
        key = tuple(np.round(t["flow"].to_numpy(), 6))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((task, t))
    group_figure(uniq, name, fig_dir, stem or ("why_profiles_by_" + name))
    return drawn


def latex_per_level(per_level, within, out_dir, name="why_profiles_levels"):
    """The full 20-level table, for the appendix."""
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        r"% Generated by experiments/scoring/why_profiles.py -- regenerate, do not edit.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{RMSE [ppm]} & \multicolumn{1}{c}{ratio} "
        r"& \multicolumn{3}{c}{within-sounding correlation} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-6}\cmidrule(lr){7-9}",
        r"Level & $\mathrm{sd}(x)$ & ACOS L2FP & Laplace & Flow & Flow/ACOS "
        r"& truth vs col. & ACOS & Flow \\",
        r"\midrule",
    ]
    for r_lv, r_w in zip(per_level.itertuples(), within.itertuples()):
        best = min(r_lv.ACOS, r_lv.Laplace, r_lv.flow)
        cell = lambda v: (r"\textbf{%.2f}" % v) if abs(v - best) < 1e-9 else ("%.2f" % v)
        lines.append(
            "%d & %.2f & %s & %s & %s & %.3f & %+.2f & %.2f & %.2f \\\\"
            % (r_lv.level, r_lv.truth_sd, cell(r_lv.ACOS), cell(r_lv.Laplace),
               cell(r_lv.flow), r_lv.flow_over_ACOS,
               r_w.truth_level_vs_truth_column, r_w.ACOS_vs_truth, r_w.flow_vs_truth))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-level accuracy on the reference-sounding-disjoint test partition, "
        r"all $4{,}489$ held-out soundings. Level $20$ is nearest the surface. "
        r"$\mathrm{sd}(x)$ is the standard deviation of the true CO$_2$ concentration at "
        r"that level, so a method with no skill would score close to it. Bold marks the "
        r"best of the three at each level. The last three columns are computed after "
        r"removing each reference sounding's mean, so they measure only per-sounding "
        r"skill: the first is how strongly the true concentration at that level tracks "
        r"the true column, and the other two are how strongly each method's estimate "
        r"tracks the truth. The learned retrieval has lower RMSE than the operational "
        r"one at $16$ of the $20$ levels; the three where it does not are those at which "
        r"the true concentration stops tracking the column and the operational retrieval's "
        r"second degree of freedom takes over.}",
        r"\label{tab:perlevel}",
        r"\end{table}",
    ]
    path = out_dir / (name + ".tex")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("    wrote " + str(path))


# ---------------------------------------------------------------------------------
def figure(pro, per_level, within, fig_dir, stem="why_profiles"):
    """Three panels: the per-level errors, the crossover, and why it is there."""
    if fig_dir is None:
        return
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    lv = per_level.level.to_numpy()

    fig, (ax_r, ax_q, ax_c) = plt.subplots(1, 3, figsize=(17.2, 5.0), **_LAYOUT)

    ax_r.plot(lv, per_level.truth_sd, ls="--", lw=1.4, color=COLORS["truth"],
              label="sd of the truth")
    ax_r.plot(lv, per_level.ACOS, marker="o", ms=4, color=COLORS["ACOS"],
              label="ACOS L2FP (v10)")
    ax_r.plot(lv, per_level.Laplace, marker="o", ms=4, color=COLORS["Laplace"],
              label="Laplace")
    ax_r.plot(lv, per_level.flow, marker="o", ms=4, color=COLORS["flow"],
              label="Normalizing flow")
    ax_r.set_xlabel("pressure level (20 = surface)")
    ax_r.set_ylabel("RMSE [ppm]")
    ax_r.set_title("(a) per-level error")
    ax_r.set_xticks([1, 5, 10, 15, 20])
    ax_r.legend(loc="upper left")

    ax_q.axhline(1.0, color="0.5", lw=1.2, ls="-")
    ax_q.plot(lv, per_level.Laplace / per_level.ACOS, marker="o", ms=4,
              color=COLORS["Laplace"], label="Laplace / ACOS")
    ax_q.plot(lv, per_level.flow / per_level.ACOS, marker="o", ms=4,
              color=COLORS["flow"], label="Normalizing flow / ACOS")
    ax_q.fill_between(lv, 1.0, per_level.flow / per_level.ACOS,
                      where=(per_level.flow / per_level.ACOS) < 1.0,
                      color=COLORS["flow"], alpha=0.12)
    ax_q.set_xlabel("pressure level (20 = surface)")
    ax_q.set_ylabel("RMSE relative to ACOS L2FP")
    ax_q.set_title("(b) the flow wins aloft, loses at the surface")
    ax_q.set_xticks([1, 5, 10, 15, 20])
    ax_q.legend(loc="upper left")

    ax_c.axhline(0.0, color="0.6", lw=1)
    ax_c.plot(lv, within.truth_level_vs_truth_column, ls="--", lw=1.6,
              color=COLORS["truth"], label="truth here vs truth column")
    ax_c.plot(lv, within.ACOS_vs_truth, marker="o", ms=4, color=COLORS["ACOS"],
              label="ACOS L2FP (v10)")
    ax_c.plot(lv, within.flow_vs_truth, marker="o", ms=4, color=COLORS["flow"],
              label="Normalizing flow")
    ax_c.set_xlabel("pressure level (20 = surface)")
    ax_c.set_ylabel("within-scene correlation with the truth")
    ax_c.set_title("(c) the second degree of freedom is what it misses")
    ax_c.set_xticks([1, 5, 10, 15, 20])
    ax_c.set_ylim(-0.4, 1.0)
    ax_c.legend(loc="lower left")

    path = fig_dir / (stem + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print()
    print("    wrote " + str(path))


# ---------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", default="experiments/scoring/profiles_bo.npz",
                    help="the 20-level artifact; everything here needs it")
    ap.add_argument("--xco2", default="experiments/scoring/xco2_bo.npz")
    ap.add_argument("--last4", default="experiments/scoring/last4_bo.npz")
    ap.add_argument("--split-csv", default="group_split_v1_cov.csv",
                    help="for the per-cluster training sizes in section 6")
    ap.add_argument("--out", default=None, help="directory for the .csv tables")
    ap.add_argument("--fig", default=None, help="directory for the figure")
    ap.add_argument("--tex", default=None,
                    help="directory for the full 20-level appendix table (.tex)")
    ap.add_argument("--group-by", default=None, metavar="COLUMN",
                    help="also break the results down by this covariate instead of by "
                         "cluster -- 'prior_psfc_pa' and 'airmass' are the two that "
                         "separate the high-altitude regime. Needs --covariate-csv for "
                         "anything predict.py did not store in the artifact.")
    ap.add_argument("--covariate-csv", default=None,
                    help="split table (joined on ID) or per-scene table (joined on "
                         "clst, ref) supplying --group-by")
    ap.add_argument("--group-bins", type=int, default=4,
                    help="quantile bins for a continuous --group-by (default 4)")
    ap.add_argument("--group-fig-stem", default=None,
                    help="filename stem for the --group-by figure (default "
                         "why_profiles_by_<column>); set it when running several months "
                         "so they do not overwrite one another")
    a = ap.parse_args(argv)

    pro = Art(Path(a.profiles))
    xc = Art(Path(a.xco2)) if Path(a.xco2).exists() else None
    l4 = Art(Path(a.last4)) if Path(a.last4).exists() else None
    out_dir = Path(a.out) if a.out else None
    align([pro, xc, l4])

    print("profiles : %s  n=%d  scenes=%d" % (a.profiles, len(pro.ids), len(pro.scenes)))
    if xc is not None:
        print("xco2     : " + a.xco2)
        if not (xc.ids == pro.ids).all():
            print("    WARNING: the xco2 artifact is not on the same soundings; the "
                  "cross-task lines below are not comparable.")
    if l4 is not None:
        print("last4    : " + a.last4)

    per_level = section_1_per_level(pro, out_dir)
    section_2_variance(pro, out_dir)
    within = section_3_orthogonality(pro, xc, out_dir)
    section_4_eofs(pro, out_dir)
    section_5_response(pro, xc, l4, out_dir)
    if l4 is not None:
        section_6_floor(pro, xc, l4, out_dir)
    section_7_clusters(pro, xc, l4, a.split_csv, out_dir)
    if a.group_by:
        section_8_groups([("xco2", xc, [0]),
                          ("last4", l4, [0, 1, 2, 3]),
                          ("profiles", pro, list(range(pro.y.shape[1])))],
                         a.group_by, a.covariate_csv, a.group_bins, out_dir,
                         a.fig, a.group_fig_stem)
    if within is not None and pro.y.shape[1] == 20:
        figure(pro, per_level, within, a.fig)
        if a.tex:
            latex_per_level(per_level, within, Path(a.tex))
    elif a.fig or a.tex:
        print()
        print("    (figure and .tex table skipped: they are defined on the full "
              "20-level profile)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
