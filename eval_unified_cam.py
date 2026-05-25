"""
eval_unified_cam.py
===================
Faithfulness + localization protocol for the unified CAM-style attribution
objective. Certifies the ONE claim Eq. (1) actually licenses: that the
genuinely-new interior region (lambda in (0,1), finite-bandwidth w) buys
something the corners (Eigen-CAM / Grad-CAM / DCAM) do not -- and certifies it
on a metric the interior was NOT directly trained on, so "we optimize class
evidence then measure class evidence" cannot be the explanation.

THE CONFOUND, AND HOW THIS HARNESS AVOIDS IT
--------------------------------------------
The interior basis is class-TILTED: lambda>0 pulls P toward directions that
carry the pre-ReLU class explanation L~(x) = <alpha, A(x)>. Deletion/insertion
AUC scores a map by how fast the *class logit* moves as you mask by saliency.
So scoring the interior on deletion/insertion half-defines the win into
existence. We therefore split the axes:

  CAUSAL axis  (deletion / insertion AUC, + comprehensiveness/sufficiency):
      self-referential to the logit. Reported for ALL methods. The honest read
      is "interior at least TIES the corners here" -- a win here is suspect.

  HELD-OUT axis (pointing game + IoU vs ground-truth boxes):
      the saliency map never saw the boxes. The interior winning HERE while at
      least tying on the causal axis is the defensible "better". This is the
      verdict metric. If no boxes are mounted, this block is skipped and the
      harness prints WHY the remaining verdict is weaker.

WHAT IS HELD FIXED ACROSS METHODS (so a delta means the basis, not the setup)
-----------------------------------------------------------------------------
  * same backbone, same target layer, same normalized activation space,
  * same query image, same rank D,
  * same perturbation schedule (deletion/insertion step grid, same blur sigma),
  * same upsampling of the [H,W] map to 224x224 (bilinear), same |.| convention.
Only (phi, w, lambda, omega) -- i.e. the corner -- changes. Matched-D, matched
perturbation, matched everything-else: this is what makes the comparison fair
and what a reviewer will check first.

USAGE
-----
  # corners vs the new interior, causal + localization, matched D
  python eval_unified_cam.py --model resnet50 --class_id 207 \\
      --methods grad_cam eigen_cam dcam new_interior \\
      --D 50 --n_images 50 --bbox_root /data/imagenet_bbox

  # no boxes available: causal axis only (localization auto-skips)
  python eval_unified_cam.py --model resnet50 --class_id 207 \\
      --methods grad_cam dcam new_interior --D 50 --n_images 50

  # the lambda sweep that shows WHERE in (0,1) the interior pays off
  python eval_unified_cam.py --model resnet50 --class_id 207 \\
      --lambda_sweep 0.0 0.25 0.5 0.75 1.0 --bandwidth 0.8 \\
      --D 50 --n_images 50 --bbox_root /data/imagenet_bbox

This reuses the EXACT image/backbone/Grad-CAM/solver contract of
run_unified_cam.py and unified_cam_joint.py. It imports CAMBackbone and the
solver rather than reimplementing either, so the maps scored here are the same
maps those drivers visualize.
"""

import argparse
import io
import sys
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import joblib
from PIL import Image

sys.path.append('.')

# the existing contract -- DO NOT reimplement these.
from run_unified_cam import CAMBackbone, load_class_images, build_bank
from unified_cam_joint import (
    SolveConfig, solve, decompose_query, normalize_acts,
    config_for_corner, CORNER_PRESETS,
)
from full_classes import IMAGENET2012_CLASSES


# ==========================================================================
# 0. one saliency map per (method, query), in the SAME space for all methods
# ==========================================================================

def saliency_for_corner(backbone: CAMBackbone, acts_norm: torch.Tensor,
                         alpha_bank: torch.Tensor, query_idx: int,
                         A_query_norm: torch.Tensor, alpha_query: torch.Tensor,
                         corner: str, D: int, lam_override: Optional[float],
                         bandwidth: float, n_restarts: int,
                         ) -> Tuple[torch.Tensor, Dict]:
    """Solve the unified objective at `corner` and read out the [H,W] saliency.

    Returns (saliency [H,W] in [0,1], diag). The map is the SOLVED-basis
    rank-D Grad-CAM reconstruction `recon` for phi=id corners; for a kernel
    corner (no additive identity) it falls back to the true Grad-CAM the solver
    still defines (decompose_query returns it as `recon` placeholder). Either
    way every method is reduced to a single normalized [H,W] map, so the
    perturbation/localization scorers downstream are method-agnostic.
    """
    cfg = config_for_corner(corner, D=D)
    if lam_override is not None:
        cfg.lam = lam_override
    cfg.bandwidth = bandwidth
    cfg.n_restarts = n_restarts

    result = solve(acts_norm, alpha_bank, alpha_query, query_idx,
                   cfg, solver="block")
    dec = decompose_query(result, A_query_norm, alpha_query)

    sal = dec["recon"].float()                       # [H,W], >=0 (post-ReLU)
    sal = _unit_normalize(sal)
    diag = {
        "corner": result.diagnostics["corner"],
        "lambda": cfg.lam, "D": D,
        "eigengap": result.diagnostics["eigengap"],
        "seed_invariant": result.diagnostics["seed_invariant"],
        "additive_exact": result.diagnostics["additive_exact"],
        "spatial_cos": dec.get("spatial_cos", float("nan")),
    }
    return sal, diag


def _unit_normalize(m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = m - m.min()
    mx = m.max().clamp(min=eps)
    return m / mx


def upsample_map(sal_hw: torch.Tensor, size: int = 224) -> torch.Tensor:
    """[H,W] -> [size,size] bilinear, the SAME upsample for every method."""
    m = sal_hw.view(1, 1, *sal_hw.shape).float()
    up = F.interpolate(m, size=(size, size), mode="bilinear",
                       align_corners=False)
    return up.view(size, size)


# ==========================================================================
# 1. CAUSAL axis: deletion / insertion AUC  (self-referential -- TIE metric)
# ==========================================================================

@torch.no_grad()
def _logit_for_class(backbone: CAMBackbone, x: torch.Tensor,
                     class_id: int) -> float:
    """Softmax-prob of `class_id` for a single preprocessed input x [1,3,H,W]."""
    out = backbone.backbone(x.to(backbone.device))
    p = torch.softmax(out, dim=1)[0, class_id]
    return float(p.item())


def _gaussian_blur(x: torch.Tensor, sigma: float = 11.0) -> torch.Tensor:
    """Blurred baseline for insertion (start = uninformative image). A blur
    baseline is the RISE convention and avoids the 'zero pixels are a valid
    class cue' artefact of a black baseline."""
    k = int(4 * sigma + 1) | 1
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g1 = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g1 = (g1 / g1.sum())
    kernel = (g1[:, None] * g1[None, :]).view(1, 1, k, k)
    kernel = kernel.expand(x.shape[1], 1, k, k)
    return F.conv2d(x, kernel, padding=k // 2, groups=x.shape[1])


@torch.no_grad()
def deletion_insertion_auc(backbone: CAMBackbone, x_pre: torch.Tensor,
                           sal_224: torch.Tensor, class_id: int,
                           n_steps: int = 50, blur_sigma: float = 11.0
                           ) -> Dict[str, float]:
    """Deletion and insertion AUC for one (image, saliency, class).

    x_pre   : [1,3,224,224] the PREPROCESSED query (normalized to backbone).
    sal_224 : [224,224] saliency in [0,1].

    Deletion : start from the full image, progressively REPLACE the most-
               salient pixels with the blurred baseline; AUC of class-prob vs
               fraction removed. LOWER is better (a faithful map kills the
               prob fast).
    Insertion: start from the blurred baseline, progressively REVEAL the most-
               salient pixels; AUC of class-prob vs fraction inserted. HIGHER
               is better.

    The pixel ordering is shared (descending saliency), the step grid is shared,
    the baseline is shared -> the ONLY thing that moves the AUC between methods
    is the map. That is the matched-perturbation guarantee.
    """
    dev = backbone.device
    x = x_pre.to(dev)
    base = _gaussian_blur(x, sigma=blur_sigma)

    order = torch.argsort(sal_224.flatten(), descending=True)  # most salient first
    n_pix = order.numel()
    step = max(1, n_pix // n_steps)
    fracs, del_scores, ins_scores = [], [], []

    x_del = x.clone()
    x_ins = base.clone()
    x_flat_del = x_del.view(1, 3, -1)
    x_flat_ins = x_ins.view(1, 3, -1)
    src = x.view(1, 3, -1)
    bse = base.view(1, 3, -1)

    del_scores.append(_logit_for_class(backbone, x_del, class_id))
    ins_scores.append(_logit_for_class(backbone, x_ins, class_id))
    fracs.append(0.0)
    for i in range(0, n_pix, step):
        idx = order[i:i + step]
        x_flat_del[0, :, idx] = bse[0, :, idx]       # delete: -> baseline
        x_flat_ins[0, :, idx] = src[0, :, idx]       # insert: -> source
        del_scores.append(_logit_for_class(
            backbone, x_flat_del.view(1, 3, 224, 224), class_id))
        ins_scores.append(_logit_for_class(
            backbone, x_flat_ins.view(1, 3, 224, 224), class_id))
        fracs.append(min(1.0, (i + step) / n_pix))

    fr = np.asarray(fracs)
    return {
        "deletion_auc": float(np.trapz(np.asarray(del_scores), fr)),
        "insertion_auc": float(np.trapz(np.asarray(ins_scores), fr)),
    }


@torch.no_grad()
def comprehensiveness_sufficiency(backbone: CAMBackbone, x_pre: torch.Tensor,
                                  sal_224: torch.Tensor, class_id: int,
                                  top_frac: float = 0.2,
                                  blur_sigma: float = 11.0) -> Dict[str, float]:
    """ERASER-style pair, no annotations needed (causal axis, single-point).

    comprehensiveness = p(full) - p(remove top-k%)   HIGHER is better
    sufficiency       = p(full) - p(keep only top-k%) LOWER is better
    A faithful map: removing its top region tanks the prob (high comp), and
    keeping only its top region preserves it (low suff)."""
    dev = backbone.device
    x = x_pre.to(dev)
    base = _gaussian_blur(x, sigma=blur_sigma)
    order = torch.argsort(sal_224.flatten(), descending=True)
    k = int(top_frac * order.numel())
    top = order[:k]

    p_full = _logit_for_class(backbone, x, class_id)

    x_rm = x.view(1, 3, -1).clone()
    x_rm[0, :, top] = base.view(1, 3, -1)[0, :, top]
    p_rm = _logit_for_class(backbone, x_rm.view(1, 3, 224, 224), class_id)

    x_keep = base.view(1, 3, -1).clone()
    x_keep[0, :, top] = x.view(1, 3, -1)[0, :, top]
    p_keep = _logit_for_class(backbone, x_keep.view(1, 3, 224, 224), class_id)

    return {"comprehensiveness": float(p_full - p_rm),
            "sufficiency": float(p_full - p_keep)}


# ==========================================================================
# 2. HELD-OUT axis: pointing game + IoU vs ground-truth boxes  (VERDICT)
# ==========================================================================

def load_bbox_for_image(bbox_root: Path, wnid: str, image_stem: str
                        ) -> Optional[List[Tuple[int, int, int, int]]]:
    """Parse an ImageNet-style PASCAL-VOC XML annotation -> list of boxes in
    ORIGINAL image pixel coords (xmin,ymin,xmax,ymax). Returns None if absent.

    Layout assumed: bbox_root/<wnid>/<image_stem>.xml . If your annotations sit
    elsewhere, point --bbox_root at the dir and/or adjust this one function;
    nothing else downstream depends on the layout."""
    xml = bbox_root / wnid / f"{image_stem}.xml"
    if not xml.exists():
        xml = bbox_root / f"{image_stem}.xml"
    if not xml.exists():
        return None
    try:
        root = ET.parse(str(xml)).getroot()
    except ET.ParseError:
        return None
    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        boxes.append((int(float(bnd.find("xmin").text)),
                      int(float(bnd.find("ymin").text)),
                      int(float(bnd.find("xmax").text)),
                      int(float(bnd.find("ymax").text))))
    return boxes or None


def boxes_to_224_mask(boxes: List[Tuple[int, int, int, int]],
                      orig_w: int, orig_h: int) -> torch.Tensor:
    """Map original-pixel boxes through the SAME Resize(256)->CenterCrop(224)
    the backbone transform uses, and rasterize to a [224,224] {0,1} mask.

    This must mirror CAMBackbone.transform EXACTLY or the localization scores
    are silently off. Resize(256) scales the short side to 256; CenterCrop(224)
    takes the central 224x224. We replicate that geometry here."""
    short = min(orig_w, orig_h)
    scale = 256.0 / short
    rw, rh = orig_w * scale, orig_h * scale
    off_x = (rw - 224) / 2.0
    off_y = (rh - 224) / 2.0
    mask = torch.zeros(224, 224)
    for (xmin, ymin, xmax, ymax) in boxes:
        x0 = int(round(xmin * scale - off_x)); x1 = int(round(xmax * scale - off_x))
        y0 = int(round(ymin * scale - off_y)); y1 = int(round(ymax * scale - off_y))
        x0 = max(0, min(223, x0)); x1 = max(0, min(224, x1))
        y0 = max(0, min(223, y0)); y1 = max(0, min(224, y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1.0
    return mask


def pointing_game(sal_224: torch.Tensor, gt_mask: torch.Tensor) -> int:
    """1 if the argmax of the saliency map lands inside any GT box, else 0.
    The classic localization hit-test: does the single hottest point sit on the
    object? Insensitive to the map's calibration -- only its peak location."""
    flat = int(torch.argmax(sal_224).item())
    y, x = divmod(flat, sal_224.shape[1])
    return int(gt_mask[y, x].item() > 0.5)


def localization_iou(sal_224: torch.Tensor, gt_mask: torch.Tensor,
                     thresh_frac: float = 0.5) -> float:
    """IoU between the thresholded saliency and the GT mask.

    Threshold at thresh_frac of the map's max (a fixed relative threshold, the
    same for every method -> calibration-fair). IoU = |pred & gt|/|pred | gt|.
    Reported alongside pointing game because pointing game is peak-only and IoU
    rewards getting the EXTENT right, which is where a class-tilted local basis
    should help or hurt visibly."""
    pred = (sal_224 >= thresh_frac * sal_224.max()).float()
    gt = (gt_mask > 0.5).float()
    inter = (pred * gt).sum()
    union = ((pred + gt) > 0.5).float().sum().clamp(min=1.0)
    return float((inter / union).item())


# ==========================================================================
# 3. paired significance: bootstrap CI on the per-image deltas
# ==========================================================================

def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                           seed: int = 0) -> Dict[str, float]:
    """Bootstrap the mean of the PAIRED delta (a-b) over images.

    Per-image pairing is the right unit: every method saw the same images, so
    the variance that matters is across images, not across methods. Returns the
    mean delta and a 95% CI; if the CI excludes 0 the difference is significant
    at ~0.05 paired. This is the number that turns 'higher mean' into 'a claim'.
    """
    d = a - b
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_delta": float(d.mean()), "ci_lo": float(lo),
            "ci_hi": float(hi), "significant": bool(lo > 0 or hi < 0),
            "n": int(n)}


# ==========================================================================
# 4. the protocol
# ==========================================================================

@dataclass
class PerImageRow:
    image_idx: int
    method: str
    deletion_auc: float
    insertion_auc: float
    comprehensiveness: float
    sufficiency: float
    pointing_hit: Optional[int]
    iou: Optional[float]
    eigengap: float
    seed_invariant: bool
    spatial_cos: float


def run_protocol(args) -> None:
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = CAMBackbone(args.model, args.target_layer or
                           __import__('run_unified_cam').MODEL_CONFIGS[
                               args.model]['default_target_layer'], device=dev)

    test_meta = Path(args.test_data_dir) / "test_metadata.pkl"
    bank_classes = args.bank_classes or [args.class_id]
    acts_norm, alpha_bank, index = build_bank(
        backbone, test_meta, bank_classes, args.bank_per_class)

    # which methods? either explicit corners, or a lambda sweep (interior dial)
    if args.lambda_sweep:
        method_specs = [(f"lambda={lam:g}", "new_interior", lam)
                        for lam in args.lambda_sweep]
    else:
        method_specs = [(m, m, None) for m in args.methods]

    # ground-truth boxes available?
    bbox_root = Path(args.bbox_root) if args.bbox_root else None
    do_loc = bbox_root is not None and bbox_root.exists()
    wnid = list(IMAGENET2012_CLASSES.keys())[args.class_id] \
        if args.class_id < len(IMAGENET2012_CLASSES) else None

    if not do_loc:
        print("\n" + "!" * 76)
        print("LOCALIZATION (held-out, verdict) axis SKIPPED: no --bbox_root "
              "mounted.\nRemaining verdict rests on the CAUSAL axis, which is "
              "self-referential\nto the class logit -- a win there for the "
              "class-tilted interior is WEAK\nevidence (it may simply reflect "
              "optimizing the metric's own signal).\nMount ground-truth boxes "
              "to certify the interior on a held-out axis.")
        print("!" * 76 + "\n")

    rows: List[PerImageRow] = []
    images = load_class_images(test_meta, args.class_id, args.offset,
                               args.n_images)

    for (img, local_idx) in images:
        # query activations + alpha (the exact tensors the solver consumes)
        A_query_raw, alpha_query, pred_label = backbone.acts_and_alpha(img)
        A_query_norm = normalize_acts(A_query_raw.cpu())
        x_pre = backbone.transform(img).unsqueeze(0)

        # locate query in bank (append if absent) -- mirrors run_unified_cam
        try:
            query_idx = index.index((args.class_id, local_idx))
            acts_use, alpha_use = acts_norm, alpha_bank
        except ValueError:
            acts_use = torch.cat([acts_norm, A_query_norm], 0)
            alpha_use = torch.cat([alpha_bank, alpha_query.cpu().unsqueeze(0)], 0)
            query_idx = acts_use.shape[0] - 1

        # GT mask for this image (held-out axis)
        gt_mask = None
        if do_loc and wnid is not None:
            stem = f"{wnid}_{local_idx}"  # adjust if your stems differ
            boxes = load_bbox_for_image(bbox_root, wnid, stem)
            if boxes is not None:
                gt_mask = boxes_to_224_mask(boxes, img.width, img.height)

        for (label, corner, lam) in method_specs:
            sal_hw, diag = saliency_for_corner(
                backbone, acts_use, alpha_use, query_idx,
                A_query_norm, alpha_query, corner, args.D, lam,
                args.bandwidth, args.n_restarts)
            sal_224 = upsample_map(sal_hw, 224)

            causal = deletion_insertion_auc(
                backbone, x_pre, sal_224, args.class_id,
                n_steps=args.n_steps, blur_sigma=args.blur_sigma)
            cs = comprehensiveness_sufficiency(
                backbone, x_pre, sal_224, args.class_id,
                top_frac=args.top_frac, blur_sigma=args.blur_sigma)

            pg, iou = None, None
            if gt_mask is not None:
                pg = pointing_game(sal_224, gt_mask)
                iou = localization_iou(sal_224, gt_mask, args.iou_thresh)

            rows.append(PerImageRow(
                image_idx=local_idx, method=label,
                deletion_auc=causal["deletion_auc"],
                insertion_auc=causal["insertion_auc"],
                comprehensiveness=cs["comprehensiveness"],
                sufficiency=cs["sufficiency"],
                pointing_hit=pg, iou=iou,
                eigengap=diag["eigengap"],
                seed_invariant=diag["seed_invariant"],
                spatial_cos=diag["spatial_cos"]))

        print(f"  scored image {local_idx}  "
              f"({len([r for r in rows if r.image_idx == local_idx])} methods)")

    _summarize(rows, method_specs, do_loc, args)


def _summarize(rows: List[PerImageRow], method_specs, do_loc: bool, args):
    labels = [s[0] for s in method_specs]

    def col(method, attr):
        return np.asarray([getattr(r, attr) for r in rows
                           if r.method == method and getattr(r, attr) is not None],
                          dtype=float)

    print("\n" + "=" * 78)
    print("PER-METHOD MEANS  (matched D=%d, matched perturbation, matched layer)"
          % args.D)
    print("=" * 78)
    header = (f"{'method':<16}{'del_auc↓':>10}{'ins_auc↑':>10}"
              f"{'comp↑':>9}{'suff↓':>9}{'point↑':>9}{'iou↑':>8}"
              f"{'eigengap':>11}")
    print(header)
    print("-" * len(header))
    for m in labels:
        da = col(m, "deletion_auc"); ia = col(m, "insertion_auc")
        cp = col(m, "comprehensiveness"); su = col(m, "sufficiency")
        pg = col(m, "pointing_hit"); io = col(m, "iou")
        eg = col(m, "eigengap")
        pg_s = f"{pg.mean():.3f}" if pg.size else "  --"
        io_s = f"{io.mean():.3f}" if io.size else "  --"
        print(f"{m:<16}{da.mean():>10.4f}{ia.mean():>10.4f}"
              f"{cp.mean():>9.4f}{su.mean():>9.4f}{pg_s:>9}{io_s:>8}"
              f"{eg.mean():>11.2e}")

    # ---- the verdict: interior vs the strongest corner, on the HELD-OUT axis
    interior = [m for m in labels if "interior" in m or "lambda" in m]
    corners = [m for m in labels if m not in interior]
    print("\n" + "=" * 78)
    print("VERDICT  (paired bootstrap, interior - corner; CI excluding 0 = sig.)")
    print("=" * 78)
    if not interior:
        print("  no interior method in the run; nothing to certify.")
        return

    # pick the interior point and the best corner to compare against
    target = interior[-1] if not args.lambda_sweep else \
        max(interior, key=lambda m: col(m, "iou").mean()
            if do_loc and col(m, "iou").size else col(m, "insertion_auc").mean())

    for verdict_attr, axis, better in [
            ("iou", "HELD-OUT (localization IoU)", "higher"),
            ("pointing_hit", "HELD-OUT (pointing game)", "higher"),
            ("insertion_auc", "causal (insertion AUC)", "higher"),
            ("deletion_auc", "causal (deletion AUC)", "lower")]:
        a = col(target, verdict_attr)
        if a.size == 0:
            if "HELD-OUT" in axis:
                print(f"  [{axis}] skipped -- no ground-truth boxes.")
            continue
        # compare against each corner; report the toughest (smallest |delta|)
        for c in corners:
            b = col(c, verdict_attr)
            if b.size == 0 or b.size != a.size:
                continue
            # for 'lower is better' metrics flip the delta sign so >0 = win
            sign = 1.0 if better == "higher" else -1.0
            bs = paired_bootstrap_delta(sign * a, sign * b)
            tag = "WIN" if (bs["significant"] and bs["mean_delta"] > 0) else (
                  "LOSS" if (bs["significant"] and bs["mean_delta"] < 0)
                  else "TIE")
            star = "  <-- held-out verdict" if "HELD-OUT" in axis else ""
            print(f"  [{axis}] {target} vs {c}: "
                  f"Δ={bs['mean_delta']:+.4f} "
                  f"CI[{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}] "
                  f"{tag}{star}")

    print("\nHow to read this:")
    print("  * A held-out WIN (localization) WHILE causal axis is TIE-or-WIN is")
    print("    the defensible 'interior is better'. The interior earned a metric")
    print("    it was not trained on.")
    print("  * A causal-only WIN with a localization TIE/LOSS is the CONFOUND")
    print("    showing through: you optimized class evidence and measured class")
    print("    evidence. Do NOT claim 'better' on that alone.")
    if not do_loc:
        print("  * NOTE: localization axis was skipped -> only the weak causal")
        print("    evidence is available. Treat any 'WIN' above as provisional.")

    # dump raw rows for the paper's table / appendix
    out = Path(args.output_json)
    out.write_text(json.dumps([r.__dict__ for r in rows], indent=2))
    print(f"\nPer-image rows written to {out}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(
        description="Faithfulness + localization protocol for the unified CAM "
                    "objective (causal axis = tie metric, localization = "
                    "held-out verdict).")
    ap.add_argument('--model', type=str, default='resnet50')
    ap.add_argument('--target_layer', type=str, default=None)
    ap.add_argument('--class_id', type=int, required=True)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--n_images', type=int, default=50)
    ap.add_argument('--methods', type=str, nargs='+',
                    default=['grad_cam', 'eigen_cam', 'dcam', 'new_interior'],
                    help=f"corners to compare; choose from "
                         f"{list(CORNER_PRESETS)}")
    ap.add_argument('--lambda_sweep', type=float, nargs='+', default=None,
                    help='If set, ignore --methods and sweep the interior '
                         'supervision dial lambda (new_interior basis).')
    ap.add_argument('--D', type=int, default=50)
    ap.add_argument('--bandwidth', type=float, default=0.8)
    ap.add_argument('--n_restarts', type=int, default=1)
    # perturbation schedule (shared across methods)
    ap.add_argument('--n_steps', type=int, default=50)
    ap.add_argument('--blur_sigma', type=float, default=11.0)
    ap.add_argument('--top_frac', type=float, default=0.2)
    # localization
    ap.add_argument('--bbox_root', type=str, default=None,
                    help='Root of PASCAL-VOC XML boxes. Absent -> localization '
                         'axis is skipped and the verdict is flagged weaker.')
    ap.add_argument('--iou_thresh', type=float, default=0.5,
                    help='Relative threshold (fraction of map max) for IoU.')
    # bank
    ap.add_argument('--bank_classes', type=int, nargs='+', default=None)
    ap.add_argument('--bank_per_class', type=int, default=12)
    ap.add_argument('--test_data_dir', type=str,
                    default='/data/imagenet1k_sampletest')
    ap.add_argument('--output_json', type=str, default='eval_rows.json')
    args = ap.parse_args()
    run_protocol(args)


if __name__ == "__main__":
    main()