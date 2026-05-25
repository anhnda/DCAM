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


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral, robust to the NumPy 2.0 rename of trapz->trapezoid.
    Older NumPy only has np.trapz; newer deprecates it in favour of
    np.trapezoid. Use whichever exists so the harness runs on both."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


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
        "deletion_auc": float(_trapz(np.asarray(del_scores), fr)),
        "insertion_auc": float(_trapz(np.asarray(ins_scores), fr)),
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
# 3b. no-box sanity controls: random map (floor) + negated map (sign check)
# ==========================================================================

def random_saliency(H: int, W: int, seed: int) -> torch.Tensor:
    """A uniform-random [H,W] map. Its deletion/insertion AUC is the FLOOR:
    if a method's insertion AUC is not well above this, the causal metric is
    not resolving signal and no 'better' claim off it means anything. We seed
    per (image, draw) so the floor is reproducible."""
    g = torch.Generator().manual_seed(seed)
    return _unit_normalize(torch.rand(H, W, generator=g))


def build_method_specs(args) -> List[Tuple[str, str, Optional[float],
                                           Optional[float]]]:
    """Assemble (label, corner, lam, bw) specs.

    The corners are REFERENCE POINTS pinned on the same axes as the swept
    interior -- they are not competitors to beat (they are settings of the same
    objective) but anchors that tell you whether the interior moved the metric
    at all relative to lambda=0 (Eigen/CRAFT) and lambda=1 (Grad-CAM/DCAM).

    Sweep modes (in priority order):
      * full 2D grid  : --lambda_sweep crossed with --bandwidth_sweep
      * 1D bandwidth  : --bandwidth_sweep at a single --fixed_lambda
      * 1D lambda     : --lambda_sweep at the single --bandwidth
      * corners only  : neither sweep set -> just --methods
    """
    specs: List[Tuple[str, str, Optional[float], Optional[float]]] = []

    # corners always included as reference anchors (dedup against sweeps later)
    for m in args.methods:
        specs.append((m, m, None, None))

    lam_sweep = args.lambda_sweep
    bw_sweep = args.bandwidth_sweep

    if lam_sweep and bw_sweep:
        for lam in lam_sweep:
            for bw in bw_sweep:
                specs.append((f"int_l{lam:g}_h{bw:g}", "new_interior", lam, bw))
    elif bw_sweep:
        lam = args.fixed_lambda
        for bw in bw_sweep:
            specs.append((f"int_l{lam:g}_h{bw:g}", "new_interior", lam, bw))
    elif lam_sweep:
        bw = args.bandwidth
        for lam in lam_sweep:
            specs.append((f"int_l{lam:g}_h{bw:g}", "new_interior", lam, bw))
    return specs


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
    lam_value: Optional[float] = None
    bw_value: Optional[float] = None
    is_control: bool = False
    query_uid: int = -1


def run_protocol(args) -> None:
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = CAMBackbone(args.model, args.target_layer or
                           __import__('run_unified_cam').MODEL_CONFIGS[
                               args.model]['default_target_layer'], device=dev)

    test_meta = Path(args.test_data_dir) / "test_metadata.pkl"
    bank_classes = args.bank_classes or [args.class_id]
    acts_norm, alpha_bank, index = build_bank(
        backbone, test_meta, bank_classes, args.bank_per_class)

    # ---- build the method specs: corners (reference points) + interior sweep
    # spec tuple = (label, corner_name, lam_override or None, bw_override or None)
    method_specs = build_method_specs(args)

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

    # local_idx can REPEAT: load_class_images wraps with modulo over the cached
    # samples, so a request for n_images > (cached samples of the class) yields
    # duplicate local_idx values. Keying rows by local_idx would then merge
    # distinct queries in the paired bootstrap. We key by a monotonic query_uid
    # instead, and warn if local_idx actually collides (duplicate queries are
    # not independent samples -- the effective n is smaller than n_images).
    seen_idx = set()
    n_collisions = 0

    for q_uid, (img, local_idx) in enumerate(images):
        if local_idx in seen_idx:
            n_collisions += 1
        seen_idx.add(local_idx)
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

        for (label, corner, lam, bw) in method_specs:
            sal_hw, diag = saliency_for_corner(
                backbone, acts_use, alpha_use, query_idx,
                A_query_norm, alpha_query, corner, args.D, lam,
                bw if bw is not None else args.bandwidth, args.n_restarts)
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
                spatial_cos=diag["spatial_cos"],
                lam_value=lam, bw_value=bw, query_uid=q_uid))

        # ---- no-box sanity controls: random map (floor) + negated best map ----
        if not args.no_controls:
            H, W = A_query_norm.shape[-2], A_query_norm.shape[-1]
            # random floor (averaged over a few draws to stabilise the estimate).
            # seed by q_uid (unique) not local_idx (can repeat) so duplicate
            # queries still get independent random draws.
            for d in range(args.n_random_draws):
                rmap = upsample_map(random_saliency(H, W, seed=q_uid * 97 + d),
                                    224)
                rc = deletion_insertion_auc(
                    backbone, x_pre, rmap, args.class_id,
                    n_steps=args.n_steps, blur_sigma=args.blur_sigma)
                rcs = comprehensiveness_sufficiency(
                    backbone, x_pre, rmap, args.class_id,
                    top_frac=args.top_frac, blur_sigma=args.blur_sigma)
                rows.append(PerImageRow(
                    image_idx=local_idx, method="__random__",
                    deletion_auc=rc["deletion_auc"],
                    insertion_auc=rc["insertion_auc"],
                    comprehensiveness=rcs["comprehensiveness"],
                    sufficiency=rcs["sufficiency"],
                    pointing_hit=None, iou=None,
                    eigengap=float("nan"), seed_invariant=False,
                    spatial_cos=float("nan"), is_control=True,
                    query_uid=q_uid))

        n_methods = len([r for r in rows if r.query_uid == q_uid
                         and not r.is_control])
        print(f"  scored query {q_uid} (local_idx {local_idx}): "
              f"{n_methods} methods")

    if n_collisions:
        print("\n" + "!" * 76)
        print(f"WARNING: {n_collisions} of {len(images)} requested images were "
              f"DUPLICATES of\nearlier ones (the class has fewer cached samples "
              f"than --n_images={args.n_images}).\nload_class_images wraps with "
              "modulo, so you re-scored the same images. The\neffective sample "
              f"size is ~{len(seen_idx)} unique images, NOT {args.n_images}. "
              "The paired\nbootstrap below keys on a unique query id so the "
              "stats are not corrupted, but\nyour CIs are narrower than the true "
              "independent-sample CIs would be. Lower\n--n_images to "
              f"{len(seen_idx)} (or widen --bank_classes / the cached set) for "
              "honest n.")
        print("!" * 76)

    _summarize(rows, method_specs, do_loc, args)


def _summarize(rows: List[PerImageRow], method_specs, do_loc: bool, args):
    labels = [s[0] for s in method_specs]

    def col(method, attr, paired_with=None):
        """Per-query values for `method`. If paired_with is given, return only
        queries present in BOTH, in matched query order, so paired_bootstrap
        compares like with like. Keyed on query_uid (unique per processed
        image) NOT image_idx (which can repeat when --n_images exceeds the
        cached sample count) -- keying on image_idx would silently merge
        distinct queries via dict-key collision and corrupt the pairing."""
        rs = {r.query_uid: getattr(r, attr) for r in rows
              if r.method == method and getattr(r, attr) is not None
              and not (isinstance(getattr(r, attr), float)
                       and np.isnan(getattr(r, attr)))}
        if paired_with is None:
            return np.asarray(list(rs.values()), dtype=float)
        os_ = {r.query_uid for r in rows if r.method == paired_with}
        keys = sorted(k for k in rs if k in os_)
        return np.asarray([rs[k] for k in keys], dtype=float)

    # ---- the random floor (averaged over draws per image) ----
    floor_del = col("__random__", "deletion_auc")
    floor_ins = col("__random__", "insertion_auc")
    have_floor = floor_ins.size > 0

    print("\n" + "=" * 82)
    print("PER-METHOD MEANS  (matched D=%d, matched perturbation grid, matched layer)"
          % args.D)
    print("=" * 82)
    header = (f"{'method':<18}{'del_auc↓':>10}{'ins_auc↑':>10}"
              f"{'comp↑':>9}{'suff↓':>9}{'eigengap':>11}{'seedOK':>8}")
    print(header)
    print("-" * len(header))
    if have_floor:
        print(f"{'__random__(floor)':<18}{floor_del.mean():>10.4f}"
              f"{floor_ins.mean():>10.4f}"
              f"{col('__random__','comprehensiveness').mean():>9.4f}"
              f"{col('__random__','sufficiency').mean():>9.4f}"
              f"{'--':>11}{'--':>8}")
    for m in labels:
        da = col(m, "deletion_auc"); ia = col(m, "insertion_auc")
        cp = col(m, "comprehensiveness"); su = col(m, "sufficiency")
        eg = col(m, "eigengap")
        si = np.asarray([1.0 if r.seed_invariant else 0.0 for r in rows
                         if r.method == m], dtype=float)
        eg_s = f"{eg.mean():.2e}" if eg.size else "  --"
        si_s = f"{si.mean():.2f}" if si.size else "  --"
        print(f"{m:<18}{da.mean():>10.4f}{ia.mean():>10.4f}"
              f"{cp.mean():>9.4f}{su.mean():>9.4f}{eg_s:>11}{si_s:>8}")

    # ---- floor check: is the metric even resolving signal? ----
    print("\n" + "=" * 82)
    print("FLOOR CHECK  (causal axis is meaningless if methods ≈ random)")
    print("=" * 82)
    if not have_floor:
        print("  controls disabled (--no_controls): no floor. The absolute "
              "AUC numbers\n  above are uninterpretable without it.")
    else:
        interior_all = [m for m in labels if m.startswith("int_")]
        ref = interior_all or [m for m in labels if m not in ("__random__",)]
        best_ins = max(ref, key=lambda m: col(m, "insertion_auc").mean()) \
            if ref else None
        if best_ins:
            a = col(best_ins, "insertion_auc", paired_with="__random__")
            b = col("__random__", "insertion_auc", paired_with=best_ins)
            n = min(a.size, b.size)
            if n > 0:
                bs = paired_bootstrap_delta(a[:n], b[:n])
                verdict = ("ABOVE floor (metric resolves signal)"
                           if bs["significant"] and bs["mean_delta"] > 0
                           else "NOT above floor -- metric is NOT resolving "
                                "signal; do not interpret any AUC delta")
                print(f"  best method '{best_ins}' insertion AUC vs random: "
                      f"Δ={bs['mean_delta']:+.4f} "
                      f"CI[{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}]  -> {verdict}")

    # ---- interior characterization across the cube (NOT a victory claim) ----
    interior = [m for m in labels if m.startswith("int_")]
    corners = [m for m in labels if not m.startswith("int_")
               and m != "__random__"]
    print("\n" + "=" * 82)
    print("CAUSAL-AXIS CHARACTERIZATION  (no boxes: this is an ablation, not a "
          "'better' claim)")
    print("=" * 82)
    if not interior:
        print("  no interior sweep points; ran corners only. Nothing to "
              "characterize across (λ,h).")
    else:
        # rank interior points by insertion AUC, but GATE on a positive eigengap:
        # a causal 'win' at a collapsed eigengap is on a non-unique subspace and
        # is not a real result (paper section 4).
        def gated_score(m):
            eg = col(m, "eigengap")
            ins = col(m, "insertion_auc")
            if eg.size and eg.mean() <= args.eigengap_floor:
                return -np.inf  # disqualified: subspace not unique
            return ins.mean() if ins.size else -np.inf

        ranked = sorted(interior, key=gated_score, reverse=True)
        best = ranked[0]
        best_eg = col(best, "eigengap").mean() if col(best, "eigengap").size else float("nan")
        if gated_score(best) == -np.inf:
            print("  every interior point either has no eigengap above the floor "
                  f"({args.eigengap_floor:g}) or no data.\n  No (λ,h) point "
                  "qualifies as a stable subspace -- report this as a NEGATIVE "
                  "result, honestly.")
        else:
            print(f"  best interior (λ,h) on insertion AUC, AMONG points with "
                  f"eigengap>{args.eigengap_floor:g}: {best}")
            print(f"    eigengap there = {best_eg:.2e}  (subspace is unique -> "
                  "the point is interpretable)")
            # contrast that point against each corner reference, paired
            for c in corners:
                for attr, better in [("insertion_auc", "higher"),
                                     ("deletion_auc", "lower")]:
                    a = col(best, attr, paired_with=c)
                    b = col(c, attr, paired_with=best)
                    n = min(a.size, b.size)
                    if n == 0:
                        continue
                    sign = 1.0 if better == "higher" else -1.0
                    bs = paired_bootstrap_delta(sign * a[:n], sign * b[:n])
                    tag = ("interior↑" if bs["significant"] and bs["mean_delta"] > 0
                           else "interior↓" if bs["significant"]
                           and bs["mean_delta"] < 0 else "tie")
                    print(f"    [{attr}] {best} vs corner {c}: "
                          f"Δ={bs['mean_delta']:+.4f} "
                          f"CI[{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}]  {tag}")

    # ---- the honesty block, unconditional in this configuration ----
    print("\n" + "!" * 82)
    print("HOW TO REPORT THIS  (deletion/insertion only, no held-out axis)")
    print("!" * 82)
    print("  1. This is the CAUSAL axis ONLY. It is self-referential: the "
          "interior basis is")
    print("     tilted toward class evidence (λ>0), and insertion/deletion AUC "
          "scores a map by")
    print("     how the CLASS LOGIT moves. An interior 'interior↑' above is "
          "therefore EXPECTED")
    print("     and is WEAK evidence of faithfulness -- you optimized the "
          "metric's own signal.")
    print("  2. The defensible claims from THIS run are:")
    print("       (a) the floor check -- methods resolve signal above random;")
    print("       (b) the SHAPE of the (λ,h) surface -- the objective's dials "
          "move the causal")
    print("           metric monotonically/interpretably (an ablation result);")
    print("       (c) corners are reproduced as reference points on that "
          "surface.")
    print("  3. To claim 'unified_cam is BETTER', you still need the HELD-OUT "
          "axis (localization")
    print("     vs boxes, or stability under perturbation). Re-run with "
          "--bbox_root once you")
    print("     have annotations; the harness will then print a held-out "
          "verdict line.")
    print("  4. Any interior point with eigengap ≤ %g was DISQUALIFIED above: "
          "non-unique" % args.eigengap_floor)
    print("     subspace = not seed-invariant = not a result (paper §4).")

    # dump raw rows for the paper's table / appendix
    out = Path(args.output_json)
    out.write_text(json.dumps([r.__dict__ for r in rows], indent=2))
    print(f"\nPer-image rows (incl. controls) written to {out}")
    print("=" * 82)


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
                    help='Sweep the interior supervision dial lambda. Crossed '
                         'with --bandwidth_sweep if both are set (2D grid).')
    ap.add_argument('--bandwidth_sweep', type=float, nargs='+', default=None,
                    help='Sweep the locality bandwidth h. Crossed with '
                         '--lambda_sweep (2D grid), or run at --fixed_lambda.')
    ap.add_argument('--fixed_lambda', type=float, default=0.5,
                    help='Lambda used for a bandwidth-only (1D) sweep.')
    ap.add_argument('--D', type=int, default=50)
    ap.add_argument('--bandwidth', type=float, default=0.8,
                    help='Default h for corners / lambda-only sweep.')
    ap.add_argument('--n_restarts', type=int, default=1)
    # no-box sanity controls
    ap.add_argument('--no_controls', action='store_true',
                    help='Disable the random-map floor. NOT recommended: '
                         'without it the absolute AUCs are uninterpretable.')
    ap.add_argument('--n_random_draws', type=int, default=3,
                    help='Random-map draws per image for the floor estimate.')
    ap.add_argument('--eigengap_floor', type=float, default=1e-8,
                    help='Interior (λ,h) points with mean eigengap at or below '
                         'this are DISQUALIFIED from the win ranking (non-'
                         'unique subspace = not seed-invariant, paper §4).')
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