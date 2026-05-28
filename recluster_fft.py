"""
recluster_fft.py
================
Reload a prior export_fft_basics.py run and re-cluster WITHOUT re-streaming.

Why this is safe to skip the heavy passes:
  export_fft_basics.py's two expensive passes (pass-1 0.99-quantile calibration,
  pass-2 streaming |FFT| + per-channel mean) produce exactly the arrays saved in
  the .npz: fft_mean [C,H,W], channel_mean [C] (= mu), channel_scale [C].
  Everything after that -- desc_transform, descriptor reduction, clustering, V
  construction -- is pure post-processing on fft_mean + mu. So we reload those
  two arrays and redo only the cheap downstream part. Seconds, no GPU, no data.

What it does:
  1. load fft_mean + mu (channel_mean) from --in_npz
  2. apply --desc_transform (none/log/log_nodc/radial_ramp/highpass)
  3. build descriptor (full/radial) and cluster (kmeans/nmf)
  4. OPTIONAL: --stability_seeds N -> recluster across N seeds, report mean
     pairwise Adjusted Rand Index (kmeans only; ARI on hard labels)
  5. save a new FFTBasisReconstructor .pkl (drop-in for hier_visualize_pca.py)
     and optionally a new .npz

Usage
-----
  # walk the ladder, each writes its own .pkl, all reusing the SAME fft_mean:
  for T in none log log_nodc radial_ramp highpass; do
    python recluster_fft.py \
      --in_npz fft_resnet50_layer3_k128.npz \
      --desc_transform $T --descriptor full --cluster kmeans --k 128 \
      --stability_seeds 3 \
      --save fft_basics_resnet50_k128_${T}_model.pkl
  done

  python hier_visualize_pca.py --class_id 207 \
      --pca_model fft_basics_resnet50_k128_log_nodc_model.pkl \
      --ring_components 16 --top_channels 4 --recon_D 64
"""

import argparse
import sys
import numpy as np
import torch
import joblib

# Pull the (patched) descriptor/cluster machinery straight from the module so
# this script and export_fft_basics.py never drift. Requires the desc_transform
# patch already applied to export_fft_basics.py.
sys.path.append('.')
from export_fft_basics import (            # noqa: E402
    build_descriptor, cluster_kmeans, cluster_nmf, membership_to_V,
    make_reconstructor, report_clusters,
)


def _adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """ARI between two hard label vectors. No sklearn dependency."""
    a = np.asarray(a); b = np.asarray(b)
    n = a.shape[0]
    ca = {v: i for i, v in enumerate(np.unique(a))}
    cb = {v: i for i, v in enumerate(np.unique(b))}
    cont = np.zeros((len(ca), len(cb)), dtype=np.int64)
    for i in range(n):
        cont[ca[a[i]], cb[b[i]]] += 1
    sum_comb = lambda x: (x * (x - 1) // 2).sum()
    index = sum_comb(cont)
    sa = sum_comb(cont.sum(axis=1))
    sb = sum_comb(cont.sum(axis=0))
    expected = sa * sb / max(n * (n - 1) // 2, 1)
    maxidx = 0.5 * (sa + sb)
    denom = maxidx - expected
    return 1.0 if denom == 0 else float((index - expected) / denom)


def stability_check(descriptor: torch.Tensor, k: int, seeds: int) -> None:
    """Recluster (kmeans) across `seeds` seeds, report mean pairwise ARI.
    High ARI (~>0.8) => the grouping is real signal; low => fitting noise."""
    label_sets = []
    for s in range(seeds):
        _, _, labels = cluster_kmeans(descriptor, k, seed=s)
        label_sets.append(labels.cpu().numpy())
    aris = []
    for i in range(seeds):
        for j in range(i + 1, seeds):
            aris.append(_adjusted_rand_index(label_sets[i], label_sets[j]))
    if aris:
        print(f"\n  [stability] mean pairwise ARI over {seeds} seeds = "
              f"{np.mean(aris):.3f}  (min={min(aris):.3f}, max={max(aris):.3f})")
        print(f"  [stability] ~>0.8 = stable grouping (real signal); "
              f"low/reshuffling = transform/k fitting noise.")
    else:
        print("  [stability] need >=2 seeds for ARI.")


def main():
    ap = argparse.ArgumentParser(
        description="Reload a prior FFT run and recluster without re-streaming.")
    ap.add_argument('--in_npz', type=str, required=True,
                    help="The .npz written by export_fft_basics.py (--save_fft). "
                         "Provides fft_mean + channel_mean (mu).")
    ap.add_argument('--desc_transform', type=str, default='none',
                    choices=['none', 'log', 'log_nodc', 'radial_ramp', 'highpass'])
    ap.add_argument('--hp_radius', type=float, default=0.25,
                    help="For --desc_transform highpass: low-freq disk radius "
                         "(fraction of r_max) to zero.")
    ap.add_argument('--descriptor', type=str, default='full',
                    choices=['full', 'radial'])
    ap.add_argument('--n_rbin', type=int, default=8)
    ap.add_argument('--cluster', type=str, default='kmeans',
                    choices=['kmeans', 'nmf'])
    ap.add_argument('--cluster_seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=128)
    ap.add_argument('--stability_seeds', type=int, default=0,
                    help="If >=2, recluster across this many seeds and report "
                         "mean pairwise ARI (kmeans only).")
    ap.add_argument('--save', type=str, default='fft_basics_recluster_model.pkl')
    ap.add_argument('--save_fft', type=str, default=None,
                    help="Optional: re-save an .npz with the new clustering "
                         "(reuses the loaded fft_mean/descriptor).")
    args = ap.parse_args()

    print("=" * 80)
    print(f"recluster_fft: reload {args.in_npz}  (no streaming)")
    data = np.load(args.in_npz, allow_pickle=True)
    fft_mean = torch.from_numpy(data['fft_mean']).to(torch.float64)   # [C,H,W]
    mu = torch.from_numpy(data['channel_mean']).to(torch.float64)     # [C]
    C, H, W = fft_mean.shape
    print(f"  loaded fft_mean[{C},{H},{W}], channel_mean[{mu.shape[0]}]")
    print(f"  desc_transform={args.desc_transform}  descriptor={args.descriptor}"
          f"  cluster={args.cluster}  k={args.k}")
    print("=" * 80)

    k = max(0, min(int(args.k), C))
    if k != args.k:
        print(f"  NOTE: clamped k from {args.k} to {k} (must be <= C={C}).")

    descriptor, desc_info = build_descriptor(
        fft_mean, mode=args.descriptor, n_rbin=args.n_rbin,
        desc_transform=args.desc_transform, hp_radius=args.hp_radius)
    print(f"  descriptor: {tuple(descriptor.shape)}  [{desc_info}]")

    if args.stability_seeds and args.stability_seeds >= 2:
        if args.cluster != 'kmeans':
            print("  [stability] ARI check is kmeans-only; skipping for nmf.")
        else:
            stability_check(descriptor, k, args.stability_seeds)

    print(f"\nClustering into {k} basics ({args.cluster}, "
          f"seed={args.cluster_seed})...")
    if args.cluster == 'kmeans':
        membership, centers, labels = cluster_kmeans(
            descriptor, k, seed=args.cluster_seed)
    else:
        membership, centers, labels = cluster_nmf(
            descriptor, k, seed=args.cluster_seed)
    report_clusters(labels, k)

    V = membership_to_V(membership)
    module = make_reconstructor(C, k, mu, V)
    joblib.dump(module, args.save)
    print(f"\nSaved FFTBasisReconstructor (k={k}, "
          f"transform={args.desc_transform}) -> {args.save}")

    if args.save_fft:
        np.savez_compressed(
            args.save_fft,
            fft_mean=fft_mean.double().cpu().numpy(),
            descriptor=descriptor.double().cpu().numpy(),
            labels=labels.long().cpu().numpy(),
            membership=membership.double().cpu().numpy(),
            centers=centers.double().cpu().numpy(),
            channel_mean=mu.double().cpu().numpy(),
            channel_scale=(data['channel_scale']
                           if 'channel_scale' in data.files else np.array([])),
            meta=np.array([{
                'reclustered_from': args.in_npz,
                'desc_transform': args.desc_transform,
                'hp_radius': args.hp_radius,
                'descriptor_mode': args.descriptor,
                'cluster': args.cluster, 'k': k,
                'cluster_seed': args.cluster_seed,
            }], dtype=object),
        )
        print(f"Saved re-clustered .npz -> {args.save_fft}")

    print("=" * 80 + "\nDone.\n" + "=" * 80)


if __name__ == "__main__":
    main()