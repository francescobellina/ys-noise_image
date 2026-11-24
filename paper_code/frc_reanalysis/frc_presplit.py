"""Independent-noise FRC: checkerboard-split the RAW image first, restore each
sub-image independently (same engine/parameters as REPORT_imaging.ipynb cell 1,
PSF resampled on the 2x coarser sub-image grid), then correlate the restored pairs."""
import json, sys, os
import numpy as np
import torch

NB, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
src = "".join(json.load(open(NB))["cells"][1]["source"])
src = src.replace('if __name__ == "__main__":\n    main()', "")
g = {}
exec(compile(src, "REPORT_imaging_cell1", "exec"), g)
MICRO, PARAMS = g["MICROSCOPE"], g["PARAMS"]
PX_REAL, thr = 150.0 / 512.0, 1.0 / 7.0

raw = g["load_image"](DATA, z_slice=11)
vmin, vmax = np.percentile(raw, 0.1), np.percentile(raw, 99.9)
raw = g["rescale_intensity"](raw, in_range=(vmin, vmax), out_range=(0.0, 1.0)).astype(np.float32)
raw = raw[: raw.shape[0] // 2 * 2, : raw.shape[1] // 2 * 2]
subs = {"a1": raw[0::2, 0::2], "a2": raw[1::2, 1::2], "b1": raw[0::2, 1::2], "b2": raw[1::2, 0::2]}

phys = g["PhysicsCore"](dict(MICRO, pixel_size=2 * MICRO["pixel_size"]))  # same PSF, coarser grid
psf = phys.make_airy_psf(size=33)  # 63 px @ 0.1 -> 33 px @ 0.2 (same physical support)
frc = g["KohoSpatialFRC"](microscope=dict(MICRO, pixel_size=PX_REAL), bins=PARAMS["frc_bins"],
                          min_pixels=PARAMS["frc_min_pixels"])
R = g["restoration"]

def pair_frc(d):
    f, c1 = frc._compute_frc_spectra(torch.from_numpy(d["a1"]).float(), torch.from_numpy(d["a2"]).float())
    _, c2 = frc._compute_frc_spectra(torch.from_numpy(d["b1"]).float(), torch.from_numpy(d["b2"]).float())
    return f, np.nan_to_num(((c1 + c2) / 2).numpy(), nan=1.0)

def cutoff(f, c):
    b = np.where(c < thr)[0]
    return float(f[b[0]]) if len(b) else float(f[-1])

res = {}
f, c = pair_frc(subs); res["Raw"] = dict(freqs=f.tolist(), frc=c.tolist(), fc=cutoff(f, c))
wien = {k: R.wiener(v, psf, balance=PARAMS["wiener_balance"], clip=True).astype(np.float32) for k, v in subs.items()}
f, c = pair_frc(wien); res["Wiener"] = dict(freqs=f.tolist(), frc=c.tolist(), fc=cutoff(f, c))

def nlm(v):
    s = float(np.mean(R.estimate_sigma(v)))
    return np.clip(R.denoise_nl_means(v, h=PARAMS["nlm_h_factor"] * s, fast_mode=True,
                   patch_size=PARAMS["nlm_patch"], patch_distance=PARAMS["nlm_dist"]), 0, None).astype(np.float32)

eng = g["RLTVEngine"](phys)
iters = list(range(1, 201)); keep = [15, 30, 50, 100, 200]
for name, pre, tv in [("RL Standard", None, 0.0), ("NLM + RL", nlm, 0.0), ("RL-TV", None, PARAMS["rl_tv_weight"])]:
    outs = {k: eng.deconvolve(pre(v) if pre else v, psf, 200, tv_weight=tv, checkpoints=iters,
                              tol=PARAMS["rl_tol"])["snapshots"] for k, v in subs.items()}
    fck, curves = [], {}
    for it in iters:
        f, c = pair_frc({k: outs[k][it] for k in subs})
        fck.append(cutoff(f, c))
        if it in keep: curves[it] = c.tolist()
    res[name] = dict(freqs=f.tolist(), curves=curves, fc_vs_iter=fck)
    fa = np.array(fck)
    print(f"{name:12s} fc@{keep}:", [round(fa[k-1], 3) for k in keep], "max", round(fa.max(), 3), "at k", int(fa.argmax()) + 1)
    del outs
print("PSF", psf.shape, "Raw fc", round(res["Raw"]["fc"], 3), "Wiener fc", round(res["Wiener"]["fc"], 3))
json.dump(dict(px_real=PX_REAL, threshold=thr, results=res), open(os.path.join(OUT, "frc_presplit.json"), "w"))
