"""Re-run of the FRC-guided comparison from REPORT_imaging.ipynb (cell 1),
reusing the notebook's own classes, with two additions:
  1. FRC of the raw image (baseline for the resolution gain);
  2. FRC cutoff tracked every 2 iterations (resolution vs. iteration).
The FRC frequency axis is expressed with the real pixel size of the acquisition
(150 um / 512 px = 0.293 um/px); PSF and deconvolution are left exactly as run.
"""
import json, sys, os
import numpy as np

NB, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)

src = "".join(json.load(open(NB))["cells"][1]["source"])
src = src.replace('if __name__ == "__main__":\n    main()', "")
g = {}
exec(compile(src, "REPORT_imaging_cell1", "exec"), g)

MICRO, PARAMS = g["MICROSCOPE"], g["PARAMS"]
PX_REAL = 150.0 / 512.0
thr = 1.0 / 7.0

raw = g["load_image"](DATA, z_slice=11)
vmin, vmax = np.percentile(raw, 0.1), np.percentile(raw, 99.9)
raw = g["rescale_intensity"](raw, in_range=(vmin, vmax), out_range=(0.0, 1.0)).astype(np.float32)

physics = g["PhysicsCore"](MICRO)
psf = physics.make_airy_psf(size=PARAMS["psf_size"])
frc = g["KohoSpatialFRC"](microscope=dict(MICRO, pixel_size=PX_REAL),
                          bins=PARAMS["frc_bins"], min_pixels=PARAMS["frc_min_pixels"])

def cutoff(freqs, curve):
    c = np.nan_to_num(curve, nan=1.0)
    below = np.where(c < thr)[0]
    return float(freqs[below[0]]) if len(below) else float(freqs[-1])

res = {}
f, c, _ = frc.measure(raw)
res["Raw"] = dict(freqs=f.tolist(), frc=np.nan_to_num(c, nan=1.0).tolist(), fc=cutoff(f, c))

wien = g["restoration"].wiener(raw, psf, balance=PARAMS["wiener_balance"], clip=True)
f, c, _ = frc.measure(wien)
res["Wiener"] = dict(freqs=f.tolist(), frc=np.nan_to_num(c, nan=1.0).tolist(), fc=cutoff(f, c))

sigma = float(np.mean(g["restoration"].estimate_sigma(raw)))
nlm = np.clip(g["restoration"].denoise_nl_means(
    raw, h=PARAMS["nlm_h_factor"] * sigma, fast_mode=True,
    patch_size=PARAMS["nlm_patch"], patch_distance=PARAMS["nlm_dist"]), 0, None)

engine = g["RLTVEngine"](physics)
iters = list(range(1, 201))
keep = [15, 30, 50, 100, 200]
for name, inp, tv in [("RL Standard", raw, 0.0), ("NLM + RL", nlm, 0.0),
                      ("RL-TV", raw, PARAMS["rl_tv_weight"])]:
    out = engine.deconvolve(inp, psf, 200, tv_weight=tv, checkpoints=iters, tol=PARAMS["rl_tol"])
    fc_k, curves = [], {}
    for k in iters:
        f, c, _ = frc.measure(out["snapshots"][k])
        fc_k.append(cutoff(f, c))
        if k in keep:
            curves[k] = np.nan_to_num(c, nan=1.0).tolist()
    res[name] = dict(freqs=f.tolist(), curves=curves, fc_vs_iter=fc_k,
                     stopped_at=out["stopped_at"])
    print(name, "stopped", out["stopped_at"], {k: round(fc_k[k - 1], 3) for k in keep})
    del out

print("Raw fc", round(res["Raw"]["fc"], 3), "Wiener fc", round(res["Wiener"]["fc"], 3))
print("f_max (sub-image Nyquist) =", round(res["Raw"]["freqs"][-1], 4), "cyc/um")
json.dump(dict(px_real=PX_REAL, threshold=thr, results=res), open(os.path.join(OUT, "frc_rerun.json"), "w"))
