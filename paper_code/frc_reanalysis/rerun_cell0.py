"""Re-run REPORT_imaging.ipynb cell 0 unchanged except: input path, output dir,
and the x-axis of the intensity profiles, which is drawn with the real pixel size
(150 um / 512 px) instead of the 0.1 um PSF grid. PSF and restoration untouched."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")

NB, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
src = "".join(json.load(open(NB))["cells"][0]["source"])

subs = [
    ('IMAGE_PATH = "/Users/francesco/Repositories /CNR_SNR_3channels/zstack_150mic_50piani_10ripetizioni_50mW_512px_770nm_Ch1.tif"',
     f'IMAGE_PATH = {DATA!r}'),
    ('cartella_salvataggio = "/Users/francesco/Repositories /noise_images/IMM"', f'cartella_salvataggio = {OUT!r}'),
    ("asse_x_roi = np.arange(x0, x1) * pixel_size", "asse_x_roi = np.arange(x0, x1) * (150.0 / 512.0)"),
    ("asse_x_um = np.arange(raw_norm.shape[1]) * pixel_size", "asse_x_um = np.arange(raw_norm.shape[1]) * (150.0 / 512.0)"),
    ("ax_overall.set_xlim(x0 * pixel_size, x1 * pixel_size)", "ax_overall.set_xlim(x0 * (150.0 / 512.0), x1 * (150.0 / 512.0))"),
]
for a, b in subs:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
src = src.replace("plt.show()", "plt.close('all')")
exec(compile(src, "REPORT_imaging_cell0", "exec"), {"__name__": "__cell0__"})
