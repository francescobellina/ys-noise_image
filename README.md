# ys noise_image

Denoising, deconvolution and Fourier ring correlation (FRC) analysis of two-photon microscopy images of a mouse kidney section (770 nm excitation).

**Start by reading the report:** [report_TPE_deconvolution.pdf](report_TPE_deconvolution.pdf)
*Contrast and resolution in two-photon imaging of mouse kidney tissue: two-photon brightness at 770 nm and an FRC-based assessment of Richardson–Lucy deconvolution* (F. Bellina).

## Contents

- `all_code/` contains all the notebooks developed in this repository: exercises, Fourier filtering, iterative methods (MLEM, gradient ascent), denoising pipelines and FRC.
- `paper_code/` contains the code used for the report:
  - `contrast_analysis/analisi.ipynb` covers Cellpose nuclear segmentation and the SNR/CNR of the three channels (Fig. 1, Table I).
  - `restoration/REPORT_imaging.ipynb` compares Wiener, RL, NLM+RL and RL-TV and contains the original FRC analysis (Figs. 3–4, Table II).
  - `restoration/Immagini_DENOISING_refactored_7.ipynb` produces the RL iteration series and the multichannel reconstruction with EDSR (Figs. 2, 7).
  - `frc_reanalysis/` holds the FRC re-analysis, comparing splits made after and before restoration and tracking resolution against iteration (Figs. 5–6, Table III). The results are in `results/`.

```bash
python paper_code/frc_reanalysis/frc_rerun.py    paper_code/restoration/REPORT_imaging.ipynb <Ch1.tif> <out_dir>
python paper_code/frc_reanalysis/frc_presplit.py paper_code/restoration/REPORT_imaging.ipynb <Ch1.tif> <out_dir>
python paper_code/frc_reanalysis/frc_figs2.py    <out_dir>/frc_rerun.json <out_dir>/frc_presplit.json <fig_dir>
```

Requirements: Python 3, numpy, scipy, scikit-image, PyTorch, matplotlib, seaborn, Cellpose, OpenCV.
The raw z-stack TIFF files are not included and are available on request.
