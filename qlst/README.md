# qlst-friendly runner (auto-detects `./data/*.zip`)

Place your dataset zip under `qlst/data/`, e.g. `qlst/data/pixelart.zip`.

Then from the `qlst` folder run:
```bash
python run_best_pack.py --outdir results_best --per_class 60 --dims 16 32 64 --bootstrap_d 64 --bootstrap_B 80
# or simply:
python run_best_pack.py  # it will auto-pick ./data/pixelart.zip if present
```
The loader accepts either:
- a `labels.csv` inside the zip (with columns like `path,label` or `idx,<one-hot…>`), or
- a folder-per-class layout inside the zip: `<class>/<filename>.png`.

Outputs in `--outdir`: `results_table.csv`, plots, `ssim_threshold.txt`, `bootstrap_cis_d*.json`, and the CI bar chart.
