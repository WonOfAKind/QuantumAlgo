# Basis-Multiplexed Shared Latent with strong baselines, CV accuracy, rotation/DCT tests.
import os, glob, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils import load_from_zip, psnr, ssim_batch, make_montage

plt.rcParams["figure.dpi"] = 120

CONFIG = {
    "ZIP": os.environ.get("QLST_ZIP", None),
    "OUTDIR": os.environ.get("QLST_OUTDIR", "results"),
    "PER_CLASS": int(os.environ.get("QLST_PER_CLASS", "120")),
    "DIMS": [int(x) for x in os.environ.get("QLST_DIMS", "16,32,64").split(",")],
    "TEST_SIZE": float(os.environ.get("QLST_TEST_SIZE", "0.2")),
    "SEED": int(os.environ.get("QLST_SEED", "101")),
}

def auto_find_zip(default_hint="data"):
    cand = []
    for folder in [default_hint, ".", "./data", "../data"]:
        cand += glob.glob(os.path.join(folder, "*.zip"))
    if not cand:
        return None
    for p in cand:
        if os.path.basename(p).lower().startswith("pixelart"):
            return p
    return cand[0]

def orthonormal(d, rng):
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    return Q

def zigzag_indices(n):
    idx = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(s, -1, -1):
                j = s - i
                if i < n and j < n:
                    idx.append((i, j))
        else:
            for j in range(s, -1, -1):
                i = s - j
                if i < n and j < n:
                    idx.append((i, j))
    return idx

def run_all():
    cfg = CONFIG.copy()
    if cfg["ZIP"] is None:
        auto = auto_find_zip(default_hint="data")
        if auto is None:
            raise SystemExit("No dataset found; set env QLST_ZIP or drop pixelart.zip into ./data/")
        cfg["ZIP"] = auto
    os.makedirs(cfg["OUTDIR"], exist_ok=True)

    print("[load] dataset from", cfg["ZIP"])
    X, y = load_from_zip(cfg["ZIP"], per_class=cfg["PER_CLASS"], size=(32, 32), seed=cfg["SEED"])
    N, H, W = X.shape
    print(f"[data] X={X.shape}, unique_classes={len(np.unique(y))}, per_class≤{cfg['PER_CLASS']}")

    Xf = X.reshape(N, -1)
    Xtr, Xte, ytr, yte = train_test_split(
        Xf, y, test_size=cfg["TEST_SIZE"], stratify=y, random_state=cfg["SEED"]
    )
    H = W = int(np.sqrt(Xtr.shape[1]))
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["SEED"])

    for d in cfg["DIMS"]:
        # --- Shared PCA latent ---
        pca = PCA(n_components=d, svd_solver="randomized", whiten=True, random_state=0).fit(Xtr)
        Ztr = pca.transform(Xtr)
        Zte = pca.transform(Xte)

        Xhat_lin = pca.inverse_transform(Zte).reshape(-1, H, W)

        scaler_out = StandardScaler(with_mean=True, with_std=True).fit(Xtr)
        mlp = MLPRegressor(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            early_stopping=True,
            n_iter_no_change=20,
            max_iter=1200,
            random_state=0,
            verbose=False,
        )
        mlp.fit(Ztr, scaler_out.transform(Xtr))
        Xhat_std = mlp.predict(Zte)
        Xhat_mlp = (Xhat_std * scaler_out.scale_) + scaler_out.mean_
        Xhat_mlp = np.clip(Xhat_mlp, 0.0, 1.0).reshape(-1, H, W)

        psnr_lin = float(np.mean([psnr(a, b) for a, b in zip(Xte.reshape(-1, H, W), Xhat_lin)]))
        ssim_lin = ssim_batch(Xte.reshape(-1, H, W), Xhat_lin, count=60)
        psnr_mlp = float(np.mean([psnr(a, b) for a, b in zip(Xte.reshape(-1, H, W), Xhat_mlp)]))
        ssim_mlp = ssim_batch(Xte.reshape(-1, H, W), Xhat_mlp, count=60)

        accs = []
        for tr, va in cv.split(Ztr, ytr):
            clf = LogisticRegression(max_iter=3000).fit(Ztr[tr], ytr[tr])
            accs.append(accuracy_score(ytr[va], clf.predict(Ztr[va])))
        acc_shared = float(np.mean(accs))

        results += [
            {"method": "SharedPCA-Linear", "d": d, "PSNR": round(psnr_lin, 3), "SSIM": round(ssim_lin, 4), "Accuracy": round(acc_shared, 4)},
            {"method": "SharedPCA-MLP", "d": d, "PSNR": round(psnr_mlp, 3), "SSIM": round(ssim_mlp, 4), "Accuracy": round(acc_shared, 4)},
        ]

        # --- Separate baselines, equal total d ---
        d_cls = min(len(np.unique(y)) - 1, max(2, d // 2))
        d_rec = max(1, d - d_cls)

        # (i) LDA + PCA
        lda = LDA(n_components=d_cls).fit(Xtr, ytr)
        Ztr_c = lda.transform(Xtr)
        accs_sep = []
        for tr, va in cv.split(Ztr_c, ytr):
            clf = LogisticRegression(max_iter=3000).fit(Ztr_c[tr], ytr[tr])
            accs_sep.append(accuracy_score(ytr[va], clf.predict(Ztr_c[va])))
        acc_sep = float(np.mean(accs_sep))

        pca_r = PCA(n_components=d_rec, svd_solver="randomized", whiten=True, random_state=1).fit(Xtr)
        Xhat_sep = pca_r.inverse_transform(pca_r.transform(Xte)).reshape(-1, H, W)
        psnr_sep = float(np.mean([psnr(a, b) for a, b in zip(Xte.reshape(-1, H, W), Xhat_sep)]))
        ssim_sep = ssim_batch(Xte.reshape(-1, H, W), Xhat_sep, count=60)
        results.append({"method": "Separate(LDA+PCA)", "d": d, "PSNR": round(psnr_sep, 3), "SSIM": round(ssim_sep, 4), "Accuracy": round(acc_sep, 4)})

        # (ii) PCA + PCA split
        d1 = d // 2
        d2 = d - d1
        pca_c = PCA(n_components=d1, svd_solver="randomized", whiten=True, random_state=2).fit(Xtr)
        Ztr_c2 = pca_c.transform(Xtr)
        accs_sep2 = []
        for tr, va in cv.split(Ztr_c2, ytr):
            clf = LogisticRegression(max_iter=3000).fit(Ztr_c2[tr], ytr[tr])
            accs_sep2.append(accuracy_score(ytr[va], clf.predict(Ztr_c2[va])))
        acc_sep2 = float(np.mean(accs_sep2))

        pca_r2 = PCA(n_components=d2, svd_solver="randomized", whiten=True, random_state=3).fit(Xtr)
        Xhat_sep2 = pca_r2.inverse_transform(pca_r2.transform(Xte)).reshape(-1, H, W)
        psnr_sep2 = float(np.mean([psnr(a, b) for a, b in zip(Xte.reshape(-1, H, W), Xhat_sep2)]))
        ssim_sep2 = ssim_batch(Xte.reshape(-1, H, W), Xhat_sep2, count=60)
        results.append({"method": "Separate(PCA+PCA)", "d": d, "PSNR": round(psnr_sep2, 3), "SSIM": round(ssim_sep2, 4), "Accuracy": round(acc_sep2, 4)})

        # --- Rotation invariance (rotate latent basis, retrain readouts) ---
        rng = np.random.RandomState(1234 + d)
        R = orthonormal(d, rng)  # d x d orthogonal
        Ztr_rot = Ztr @ R
        Zte_rot = Zte @ R
        Wp = R.T @ pca.components_
        Xhat_rot = (Zte_rot @ Wp) + pca.mean_
        Xhat_rot = Xhat_rot.reshape(-1, H, W)
        psnr_rot = float(np.mean([psnr(a, b) for a, b in zip(Xte.reshape(-1, H, W), Xhat_rot)]))
        ssim_rot = ssim_batch(Xte.reshape(-1, H, W), Xhat_rot, count=60)
        accs_rot = []
        for tr, va in cv.split(Ztr_rot, ytr):
            clf = LogisticRegression(max_iter=3000).fit(Ztr_rot[tr], ytr[tr])
            accs_rot.append(accuracy_score(ytr[va], clf.predict(Ztr_rot[va])))
        results.append({"method": "SharedPCA-ROT", "d": d, "PSNR": round(psnr_rot, 3), "SSIM": round(ssim_rot, 4), "Accuracy": round(float(np.mean(accs_rot)), 4)})

        # --- Harmonic (DCT) baseline (top-d zig-zag coefficients) ---
        n = H
        zz = zigzag_indices(n)
        keep = set(zz[:d])
        def dct_feats(x):
            F = np.fft.fft2(x.reshape(H, W))
            feats = []
            for i in range(H):
                for j in range(W):
                    if (i, j) in keep:
                        feats.append(F[i, j].real)
            return np.array(feats, dtype=float)
        Zc = np.stack([dct_feats(x) for x in Xtr])
        accs_dct = []
        for tr, va in cv.split(Zc, ytr):
            clf = LogisticRegression(max_iter=3000).fit(Zc[tr], ytr[tr])
            accs_dct.append(accuracy_score(ytr[va], clf.predict(Zc[va])))
        acc_dct = float(np.mean(accs_dct))

        Xte_imgs = Xte.reshape(-1, H, W)
        Xhat_dct = []
        for x in Xte_imgs:
            F = np.fft.fft2(x)
            F2 = np.zeros_like(F, dtype=complex)
            for i in range(H):
                for j in range(W):
                    if (i, j) in keep:
                        F2[i, j] = F[i, j]
            xr = np.fft.ifft2(F2).real
            Xhat_dct.append(np.clip(xr, 0, 1))
        Xhat_dct = np.stack(Xhat_dct)
        psnr_dct = float(np.mean([psnr(a, b) for a, b in zip(Xte_imgs, Xhat_dct)]))
        ssim_dct = ssim_batch(Xte_imgs, Xhat_dct, count=60)
        results.append({"method": "Harmonic(DCT)", "d": d, "PSNR": round(psnr_dct, 3), "SSIM": round(ssim_dct, 4), "Accuracy": round(acc_dct, 4)})

        # Montage at max d
        if d == max(cfg["DIMS"]):
            idxs = np.random.RandomState(42).choice(np.arange(Xte_imgs.shape[0]), size=min(6, Xte_imgs.shape[0]), replace=False)
            rows = [
                [Xte_imgs[i] for i in idxs],
                [Xhat_lin[i] for i in idxs],
                [Xhat_mlp[i] for i in idxs],
            ]
            mg = make_montage(rows)
            if mg is not None:
                plt.figure(figsize=(2.0 * len(idxs), 6))
                plt.imshow(mg, vmin=0, vmax=1, cmap="gray")
                plt.axis("off")
                plt.title(f"Original | Shared-Linear | Shared-MLP (d={d})")
                plt.tight_layout()
                plt.savefig(os.path.join(cfg["OUTDIR"], f"montage_d{d}.png"), dpi=160)
                plt.close()

    df = pd.DataFrame(results).sort_values(["method", "d"])
    df.to_csv(os.path.join(cfg["OUTDIR"], "results_table.csv"), index=False)

    # Plots
    for metric, fname in [("PSNR", "psnr_vs_d.png"), ("SSIM", "ssim_vs_d.png"), ("Accuracy", "acc_vs_d.png")]:
        plt.figure(figsize=(6.5, 4))
        for meth, marker in [
            ("SharedPCA-Linear", "o"),
            ("SharedPCA-MLP", "^"),
            ("Separate(LDA+PCA)", "s"),
            ("Separate(PCA+PCA)", "x"),
            ("SharedPCA-ROT", "D"),
            ("Harmonic(DCT)", "v"),
        ]:
            sub = df[df["method"] == meth].sort_values("d")
            if not len(sub):
                continue
            plt.plot(sub["d"], sub[metric], marker=marker, label=meth)
        plt.xlabel("latent dimension d")
        plt.ylabel(metric)
        plt.title(f"{metric} vs d (shared vs separate)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg["OUTDIR"], fname), dpi=160)
        plt.close()

    # Pareto: Accuracy vs PSNR
    plt.figure(figsize=(6.5, 5))
    for meth, mark in [("SharedPCA-MLP", "o"), ("Separate(LDA+PCA)", "s"), ("Separate(PCA+PCA)", "x"), ("Harmonic(DCT)", "v")]:
        sub = df[df["method"] == meth]
        if not len(sub):
            continue
        plt.scatter(sub["PSNR"], sub["Accuracy"], marker=mark, label=meth)
    plt.xlabel("PSNR")
    plt.ylabel("Accuracy")
    plt.title("Pareto: Accuracy vs PSNR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUTDIR"], "pareto_frontier.png"), dpi=160)
    plt.close()

    # Simple break-even sketch
    C_store, C_read, C_decode = 1.0, 0.02, 0.2
    C_prep, C_readout = 1.2, 0.03
    ks = np.arange(1, 201)
    C_class = C_store + ks * (C_read + C_decode)
    C_shared = C_prep + ks * C_readout
    k_star = int(np.argmin(np.abs(C_shared - C_class))) + 1
    plt.figure(figsize=(6.5, 4))
    plt.plot(ks, C_class, label="Classical: store+decode per use")
    plt.plot(ks, C_shared, label="Shared latent: prepare once + readouts")
    plt.axvline(k_star, ls="--", label=f"k* ≈ {k_star}")
    plt.xlabel("number of uses per item (k)")
    plt.ylabel("cost (arb. units)")
    plt.legend()
    plt.title("Systems-level break-even")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUTDIR"], "breakeven_inset.png"), dpi=160)
    plt.close()

    print("[done] results saved to", cfg["OUTDIR"])

if __name__ == "__main__":
    run_all()

