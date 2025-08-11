# Label-scarcity ablation for basis-multiplexed latents.
# Trains the shared PCA code unsupervised on all train images,
# then limits the number of labels available to the classifier (ρ ∈ {1.0, 0.5, 0.2, 0.1}).
# Baseline: Separate(LDA+PCA) learns LDA only from the ρ-fraction labeled subset.

import os, glob, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import load_from_zip

plt.rcParams["figure.dpi"] = 120

# --- config ---
ZIP = os.environ.get("QLST_ZIP", None)
OUTDIR = os.environ.get("QLST_OUTDIR", "results")
PER_CLASS = int(os.environ.get("QLST_PER_CLASS", "120"))
TEST_SIZE = float(os.environ.get("QLST_TEST_SIZE", "0.2"))
SEED = int(os.environ.get("QLST_SEED", "101"))
D = int(os.environ.get("QLST_LABEL_ABLATION_D", "32"))
RHO_LIST = [1.0, 0.5, 0.2, 0.1]  # label fractions to test

def auto_find_zip(default_hint="data"):
    cand = []
    for folder in [default_hint, ".", "./data", "../data"]:
        cand += glob.glob(os.path.join(folder, "*.zip"))
    if not cand: return None
    for p in cand:
        if os.path.basename(p).lower().startswith("pixelart"): return p
    return cand[0]

def stratified_take(y, frac, rng):
    """Return indices of a stratified subset with fraction 'frac' per class (min 2 per class)."""
    idxs = []
    y = np.asarray(y)
    classes = np.unique(y)
    for c in classes:
        ix = np.flatnonzero(y == c)
        n = len(ix)
        k = max(2, int(np.floor(frac * n)))
        if k > n: k = n
        take = rng.choice(ix, size=k, replace=False)
        idxs.extend(take.tolist())
    return np.array(sorted(idxs), dtype=int)


def make_readout(n_labeled):
    # Few labels → ridge is very stable; otherwise LR-CV finds a good C automatically.
    if n_labeled < 80:   # ~≤16 per class in your setting
        return RidgeClassifier(alpha=1.0)   # fast, robust with tiny data
    else:
        return LogisticRegressionCV(Cs=np.logspace(-2,2,9), cv=3,
                                    class_weight='balanced', max_iter=10000,
                                    solver='lbfgs', n_jobs=-1)

def main():
    rng = np.random.RandomState(SEED)

    if ZIP is None:
        auto = auto_find_zip("data")
        if auto is None:
            raise SystemExit("No dataset zip; set QLST_ZIP or put pixelart.zip under ./data")
        zpath = auto
    else:
        zpath = ZIP

    os.makedirs(OUTDIR, exist_ok=True)

    print("[load] dataset from", zpath)
    X, y = load_from_zip(zpath, per_class=PER_CLASS, size=(32,32), seed=SEED)
    N, H, W = X.shape
    Xf = X.reshape(N, -1)

    Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    n_classes = len(np.unique(y))

    # --- Shared code: PCA fit on ALL train images (unsupervised, fixed across rhos)
    pca = PCA(n_components=D, svd_solver="randomized", whiten=True, random_state=0).fit(Xtr)
    Ztr = pca.transform(Xtr)
    Zte = pca.transform(Xte)

    # --- Separate(LDA+PCA): recon PCA fit unsupervised (fixed); LDA will see only labeled subset
    d_cls = min(n_classes - 1, max(2, D // 2))
    d_rec = max(1, D - d_cls)
    pca_r = PCA(n_components=d_rec, svd_solver="randomized", whiten=True, random_state=1).fit(Xtr)

    rows = []
    for rho in RHO_LIST:
        # choose labeled subset
        lab_idx = stratified_take(ytr, rho, rng)

        # Shared: train classifier ONLY on labeled subset of Ztr
        clf_sh = make_readout(len(lab_idx))
        clf_sh.fit(Ztr[lab_idx], ytr[lab_idx])
        acc_sh = float(accuracy_score(yte, clf_sh.predict(Zte)))

        # Separate(LDA+PCA): train LDA ONLY on labeled subset, then classifier on that same subset
        lda = LDA(n_components=d_cls).fit(Xtr[lab_idx], ytr[lab_idx])
        Ztr_c = lda.transform(Xtr)
        Zte_c = lda.transform(Xte)
        clf_sp = make_readout(len(lab_idx))
        clf_sp.fit(Ztr_c[lab_idx], ytr[lab_idx])
        acc_sp = float(accuracy_score(yte, clf_sp.predict(Zte_c)))

        rows.append({"rho": rho, "d": D, "Shared(PCA)-Acc": round(acc_sh, 4), "Separate(LDA+PCA)-Acc": round(acc_sp, 4)})

        print(f"[rho={rho:.2f}] Shared-Acc={acc_sh:.4f} | Separate(LDA+PCA)-Acc={acc_sp:.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTDIR, f"label_scarcity_d{D}.csv")
    df.to_csv(csv_path, index=False)
    print("[save]", csv_path)

    # Plot
    plt.figure(figsize=(6.5,4))
    plt.plot(df["rho"], df["Shared(PCA)-Acc"], marker="o", label="Shared(PCA) readout")
    plt.plot(df["rho"], df["Separate(LDA+PCA)-Acc"], marker="s", label="Separate(LDA+PCA)")
    plt.gca().invert_xaxis()  # visually: harder (left) -> easier (right)
    plt.xlabel("label fraction ρ (train)  ⟵ fewer labels")
    plt.ylabel("test accuracy")
    plt.title(f"Label-Scarcity (d={D})")
    plt.legend()
    out_png = os.path.join(OUTDIR, f"label_scarcity_d{D}.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
    print("[save]", out_png)

if __name__ == "__main__":
    main()
