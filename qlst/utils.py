import io, os, re, zipfile
import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def _is_image(path):
    return os.path.splitext(path)[1].lower() in IMG_EXTS

def _load_img_from_zip(zf, path, size=(32, 32)):
    with zf.open(path) as f:
        im = Image.open(io.BytesIO(f.read())).convert("L").resize(size, Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0

def _index_by_numeric_token(paths):
    """Map last number in filename to that path: image_123.JPEG -> key 123."""
    pat = re.compile(r"(\d+)")
    idx = {}
    for p in paths:
        base = os.path.basename(p)
        m = list(pat.finditer(base))
        if not m:
            continue
        gid = int(m[-1].group(1))
        idx.setdefault(gid, p)
    return idx

def _parse_label_string(lbl):
    """labels.csv has one-hot strings like '[1. 0. 0. 0. 0.]' -> class id."""
    if not isinstance(lbl, str):
        try:
            return int(lbl)
        except Exception:
            return None
    s = lbl.strip()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) >= 2:
        arr = np.array([float(x) for x in nums], dtype=float)
        return int(np.argmax(arr))
    try:
        return int(s)
    except Exception:
        return None

def load_from_zip(zip_path, per_class=80, size=(32, 32), seed=1337, max_rows=500000):
    """Loads a balanced subset from your pixelart.zip."""
    import pandas as pd
    rng = np.random.RandomState(seed)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        images = [n for n in names if _is_image(n)]
        id_map = _index_by_numeric_token(images)

        # find labels.csv
        label_name = None
        for n in names:
            if n.lower().endswith("labels.csv"):
                label_name = n
                break
        if label_name is None:
            raise FileNotFoundError("labels.csv not found inside the zip")

        df = pd.read_csv(io.StringIO(z.read(label_name).decode("utf-8", errors="replace")))
        cols = {c.lower().strip(): c for c in df.columns}
        if "image index" in cols and "label" in cols:
            x_idx = df[cols["image index"]].astype(int)
            y_idx = df[cols["label"]].apply(_parse_label_string).astype(int)
            pairs = []
            for i, lab in zip(x_idx, y_idx):
                # image indices in labels.csv are 1-based; try i then i-1
                p = id_map.get(i) or id_map.get(i - 1)
                if p is not None:
                    pairs.append((p, lab))
        else:
            raise RuntimeError("Unsupported labels.csv format")

        if not pairs:
            raise RuntimeError("No (path,label) pairs could be built")

        import pandas as pd
        df_pairs = pd.DataFrame(pairs, columns=["path", "cls"])

        # balanced sample
        picks = []
        for c in sorted(df_pairs["cls"].unique()):
            sub = df_pairs[df_pairs["cls"] == c]
            take = min(per_class, len(sub))
            if take >= 1:
                picks += list(sub.sample(take, random_state=rng).index)
        df_sub = df_pairs.loc[picks].reset_index(drop=True)

        X_list, y_list = [], []
        for _, r in df_sub.iterrows():
            try:
                X_list.append(_load_img_from_zip(z, r["path"], size=size))
                y_list.append(int(r["cls"]))
            except Exception:
                pass

    if not X_list:
        raise RuntimeError("After balancing/reading, no images were loaded.")
    X = np.stack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y

# --- metrics & helpers ---
def psnr(a, b, eps=1e-12):
    import math
    mse = float(np.mean((a - b) ** 2))
    if mse < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(size) - size // 2
    k = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return k

def conv2_separable(img, k):
    pad = len(k) // 2
    tmp = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    tmp2 = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, tmp)
    tmp2 = np.pad(tmp2, ((pad, pad), (0, 0)), mode="reflect")
    out = np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, tmp2)
    return out

def ssim_batch(X_true, X_pred, count=40):
    k = gaussian_kernel(5, 1.0)
    m = min(count, X_true.shape[0])
    vals = []
    for i in range(m):
        a = X_true[i]
        b = X_pred[i]
        mu1 = conv2_separable(a, k)
        mu2 = conv2_separable(b, k)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        a2 = a * a
        b2 = b * b
        ab = a * b
        s1 = conv2_separable(a2, k) - mu1_sq
        s2 = conv2_separable(b2, k) - mu2_sq
        s12 = conv2_separable(ab, k) - mu1_mu2
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        num = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
        vals.append(float(np.mean(num / (den + 1e-12))))
    return float(np.mean(vals))

def make_montage(rows):
    R = len(rows)
    C = len(rows[0]) if R > 0 else 0
    if R == 0 or C == 0:
        return None
    H, W = rows[0][0].shape
    canvas = np.zeros((R * H, C * W), dtype=np.float32)
    for r in range(R):
        for c in range(C):
            canvas[r * H : (r + 1) * H, c * W : (c + 1) * W] = rows[r][c]
    return canvas

