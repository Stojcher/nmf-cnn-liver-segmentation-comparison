"""
Shared CNN core for the CNN-only and NMF+CNN cascade arms.

ONE code path trains/evaluates both arms; they differ ONLY in whether an NMF ROI
is supplied (roi_provider):
    roi_provider=None            -> CNN-only   (full body-masked frame)
    roi_provider=nmf_roi.roi_...  -> NMF+CNN    (frame cropped to the NMF ROI)
Because everything else (architecture, preprocessing, folds, loss, training,
scoring) is identical, the measured difference is attributable to NMF localization.

The U-Net / loss / augmentation are copied (not imported) from
    oldresults1/U-net Claude simplified all patients.py
and retargeted to train on GROUND-TRUTH masks under the shared 5-fold CV.
Tuned for CPU: small U-Net, 128x128 input, few epochs, early stopping, no AMP.

Nothing in the existing project is modified; we only import from
liver_nmf_segmentation (via common_data) and reuse its cached preprocessing.
"""
import os
import sys
import gc
import json
import time
import copy

import numpy as np
from skimage.transform import resize as sk_resize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd            # noqa: E402
import liver_nmf_segmentation as L  # noqa: E402  (for postprocess_mask, MIN_LIVER_PX)

import torch                        # noqa: E402
import torch.nn as nn               # noqa: E402
import torch.nn.functional as F     # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402


# ==========================================================================
#  CONFIG  (CPU-friendly defaults; override via run_cv(cfg=...))
# ==========================================================================
DEFAULT_CFG = dict(
    input_size=128,
    features=[16, 32, 64, 128],
    epochs=25,
    patience=8,
    batch_size=8,
    lr=3e-4,
    weight_decay=1e-4,
    val_frac=0.15,
    threshold=0.5,
    seed=0,
    augment=True,
)
DEVICE = "cpu"


# ==========================================================================
#  U-NET  (copied from the old self-supervised script, in_ch parameterized)
# ==========================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, features=(16, 32, 64, 128), in_ch=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        c = in_ch
        for f in features:
            self.downs.append(DoubleConv(c, f)); c = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))
        self.final = nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-(i // 2 + 1)]
            if x.shape != s.shape:
                x = F.interpolate(x, size=s.shape[2:])
            x = self.ups[i + 1](torch.cat([s, x], dim=1))
        return self.final(x)


def dice_bce(logits, targets, smooth=1.0, bce_w=0.4):
    p = torch.sigmoid(logits)
    num = (p * targets).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dloss = (1 - (2 * num + smooth) / (den + smooth)).mean()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return dloss * (1 - bce_w) + bce * bce_w


# ==========================================================================
#  CROP / RESIZE HELPERS
# ==========================================================================
def _crop(arr, bbox):
    if bbox is None:
        return arr
    r0, r1, c0, c1 = bbox
    return arr[r0:r1 + 1, c0:c1 + 1]


def _to_size(img, size, order):
    aa = (order == 1)
    return sk_resize(img, (size, size), order=order, preserve_range=True,
                     anti_aliasing=aa).astype(np.float32)


# ==========================================================================
#  DATASET  (crop+resize precomputed once; light augmentation on the fly)
# ==========================================================================
class SliceDS(Dataset):
    def __init__(self, samples, input_size, augment):
        """samples: list of dict(img, gt, bbox, extra).  Precompute cropped+resized.

        Each input is stacked (C, S, S): channel 0 = T2 intensity, channels 1..k =
        optional NMF component maps (s['extra'] of shape (H,W,k) or None).
        """
        self.augment = augment
        self.inps, self.masks = [], []
        for s in samples:
            bbox = s["bbox"]
            base = np.clip(_to_size(_crop(s["img"], bbox), input_size, 1), 0, 1)
            chans = [base.astype(np.float32)]
            extra = s.get("extra")
            if extra is not None:
                for j in range(extra.shape[-1]):
                    chans.append(_to_size(_crop(extra[..., j], bbox), input_size, 1))
            self.inps.append(np.stack(chans, 0).astype(np.float32))   # (C,S,S)
            gt = _to_size(_crop(s["gt"].astype(np.float32), bbox), input_size, 0)
            self.masks.append((gt > 0.5).astype(np.float32))

    @property
    def n_channels(self):
        return self.inps[0].shape[0] if self.inps else 1

    def __len__(self):
        return len(self.inps)

    def __getitem__(self, i):
        inp, gt = self.inps[i], self.masks[i]           # inp (C,S,S), gt (S,S)
        if self.augment:
            if np.random.rand() < 0.5:
                inp, gt = inp[:, :, ::-1].copy(), gt[:, ::-1].copy()
            if np.random.rand() < 0.5:
                inp, gt = inp[:, ::-1, :].copy(), gt[::-1, :].copy()
            # intensity jitter only on the T2 channel; NMF maps are structural
            gain = np.float32(np.random.uniform(0.85, 1.15))
            bias = np.float32(np.random.uniform(-0.08, 0.08))
            inp = inp.copy()
            inp[0] = np.clip(inp[0] * gain + bias, 0, 1)
        return torch.from_numpy(inp), torch.from_numpy(gt[None])


# ==========================================================================
#  TRAIN / PREDICT
# ==========================================================================
def _build_samples(slices_by_patient, patients, rois, channels=None):
    samples = []
    for p in patients:
        bbox = rois.get(p) if rois else None
        ex_list = channels.get(p) if channels else None
        for i, s in enumerate(slices_by_patient[p]):
            samples.append(dict(img=s["img"], gt=s["gt"], bbox=bbox,
                                extra=(ex_list[i] if ex_list is not None else None)))
    return samples


def train_unet(train_samples, cfg, log=print):
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(train_samples))
    n_val = max(1, int(cfg["val_frac"] * len(train_samples)))
    val_idx, tr_idx = set(idx[:n_val].tolist()), idx[n_val:]
    tr = [train_samples[i] for i in tr_idx]
    va = [train_samples[i] for i in sorted(val_idx)]

    tr_ds = SliceDS(tr, cfg["input_size"], augment=cfg["augment"])
    va_ds = SliceDS(va, cfg["input_size"], augment=False)
    tr_ld = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    torch.manual_seed(cfg["seed"])
    model = UNet(features=tuple(cfg["features"]), in_ch=tr_ds.n_channels).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       factor=0.5, patience=4)
    best_val, best_state, wait = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); tr_loss = 0.0
        for x, y in tr_ld:
            opt.zero_grad(set_to_none=True)
            loss = dice_bce(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(tr_ds)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x, y in va_ld:
                val_loss += dice_bce(model(x), y).item() * x.size(0)
        val_loss /= len(va_ds)
        sched.step(val_loss)

        improved = val_loss < best_val - 1e-4
        if improved:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        log(f"      epoch {ep:3d}/{cfg['epochs']}  tr={tr_loss:.4f}  "
            f"val={val_loss:.4f}  [{time.time()-t0:.0f}s]{'  *' if improved else ''}")
        if wait >= cfg["patience"]:
            log(f"      early stop at epoch {ep} (best val {best_val:.4f})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_patient(model, slices, bbox, cfg, extras=None):
    """Return (preds, gts): native-shape bool masks for each slice of a patient.

    extras: optional list of (H,W,k) NMF channel maps aligned with `slices`.
    """
    model.eval()
    preds, gts = [], []
    size = cfg["input_size"]
    with torch.no_grad():
        for idx, s in enumerate(slices):
            crop = _crop(s["img"], bbox)
            chans = [_to_size(crop, size, 1)]
            if extras is not None:
                ex = extras[idx]
                for j in range(ex.shape[-1]):
                    chans.append(_to_size(_crop(ex[..., j], bbox), size, 1))
            x = torch.from_numpy(np.stack(chans, 0)[None])   # (1,C,S,S)
            prob = torch.sigmoid(model(x)).cpu().numpy()[0, 0]
            # resize prob back to the crop's native size, then place in the frame
            prob_native = sk_resize(prob, crop.shape, order=1, preserve_range=True,
                                    anti_aliasing=True).astype(np.float32)
            full = np.zeros(s["shape"], np.float32)
            if bbox is None:
                full = prob_native
            else:
                r0, r1, c0, c1 = bbox
                full[r0:r1 + 1, c0:c1 + 1] = prob_native
            raw = full >= cfg["threshold"]
            mask = L.postprocess_mask(raw) if raw.sum() >= L.MIN_LIVER_PX \
                else np.zeros(s["shape"], bool)
            preds.append(mask)
            gts.append(s["gt"])
    return preds, gts


# ==========================================================================
#  CROSS-VALIDATION (same folds as the NMF arm)
# ==========================================================================
def run_cv(roi_provider=None, channel_provider=None, cfg=None, arm_name="cnn",
           folds_to_run=None, fresh=False, log=print):
    """Train + evaluate a CNN arm under the shared 5-fold patient-level CV.

    roi_provider     : None or callable(slices, patient) -> bbox   (cascade arm)
    channel_provider : None or callable(slices, patient) -> list of (H,W,k) maps
                       (input-channel fusion arm; adds k extra input channels)
    Writes per-patient Dice to comparison/results/<arm_name>.json and returns it.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    log(f"[{arm_name}] cfg={cfg}")

    slices = cd.load_slices(cd.ALL20)
    rois = None
    if roi_provider is not None:
        log(f"[{arm_name}] computing NMF ROIs ...")
        rois = {p: roi_provider(slices[p], p) for p in cd.ALL20}
    chans = None
    if channel_provider is not None:
        log(f"[{arm_name}] computing NMF channel maps ...")
        chans = {p: channel_provider(slices[p], p) for p in cd.ALL20}

    in_ch = 1 + (chans[cd.ALL20[0]][0].shape[-1] if chans else 0)
    folds = cd.make_folds()
    per = {}
    for i, test_p in enumerate(folds):
        if folds_to_run is not None and i not in folds_to_run:
            continue
        train_p = [p for p in cd.ALL20 if p not in test_p]
        log(f"\n[{arm_name}] fold {i+1}/{len(folds)}  test={test_p}")

        mpath = os.path.join(cd.MODELS_DIR, f"{arm_name}_fold{i}.pt")
        model = UNet(features=tuple(cfg["features"]), in_ch=in_ch).to(DEVICE)
        if os.path.exists(mpath) and not fresh:
            model.load_state_dict(torch.load(mpath, map_location=DEVICE))
            log(f"    loaded cached model {os.path.basename(mpath)}")
        else:
            samples = _build_samples(slices, train_p, rois, chans)
            log(f"    training on {len(samples)} slices from {len(train_p)} patients "
                f"(in_ch={in_ch}) ...")
            model = train_unet(samples, cfg, log=log)
            torch.save(model.state_dict(), mpath)

        for p in test_p:
            preds, gts = predict_patient(model, slices[p],
                                         (rois.get(p) if rois else None), cfg,
                                         extras=(chans.get(p) if chans else None))
            per[p] = cd.volumetric_dice(preds, gts)
            log(f"    patient {p}: full-vol {per[p][0]:.3f}  liver {per[p][1]:.3f}")
        del model
        gc.collect()

    _save_results(arm_name, per, cfg)
    _summary(arm_name, per, log)
    return per


def _save_results(arm_name, per, cfg):
    path = os.path.join(cd.RESULTS_DIR, f"{arm_name}.json")
    blob = dict(arm=arm_name, cfg=cfg,
                per_patient={p: dict(full_vol=v[0], liver_vol=v[1])
                             for p, v in per.items()})
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"[{arm_name}] wrote {path}")


def _summary(arm_name, per, log):
    if not per:
        return
    full = np.array([per[p][0] for p in per])
    liv = np.array([per[p][1] for p in per])
    log(f"\n[{arm_name}] === {len(per)} patients ===")
    log(f"    full-volume Dice : {full.mean():.3f} +/- {full.std():.3f} "
        f"(median {np.median(full):.3f}, min {full.min():.3f})")
    log(f"    liver-only  Dice : {np.nanmean(liv):.3f} +/- {np.nanstd(liv):.3f}")
    log(f"    patients >= 0.70 full-vol : {(full >= 0.70).sum()}/{len(per)}")
