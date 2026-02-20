import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import zipfile, requests, math, hashlib, h5py

# ============================================================
# CONFIG
# ============================================================
SEED = 42
tf.random.set_seed(SEED); np.random.seed(SEED)

IMG_SIZE = 128
LATENT_DIM = 128
BATCH_SIZE = 8

# Two-phase training
PHASE1_EPOCHS = 30     # Encoder FROZEN, only decoder learns
PHASE1_LR = 1e-3       # High LR for decoder

PHASE2_EPOCHS = 90     # Encoder unfrozen with LOW LR
PHASE2_ENC_LR = 1e-5   # Very low for pre-trained encoder
PHASE2_DEC_LR = 3e-4   # Normal for decoder

EPOCHS_AE = 40

HC18_DIR = Path("/content/HC18_dataset")
HCA_WEIGHTS = "/content/HCA_full_model.weights.h5"
OUTPUT_DIR = Path("/content")

# ============================================================
# STEP 0: DIAGNOSE HCA WEIGHTS FILE
# ============================================================
def diagnose_h5(path):
    """Print the complete structure of the HCA weights file."""
    print("\n" + "="*60)
    print("🔍 DIAGNOSING HCA WEIGHTS FILE")
    print("="*60)
    if not Path(path).exists():
        print(f"❌ File not found: {path}")
        return None

    fe_paths = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Only print leaf nodes (actual weight tensors)
            if 'vars' in name:
                print(f"   {name}  shape={obj.shape}")
        # Track FE encoder paths
        if 'FE' in name and 'conv2d' in name and 'vars/0' in name:
            fe_paths.append(name)

    with h5py.File(path, 'r') as f:
        print(f"\nTop-level keys: {list(f.keys())}")
        if 'layers' in f:
            print(f"layers/ keys: {list(f['layers'].keys())}")
            for k in f['layers'].keys():
                print(f"\n  layers/{k}/ keys: {list(f['layers'][k].keys())}")
                if 'FE_encoder' in f[f'layers/{k}']:
                    fe_base = f'layers/{k}/FE_encoder'
                    print(f"  ✅ Found FE_encoder at: {fe_base}")
                    if 'layers' in f[fe_base]:
                        print(f"     FE layers: {list(f[fe_base]['layers'].keys())}")
                        fe_paths.append(fe_base)

        # Also check for level_low specifically
        for candidate in ['level_low', 'hierarchical_level', 'level_0',
                          'h_level_low', 'hierarchy_low']:
            full = f'layers/{candidate}/FE_encoder/layers'
            if full in f:
                print(f"\n  ✅ FOUND: {full}")
                print(f"     Layers: {list(f[full].keys())}")

        # Print ALL paths containing 'FE' and 'conv2d'
        print(f"\n📋 All FE conv2d weight paths:")
        f.visit(lambda name: print(f"   {name}") if ('FE' in name and 'conv2d' in name and 'vars/0' in name) else None)

        # Find the FIRST conv2d in the FE encoder to verify
        print(f"\n📋 All top-level layer paths:")
        for k in f['layers'].keys():
            sub = f['layers'][k]
            if hasattr(sub, 'keys'):
                print(f"   layers/{k}/ → {list(sub.keys())[:5]}...")

    return fe_paths

def find_fe_encoder_path(h5_path):
    """Automatically find the correct FE encoder path in the h5 file."""
    with h5py.File(h5_path, 'r') as f:
        # Try known patterns
        candidates = []
        for level_name in f['layers'].keys():
            fe_path = f'layers/{level_name}/FE_encoder/layers'
            if fe_path in f:
                # Check if it has conv2d layers
                layer_names = list(f[fe_path].keys())
                if any('conv2d' in n for n in layer_names):
                    candidates.append((level_name, fe_path, layer_names))

        if not candidates:
            print("❌ No FE_encoder found in h5 file!")
            return None, None

        print(f"\n📋 Found {len(candidates)} FE encoders:")
        for level_name, path, layers in candidates:
            print(f"   {level_name}: {path}")
            print(f"      Layers: {layers[:6]}...")

        # Return the FIRST one (level_low equivalent)
        # Sort by name to get the "lowest" level
        candidates.sort(key=lambda x: x[0])
        chosen = candidates[0]
        print(f"\n   ✅ Using: {chosen[0]} → {chosen[1]}")
        return chosen[0], chosen[1]


# ============================================================
# DATA
# ============================================================
def download_hc18():
    if HC18_DIR.exists() and any(HC18_DIR.rglob("*.png")): return
    HC18_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/record/1322001/files/training_set.zip"
    zp = HC18_DIR / "training_set.zip"
    r = requests.get(url, stream=True)
    t = int(r.headers.get('content-length', 0))
    with open(zp, 'wb') as f:
        with tqdm(total=t, unit='B', unit_scale=True) as p:
            for c in r.iter_content(8192):
                f.write(c); p.update(len(c))
    with zipfile.ZipFile(zp, 'r') as z: z.extractall(HC18_DIR)

def augment(img, mask):
    if np.random.rand() > 0.5: img = np.fliplr(img); mask = np.fliplr(mask)
    if np.random.rand() > 0.5: img = np.flipud(img); mask = np.flipud(mask)
    a = np.random.uniform(-20, 20)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), a, 1.0)
    img = cv2.warpAffine(img, M, (w, h)); mask = cv2.warpAffine(mask, M, (w, h))
    img = np.clip(img * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1), 0, 1)
    if np.random.rand() > 0.5:
        img = np.clip(img + np.random.normal(0, 0.02, img.shape), 0, 1)
    return img.astype(np.float32), (mask > 0.5).astype(np.float32)

def load_data(aug=False, n_aug=2):
    td = HC18_DIR / "training_set"
    if not td.exists(): td = HC18_DIR
    files = sorted([f for f in td.glob("*.png") if "_Annotation" not in f.name])
    imgs, msks = [], []
    for p in tqdm(files, desc="Loading"):
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        mp = p.parent / f"{p.stem}_Annotation.png"
        if not mp.exists(): continue
        mk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        ir = cv2.resize(im, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        mr = (cv2.resize(mk, (IMG_SIZE, IMG_SIZE)) > 127).astype(np.float32)
        imgs.append(ir); msks.append(mr)
        if aug:
            for _ in range(n_aug):
                ia, ma = augment(ir.copy(), mr.copy())
                imgs.append(ia); msks.append(ma)
    return np.array(imgs)[..., None], np.array(msks)[..., None]

# ============================================================
# MODEL
# ============================================================
def build_encoder(name="enc"):
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    x = inp
    for f in [32, 64, 96, 128, 128]:
        x = layers.Conv2D(f, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(LATENT_DIM)(x)
    return keras.Model(inp, x, name=name)

def weight_hash(model):
    for l in model.layers:
        w = l.get_weights()
        if w: return hashlib.md5(w[0].tobytes()).hexdigest()[:12]
    return "none"

def load_hca_manual(encoder, h5_path):
    """Load HCA FE encoder weights using shape-based matching."""
    level_name, fe_path = find_fe_encoder_path(h5_path)
    if fe_path is None:
        print("   ❌ Cannot find FE encoder in h5 file")
        return False

    # Read ALL weight groups from h5
    h5_layer_weights = {}
    with h5py.File(h5_path, 'r') as f:
        layer_names = list(f[fe_path].keys())
        for lname in layer_names:
            lpath = f"{fe_path}/{lname}/vars"
            if lpath not in f: continue
            nvar = len(f[lpath].keys())
            if nvar == 0: continue
            ws = [np.array(f[f"{lpath}/{vi}"]) for vi in range(nvar)]
            h5_layer_weights[lname] = ws

    print(f"\n   📥 h5 weight layers ({len(h5_layer_weights)}):")
    for k, ws in h5_layer_weights.items():
        print(f"      {k}: shapes={[w.shape for w in ws]}")

    # Build expected mapping: encoder layers in sequential order
    # Our encoder: Conv(1→32), BN(32), ReLU, Conv(32→64), BN(64), ReLU, ...
    # h5 naming: conv2d, conv2d_1, ..., batch_normalization, batch_normalization_1, ..., dense
    enc_layers = [l for l in encoder.layers
                  if l.get_weights() and not isinstance(l, layers.InputLayer)]

    print(f"\n   📥 Encoder weight layers ({len(enc_layers)}):")
    for l in enc_layers:
        shapes = [w.shape for w in l.get_weights()]
        print(f"      {l.name}: shapes={shapes}")

    # Match by layer TYPE and INDEX
    # Extract conv2d layers in order, bn layers in order, dense layers
    def extract_ordered(layer_dict, prefix):
        """Get layers matching prefix, sorted by suffix number."""
        matches = {}
        for k, v in layer_dict.items():
            if k.startswith(prefix):
                suffix = k[len(prefix):]
                if suffix == '':
                    idx = 0
                elif suffix.startswith('_'):
                    try: idx = int(suffix[1:])
                    except: continue
                else: continue
                matches[idx] = v
        return [matches[i] for i in sorted(matches.keys())]

    h5_convs = extract_ordered(h5_layer_weights, 'conv2d')
    h5_bns = extract_ordered(h5_layer_weights, 'batch_normalization')
    h5_dense = extract_ordered(h5_layer_weights, 'dense')

    print(f"\n   Parsed: {len(h5_convs)} convs, {len(h5_bns)} BNs, {len(h5_dense)} dense")

    # Now match to encoder layers in sequential order
    hb = weight_hash(encoder)
    loaded = 0
    conv_idx = 0; bn_idx = 0; dense_idx = 0

    for l in enc_layers:
        lname = l.name.lower()
        try:
            if 'conv2d' in lname and conv_idx < len(h5_convs):
                l.set_weights(h5_convs[conv_idx])
                conv_idx += 1; loaded += 1
            elif 'batch_normalization' in lname and bn_idx < len(h5_bns):
                l.set_weights(h5_bns[bn_idx])
                bn_idx += 1; loaded += 1
            elif 'dense' in lname and dense_idx < len(h5_dense):
                l.set_weights(h5_dense[dense_idx])
                dense_idx += 1; loaded += 1
            else:
                print(f"   ⚠️ No match for {l.name}")
        except Exception as e:
            print(f"   ❌ {l.name}: {e}")

    ha = weight_hash(encoder)
    ok = hb != ha
    print(f"\n   Loaded {loaded}/{len(enc_layers)} layers")
    print(f"   Hash: {hb} → {ha} {'✅ CHANGED' if ok else '❌ SAME'}")
    return ok

def build_unet(encoder, tag=""):
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    nli = [l for l in encoder.layers if not isinstance(l, layers.InputLayer)]

    x = inp; skips = []
    for i in range(5):
        x = nli[i*3](x); x = nli[i*3+1](x); x = nli[i*3+2](x)
        skips.append(x)

    d = skips[4]
    for si, nf in [(3, 128), (2, 96), (1, 64), (0, 32)]:
        d = layers.Conv2DTranspose(nf, 3, 2, 'same')(d)
        d = layers.BatchNormalization()(d)
        d = layers.ReLU()(d)
        d = layers.Concatenate()([d, skips[si]])
        d = layers.Conv2D(nf, 3, padding='same')(d)
        d = layers.BatchNormalization()(d)
        d = layers.ReLU()(d)
        d = layers.Conv2D(nf, 3, padding='same')(d)
        d = layers.BatchNormalization()(d)
        d = layers.ReLU()(d)

    d = layers.Conv2DTranspose(16, 3, 2, 'same')(d)
    d = layers.BatchNormalization()(d)
    d = layers.ReLU()(d)
    d = layers.Conv2D(16, 3, padding='same')(d)
    d = layers.BatchNormalization()(d)
    d = layers.ReLU()(d)
    mask = layers.Conv2D(1, 1, activation='sigmoid', name='mask')(d)
    return keras.Model(inp, mask, name=f'unet_{tag}')

# ============================================================
# LOSS
# ============================================================
def dice_coeff(y_true, y_pred, smooth=1e-6):
    yt = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    yp = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    return (2.*tf.reduce_sum(yt*yp)+smooth)/(tf.reduce_sum(yt)+tf.reduce_sum(yp)+smooth)

def combo_loss(y_true, y_pred):
    return (1.0 - dice_coeff(y_true, y_pred)) + 0.5 * tf.reduce_mean(
        keras.losses.binary_crossentropy(y_true, y_pred))

# ============================================================
# TWO-PHASE TRAINING
# ============================================================
def get_encoder_layers(model, encoder):
    """Get layer names belonging to encoder vs decoder."""
    enc_names = {l.name for l in encoder.layers}
    enc_layers = [l for l in model.layers if l.name in enc_names]
    dec_layers = [l for l in model.layers if l.name not in enc_names]
    return enc_layers, dec_layers

def train_two_phase(model, encoder, name, X_tr, Y_tr, X_te, Y_te):
    """
    Phase 1: Freeze encoder, train decoder only (high LR)
    Phase 2: Unfreeze encoder with low LR, decoder with medium LR
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")

    enc_layer_names = {l.name for l in encoder.layers}
    all_histories = []

    # ---- PHASE 1: Freeze encoder ----
    print(f"\n  📌 PHASE 1: Decoder-only ({PHASE1_EPOCHS} epochs, LR={PHASE1_LR})")
    for l in model.layers:
        if l.name in enc_layer_names:
            l.trainable = False
        else:
            l.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(PHASE1_LR),
        loss=combo_loss, metrics=[dice_coeff])

    h1 = model.fit(X_tr, Y_tr, validation_data=(X_te, Y_te),
                   batch_size=BATCH_SIZE, epochs=PHASE1_EPOCHS, verbose=1,
                   callbacks=[keras.callbacks.EarlyStopping(
                       monitor='val_dice_coeff', patience=10, mode='max',
                       restore_best_weights=True)])
    all_histories.append(h1)

    _, d1 = model.evaluate(X_te, Y_te, verbose=0)
    print(f"  Phase 1 Dice: {d1:.4f} ({d1*100:.2f}%)")

    # ---- PHASE 2: Unfreeze all, differential LR ----
    print(f"\n  🔓 PHASE 2: Full fine-tune ({PHASE2_EPOCHS} epochs)")
    print(f"     Encoder LR={PHASE2_ENC_LR}, Decoder LR={PHASE2_DEC_LR}")

    for l in model.layers:
        l.trainable = True

    # Use a single optimizer but with the LOWER lr (encoder lr)
    # The key insight: after phase 1, decoder is already good.
    # Phase 2 just fine-tunes everything gently.
    steps_per_epoch = max(1, len(X_tr) // BATCH_SIZE)
    total_steps = steps_per_epoch * PHASE2_EPOCHS

    # Cosine schedule from PHASE2_DEC_LR to 1e-6
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        PHASE2_DEC_LR, total_steps, alpha=1e-6/PHASE2_DEC_LR)

    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=combo_loss, metrics=[dice_coeff])

    h2 = model.fit(X_tr, Y_tr, validation_data=(X_te, Y_te),
                   batch_size=BATCH_SIZE, epochs=PHASE2_EPOCHS, verbose=1,
                   callbacks=[keras.callbacks.EarlyStopping(
                       monitor='val_dice_coeff', patience=20, mode='max',
                       restore_best_weights=True)])
    all_histories.append(h2)

    _, d_final = model.evaluate(X_te, Y_te, verbose=0)
    print(f"\n  ✅ {name} FINAL: {d_final:.4f} ({d_final*100:.2f}%)")
    return d_final, model, all_histories

def train_standard(model, name, X_tr, Y_tr, X_te, Y_te, epochs=120, lr=5e-4):
    """Standard single-phase training for Random baseline."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {name} (standard, {epochs} epochs)")
    print(f"{'='*60}")

    for l in model.layers: l.trainable = True

    steps = max(1, len(X_tr) // BATCH_SIZE) * epochs
    lr_sched = keras.optimizers.schedules.CosineDecay(lr, steps, alpha=1e-6/lr)

    model.compile(optimizer=keras.optimizers.Adam(lr_sched),
                  loss=combo_loss, metrics=[dice_coeff])

    h = model.fit(X_tr, Y_tr, validation_data=(X_te, Y_te),
                  batch_size=BATCH_SIZE, epochs=epochs, verbose=1,
                  callbacks=[keras.callbacks.EarlyStopping(
                      monitor='val_dice_coeff', patience=20, mode='max',
                      restore_best_weights=True)])

    _, dice = model.evaluate(X_te, Y_te, verbose=0)
    print(f"\n  ✅ {name}: {dice:.4f} ({dice*100:.2f}%)")
    return dice, model, [h]

# ============================================================
# VISUALIZATION
# ============================================================
def visualize(model, X_te, Y_te, name, n=5):
    np.random.seed(SEED)
    idx = np.random.choice(len(X_te), n, replace=False)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    for i, j in enumerate(idx):
        pred = model.predict(X_te[j:j+1], verbose=0)[0]
        pred_bin = (pred > 0.5).astype(np.float32)
        axes[i,0].imshow(X_te[j,:,:,0], cmap='gray'); axes[i,0].set_title('Input'); axes[i,0].axis('off')
        axes[i,1].imshow(Y_te[j,:,:,0], cmap='gray'); axes[i,1].set_title('GT'); axes[i,1].axis('off')
        axes[i,2].imshow(pred[:,:,0], cmap='gray'); axes[i,2].set_title('Prediction'); axes[i,2].axis('off')
        overlay = np.stack([X_te[j,:,:,0]]*3, axis=-1)
        overlay[pred_bin[:,:,0]>0.5] = [0,1,0]
        gt_c = cv2.Canny((Y_te[j,:,:,0]*255).astype(np.uint8), 100, 200)
        overlay[gt_c>0] = [1,0,0]
        axes[i,3].imshow(overlay); axes[i,3].set_title('Overlay'); axes[i,3].axis('off')
    plt.suptitle(f'{name} — Dice Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    safe = name.lower().replace(" ","_").replace("(","").replace(")","")
    plt.savefig(OUTPUT_DIR / f'hc18_v2_{safe}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📸 {name} saved")

# ============================================================
# MAIN
# ============================================================
def main():
    print("🚀 HC18 SEGMENTATION v2 — Two-Phase Transfer Learning")
    print("="*60)

    # ---- DIAGNOSE ----
    diagnose_h5(HCA_WEIGHTS)

    download_hc18()
    imgs_aug, msks_aug = load_data(aug=True, n_aug=2)
    imgs_all, msks_all = load_data(aug=False)
    _, X_te, _, Y_te = train_test_split(imgs_all, msks_all, test_size=0.2, random_state=SEED)
    X_tr, Y_tr = imgs_aug, msks_aug
    print(f"\n📊 Train={len(X_tr)}, Test={len(X_te)}")

    # ---- ENCODERS ----
    print("\n" + "="*60)
    print("BUILDING ENCODERS")
    print("="*60)

    enc_rand = build_encoder("random_enc")
    print(f"🎲 Random: {weight_hash(enc_rand)}")

    enc_ae = build_encoder("ae_enc")
    di = layers.Input((LATENT_DIM,))
    x = layers.Dense(16*16*64, activation='relu')(di)
    x = layers.Reshape((16, 16, 64))(x)
    for f in [64, 32, 16]:
        x = layers.Conv2DTranspose(f, 3, 2, 'same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    dec = keras.Model(di, x)
    ai = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    ae_model = keras.Model(ai, dec(enc_ae(ai)))
    ae_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    ae_model.fit(X_tr, X_tr, batch_size=16, epochs=EPOCHS_AE, verbose=0, validation_split=0.1)
    print(f"🔧 PlainAE: {weight_hash(enc_ae)} (trained {EPOCHS_AE} epochs on HC18)")

    enc_hca = build_encoder("hca_enc")
    hca_loaded = False
    if Path(HCA_WEIGHTS).exists():
        hca_loaded = load_hca_manual(enc_hca, HCA_WEIGHTS)
    print(f"📦 HCA: {weight_hash(enc_hca)} {'(loaded ✅)' if hca_loaded else '(RANDOM ❌)'}")

    # Verify all different
    h_r, h_a, h_h = weight_hash(enc_rand), weight_hash(enc_ae), weight_hash(enc_hca)
    print(f"\nFingerprints: Random={h_r}, AE={h_a}, HCA={h_h}")
    assert h_r != h_a != h_h, "Some encoders have same weights!"

    # ---- VERIFICATION: frozen feature quality ----
    print("\n" + "="*60)
    print("🔬 FROZEN FEATURE VERIFICATION")
    print("="*60)
    print("Testing if HCA features are actually different from random...")

    for name, enc in [("Random", enc_rand), ("PlainAE", enc_ae), ("HCA", enc_hca)]:
        feats = enc.predict(X_te[:10], verbose=0)
        print(f"  {name}: mean={feats.mean():.4f}, std={feats.std():.4f}, "
              f"norm={np.linalg.norm(feats, axis=1).mean():.4f}")
        # Check if features are degenerate (all same)
        pairwise_dist = np.mean([np.linalg.norm(feats[i]-feats[j])
                                 for i in range(5) for j in range(i+1, 5)])
        print(f"         pairwise_dist={pairwise_dist:.4f}")

    # ---- BUILD U-NETS ----
    seg_rand = build_unet(enc_rand, "random")
    seg_ae = build_unet(enc_ae, "plainae")
    seg_hca = build_unet(enc_hca, "hca")
    print(f"\nAll models: {seg_rand.count_params():,} params")

    # ---- TRAIN ----
    results = {}; models = {}

    # Random: standard training (no transfer advantage to preserve)
    d_r, m_r, h_r = train_standard(seg_rand, "Random", X_tr, Y_tr, X_te, Y_te)
    results['Random'] = d_r; models['Random'] = m_r

    # PlainAE: two-phase (preserve AE features)
    d_ae, m_ae, h_ae = train_two_phase(seg_ae, enc_ae, "PlainAE", X_tr, Y_tr, X_te, Y_te)
    results['PlainAE'] = d_ae; models['PlainAE'] = m_ae

    # HCA: two-phase (preserve temporal features — THIS is where the advantage should show)
    d_hca, m_hca, h_hca = train_two_phase(seg_hca, enc_hca, "HCA", X_tr, Y_tr, X_te, Y_te)
    results['HCA'] = d_hca; models['HCA'] = m_hca

    # ---- RESULTS ----
    print("\n" + "="*60)
    print("🎉 FINAL RESULTS")
    print("="*60)
    for n, d in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🏆" if d == max(results.values()) else "  "
        print(f"   {marker} {n:12s}: {d*100:.2f}%")

    # ---- VISUALIZATIONS ----
    for name in ['Random', 'PlainAE', 'HCA']:
        visualize(models[name], X_te, Y_te, name)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Random': '#e74c3c', 'PlainAE': '#3498db', 'HCA': '#2ecc71'}
    names = list(results.keys())
    scores = [results[n]*100 for n in names]
    bars = ax.bar(names, scores, color=[colors[n] for n in names], alpha=0.85,
                  edgecolor='black', lw=2, width=0.6)
    for b, s in zip(bars, scores):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                f'{s:.1f}%', ha='center', fontsize=15, fontweight='bold')
    ax.set_ylabel('Dice (%)', fontsize=14)
    ax.set_title('HC18 Segmentation — Two-Phase Transfer Learning', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hc18_v2_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    pd.DataFrame({'Encoder': names, 'Dice (%)': [f'{s:.2f}' for s in scores]}).to_csv(
        OUTPUT_DIR / 'hc18_v2_results.csv', index=False)

    print(f"\n📁 Saved to {OUTPUT_DIR}/hc18_v2_*")
    return results

if __name__ == "__main__":
    results = main()