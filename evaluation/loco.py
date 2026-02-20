import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

# ============================================================
# 0. CONFIGURATION
# ============================================================
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Paths
ROOT         = Path("/content/dataset/preprocessed")
DIRS_TO_SCAN = [ROOT / "A_fixed", ROOT / "B_variable"]

# Trained model weight paths 
HCA_WEIGHTS_PATH = "/content/HCA_best_model.weights.h5"
AE_WEIGHTS_PATH  = "/content/AE_best_model.weights.h5"

# Hyperparameters
IMG_SIZE     = 128
T_WIN        = 64
LATENT_DIM   = 128
BATCH_SIZE   = 4
NUM_SEGMENTS = 4
SEGMENT_SIZE = T_WIN // NUM_SEGMENTS   # 16

# Loss weights (AE)
W_REC   = 1.0
W_SCALE = 1.0
W_SPEC  = 0.5
W_ADV   = 1.0
W_BATCH = 1.0
W_KL    = 0.01

# Loss weights (HCA)
W_REC_HCA    = 0.1
W_CROSS_LOW  = 0.3
W_CROSS_MID  = 0.5
W_CROSS_HIGH = 0.7
W_HIERARCHY  = 0.2

# Evaluation
MAX_CLIPS         = 500
BOOTSTRAP_ITERS   = 1000
PERMUTATION_ITERS = 500

# Centre definitions
CENTRE_MAP  = {'Merkez 2': 0, 'Merkez 6': 1, 'Merkez 8': 2}
CENTRES     = list(CENTRE_MAP.keys())
NUM_CENTRES = len(CENTRES)

# ============================================================
# 1. METADATA
# ============================================================
def create_metadata():
    """
    Scan A_fixed and B_variable for .npy clips.
    Returns a DataFrame with columns:
      centre, subject, clip, full_path_A, full_path_V, valid_exists
    """
    rows = []
    print(f"Scanning: {[d.name for d in DIRS_TO_SCAN]}")

    for root_path in DIRS_TO_SCAN:
        if not root_path.exists():
            print(f"  Warning: {root_path} not found. Skipping.")
            continue

        all_npy     = list(root_path.rglob("*.npy"))
        video_files = [f for f in all_npy if "_valid" not in f.name]
        print(f"  -> {len(video_files)} clips in {root_path.name}")

        for clip_file in video_files:
            path_str = str(clip_file)
            centre   = None
            if   "Merkez 2" in path_str: centre = "Merkez 2"
            elif "Merkez 6" in path_str: centre = "Merkez 6"
            elif "Merkez 8" in path_str: centre = "Merkez 8"

            if centre is None:
                continue

            subject    = clip_file.parent.name
            clip       = clip_file.stem
            valid_file = clip_file.parent / f"{clip}_valid.npy"

            rows.append({
                'dataset_type': root_path.name,
                'centre'      : centre,
                'subject'     : subject,
                'clip'        : clip,
                'full_path_A' : str(clip_file),
                'full_path_V' : str(valid_file),
                'valid_exists': valid_file.exists()
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No data found. Check DIRS_TO_SCAN paths.")

    print(f"\nTotal clips: {len(df)}")
    print(df.groupby('centre').size().to_string())
    return df


# ============================================================
# 2. LOCO SPLIT — PURE CENTRE-LEVEL HOLDOUT
# ============================================================
def loco_split(df, held_out_centre):
    """
    Returns:
      df_train : all subjects NOT from held_out_centre
      df_test  : all subjects from held_out_centre

    Note: subject-level integrity is guaranteed because every subject
    belongs to exactly one centre. No further filtering needed.
    """
    df_train = df[df['centre'] != held_out_centre].reset_index(drop=True)
    df_test  = df[df['centre'] == held_out_centre].reset_index(drop=True)

    print(f"\n  Held-out centre : {held_out_centre}")
    print(f"  Train clips     : {len(df_train)}  "
          f"({df_train.groupby('centre').size().to_dict()})")
    print(f"  Test  clips     : {len(df_test)}")

    return df_train, df_test


# ============================================================
# 3. MODEL BUILDING BLOCKS  
# ============================================================
def build_cnn_backbone(latent_dim, name="encoder"):
    """5-layer progressive downsampling CNN backbone."""
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    x   = inp
    for filters in [32, 64, 96, 128, 128]:
        x = layers.Conv2D(filters, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(latent_dim)(x)
    return keras.Model(inp, x, name=name)


def build_decoder(latent_dim):
    """Decoder: latent vector → frame."""
    inp = layers.Input((latent_dim,))
    x   = layers.Dense((IMG_SIZE // 8) * (IMG_SIZE // 8) * 64)(inp)
    x   = layers.Reshape((IMG_SIZE // 8, IMG_SIZE // 8, 64))(x)
    for filters in [64, 32, 16]:
        x = layers.Conv2DTranspose(filters, 3, 2, 'same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    out = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    return keras.Model(inp, out, name="decoder")


class TemporalAttention(layers.Layer):
    """Multi-head self-attention over the temporal axis."""
    def __init__(self, dim, num_heads=4, dropout=0.1, name_prefix=""):
        super().__init__(name=f"{name_prefix}_temporal_attn")
        self.mha  = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout)
        self.norm = layers.LayerNormalization()

    def call(self, z, training=False):
        return self.norm(z + self.mha(z, z, training=training))


class CrossAttention(layers.Layer):
    """Bidirectional cross-attention between FE and RE streams."""
    def __init__(self, dim, num_heads=4, dropout=0.1, name_prefix=""):
        super().__init__(name=f"{name_prefix}_cross_attn")
        self.FE_to_RE = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout)
        self.RE_to_FE = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout)
        self.norm_FE  = layers.LayerNormalization()
        self.norm_RE  = layers.LayerNormalization()
        self.gate_FE  = layers.Dense(dim, activation='sigmoid')
        self.gate_RE  = layers.Dense(dim, activation='sigmoid')

    def call(self, z_FE, z_RE, training=False):
        FE_cross     = self.FE_to_RE(query=z_FE, key=z_RE, value=z_RE, training=training)
        z_FE_updated = self.norm_FE(z_FE + self.gate_FE(z_FE) * FE_cross)
        RE_cross     = self.RE_to_FE(query=z_RE, key=z_FE, value=z_FE, training=training)
        z_RE_updated = self.norm_RE(z_RE + self.gate_RE(z_RE) * RE_cross)
        return z_FE_updated, z_RE_updated


class BatchConditionalNorm(layers.Layer):
    """Centre-conditioned normalization."""
    def __init__(self, num_centres, dim):
        super().__init__()
        self.gamma_emb = layers.Embedding(num_centres, dim)
        self.beta_emb  = layers.Embedding(num_centres, dim)

    def call(self, z, centre_id):
        eps    = 1e-6
        mean   = tf.reduce_mean(z, axis=[1, 2], keepdims=True)
        std    = tf.math.reduce_std(z, axis=[1, 2], keepdims=True)
        z_norm = (z - mean) / (std + eps)
        gamma  = self.gamma_emb(centre_id)
        beta   = self.beta_emb(centre_id)
        return gamma[:, None, :] * z_norm + beta[:, None, :]


class CentreDiscriminator(keras.Model):
    """Adversarial centre discriminator."""
    def __init__(self, num_centres):
        super().__init__(name="centre_discriminator")
        self.net = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64,  activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_centres)
        ])

    def call(self, z, training=False):
        return self.net(tf.reduce_mean(z, axis=1), training=training)


class BayesianEncoder(keras.Model):
    """Variational encoder for the RE stream."""
    def __init__(self, latent_dim):
        super().__init__(name="bayesian_encoder")
        self.backbone  = build_cnn_backbone(latent_dim * 2)
        self.latent_dim = latent_dim

    def call(self, x):
        h       = self.backbone(x)
        mu      = h[..., :self.latent_dim]
        log_var = h[..., self.latent_dim:]
        std     = tf.exp(0.5 * log_var)
        z       = mu + tf.random.normal(tf.shape(std)) * std
        return z, mu, log_var


# ============================================================
# 4a. AUTOENCODER MODEL
# ============================================================
class AutoencoderModel(keras.Model):
    """
    Temporal Autoencoder with:
      - Frame-level CNN encoder
      - Temporal self-attention
      - Centre-conditioned batch normalization
      - Adversarial centre discriminator
    """
    def __init__(self, latent_dim=LATENT_DIM, num_centres=NUM_CENTRES):
        super().__init__(name="autoencoder")
        self.encoder          = build_cnn_backbone(latent_dim, "ae_encoder")
        self.temp_attention   = TemporalAttention(latent_dim, name_prefix="ae")
        self.batch_norm       = BatchConditionalNorm(num_centres, latent_dim)
        self.decoder          = build_decoder(latent_dim)
        self.discriminator    = CentreDiscriminator(num_centres)
        self.centre_classifier = layers.Dense(num_centres, name="centre_classifier")
        self.latent_dim       = latent_dim
        self.num_centres      = num_centres

    def call(self, frames, centre_id, training=False):
        # frames: (B, T, H, W, 1)
        z_raw  = layers.TimeDistributed(self.encoder)(frames)         # (B,T,D)
        z_attn = self.temp_attention(z_raw, training=training)        # (B,T,D)
        z      = self.batch_norm(z_attn, centre_id)                   # (B,T,D)
        x_rec  = layers.TimeDistributed(self.decoder)(z)              # (B,T,H,W,1)
        return {'z': z, 'z_raw': z_raw, 'x_rec': x_rec}


# ============================================================
# 4b. HCA ENCODER MODEL  
# ============================================================
class HierarchicalLevel(keras.Model):
    """Single hierarchical level: FE + RE + cross-attention."""
    def __init__(self, latent_dim, num_centres, level_name):
        super().__init__(name=f"level_{level_name}")
        self.FE_encoder    = build_cnn_backbone(latent_dim, f"FE_{level_name}")
        self.FE_attention  = TemporalAttention(latent_dim, name_prefix=f"FE_{level_name}")
        self.RE_encoder    = BayesianEncoder(latent_dim)
        self.RE_norm       = BatchConditionalNorm(num_centres, latent_dim)
        self.cross_attn    = CrossAttention(latent_dim, name_prefix=level_name)
        self.latent_dim    = latent_dim

    def call(self, frames, centre_id, training=False):
        if len(frames.shape) == 5:   # raw frames (B,T,H,W,1)
            z_FE = layers.TimeDistributed(self.FE_encoder)(frames)
            z_FE = self.FE_attention(z_FE, training=training)

            z_RE_list, mu_list, lv_list = [], [], []
            for frame_t in tf.unstack(frames, axis=1):
                z_t, mu_t, lv_t = self.RE_encoder(frame_t)
                z_RE_list.append(z_t); mu_list.append(mu_t); lv_list.append(lv_t)

            z_RE     = tf.stack(z_RE_list, axis=1)
            mu_RE    = tf.stack(mu_list,   axis=1)
            log_var_RE = tf.stack(lv_list, axis=1)
            z_RE     = self.RE_norm(z_RE, centre_id)
        else:                        # already features (B,T,D)
            z_FE       = self.FE_attention(frames, training=training)
            z_RE       = frames
            mu_RE      = tf.zeros_like(z_RE)
            log_var_RE = tf.zeros_like(z_RE)

        z_FE_cross, z_RE_cross = self.cross_attn(z_FE, z_RE, training=training)
        return {
            'z_FE': z_FE_cross, 'z_RE': z_RE_cross,
            'mu_RE': mu_RE, 'log_var_RE': log_var_RE
        }


class HCAEncoderModel(keras.Model):
    """
    3-level Hierarchical Cross-Attention Encoder.
    Level 1 (Low)  : frame-level  (T=64)
    Level 2 (Mid)  : segment-level (4 segments)
    Level 3 (High) : global-level  (1 global)
    """
    def __init__(self, latent_dim=LATENT_DIM, num_centres=NUM_CENTRES):
        super().__init__(name="hca_encoder")
        self.level_low  = HierarchicalLevel(latent_dim, num_centres, "low")
        self.level_mid  = HierarchicalLevel(latent_dim, num_centres, "mid")
        self.level_high = HierarchicalLevel(latent_dim, num_centres, "high")

        self.discriminator_low  = CentreDiscriminator(num_centres)
        self.discriminator_mid  = CentreDiscriminator(num_centres)
        self.discriminator_high = CentreDiscriminator(num_centres)

        self.RE_classifier_low  = layers.Dense(num_centres, name="RE_classifier_low")
        self.RE_classifier_mid  = layers.Dense(num_centres, name="RE_classifier_mid")
        self.RE_classifier_high = layers.Dense(num_centres, name="RE_classifier_high")

        self.decoder           = build_decoder(latent_dim)
        self.pool_low_to_mid   = layers.AveragePooling1D(pool_size=SEGMENT_SIZE)

        self.latent_dim  = latent_dim
        self.num_centres = num_centres

    def call(self, frames, centre_id, training=False):
        # Level 1: frame-level
        out_low   = self.level_low(frames, centre_id, training=training)
        z_FE_low  = out_low['z_FE']    # (B,64,D)
        z_RE_low  = out_low['z_RE']

        # Level 2: segment-level
        z_mid_in  = (self.pool_low_to_mid(z_FE_low) +
                     self.pool_low_to_mid(z_RE_low)) / 2.0   # (B,4,D)
        out_mid   = self.level_mid(z_mid_in, centre_id, training=training)
        z_FE_mid  = out_mid['z_FE']    # (B,4,D)
        z_RE_mid  = out_mid['z_RE']

        # Level 3: global
        z_high_in  = (tf.reduce_mean(z_FE_mid, axis=1, keepdims=True) +
                      tf.reduce_mean(z_RE_mid, axis=1, keepdims=True)) / 2.0  # (B,1,D)
        out_high   = self.level_high(z_high_in, centre_id, training=training)
        z_FE_high  = out_high['z_FE']  # (B,1,D)
        z_RE_high  = out_high['z_RE']

        return {
            'z_FE_low'     : z_FE_low,
            'z_RE_low'     : z_RE_low,
            'mu_RE_low'    : out_low['mu_RE'],
            'log_var_RE_low': out_low['log_var_RE'],
            'z_FE_mid'     : z_FE_mid,
            'z_RE_mid'     : z_RE_mid,
            'z_FE_high'    : z_FE_high,
            'z_RE_high'    : z_RE_high,
        }


# ============================================================
# 5. FEATURE EXTRACTION  
# ============================================================
def _load_clip(row):
    """
    Load a single clip, apply valid mask, sample T_WIN frames,
    resize to IMG_SIZE x IMG_SIZE.
    Returns numpy array (T, IMG_SIZE, IMG_SIZE, 1) or None on failure.
    """
    A_path = Path(row['full_path_A'])
    V_path = Path(row['full_path_V'])

    if not A_path.exists():
        return None

    try:
        A     = np.load(A_path).astype(np.float32) / 255.0
        valid = (np.load(V_path).astype(bool)
                 if row['valid_exists'] and V_path.exists()
                 else np.ones(len(A), dtype=bool))

        idx = np.where(valid)[0]
        if len(idx) < 5:
            return None

        T       = min(len(idx), T_WIN)
        sampled = np.linspace(idx[0], idx[-1], T).astype(int)
        A       = A[sampled]
        A       = tf.image.resize(A[..., None], (IMG_SIZE, IMG_SIZE),
                                  method='area').numpy()
        return A

    except Exception:
        return None


def extract_features_AE(model, df, max_clips=MAX_CLIPS):
    """
    Extract delta-based temporal features using the Autoencoder.
    Returns X (N, D), Y (N,), centres (N,)
    """
    X, Y, centres_list = [], [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips),
             desc="  AE feature extraction")
    ):
        if i >= max_clips:
            break

        A = _load_clip(row)
        if A is None:
            continue

        try:
            centre_id    = CENTRE_MAP[row['centre']]
            centre_id_tf = tf.constant([centre_id], dtype=tf.int32)
            A_batch      = A[None, ...]                   # (1,T,H,W,1)

            # ── POSITIVE ──────────────────────────────────────
            outputs = model(A_batch, centre_id_tf, training=False)
            z       = outputs['z'].numpy()[0]              # (T,D)
            dz      = z[1:] - z[:-1]

            feat = np.concatenate([
                dz.mean(0), dz.std(0),
                np.percentile(dz, [25, 50, 75], axis=0).reshape(-1)
            ])
            X.append(feat); Y.append(1); centres_list.append(centre_id)

            # ── NEGATIVE (shuffled) ────────────────────────────
            perm      = np.random.permutation(len(A))
            outputs_s = model(A[perm][None, ...], centre_id_tf, training=False)
            z_s       = outputs_s['z'].numpy()[0]
            dz_s      = z_s[1:] - z_s[:-1]

            feat_s = np.concatenate([
                dz_s.mean(0), dz_s.std(0),
                np.percentile(dz_s, [25, 50, 75], axis=0).reshape(-1)
            ])
            X.append(feat_s); Y.append(0); centres_list.append(centre_id)

        except Exception:
            continue

    return np.array(X), np.array(Y), np.array(centres_list)


def extract_features_HCA(model, df, max_clips=MAX_CLIPS):
    """
    Extract hierarchical delta features using the HCA Encoder.
    Combines all 3 levels: z_FE_low (T,D) + z_FE_mid (4,D) + z_FE_high (1,D)
    → upsampled and concatenated → (T, 3D).
    """
    X, Y, centres_list = [], [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips),
             desc="  HCA feature extraction")
    ):
        if i >= max_clips:
            break

        A = _load_clip(row)
        if A is None:
            continue

        try:
            T            = len(A)
            centre_id    = CENTRE_MAP[row['centre']]
            centre_id_tf = tf.constant([centre_id], dtype=tf.int32)
            A_batch      = A[None, ...]                    # (1,T,H,W,1)

            # ── POSITIVE ──────────────────────────────────────
            outputs   = model(A_batch, centre_id_tf, training=False)
            z_FE_low  = outputs['z_FE_low'].numpy()[0]     # (T, D)
            z_FE_mid  = outputs['z_FE_mid'].numpy()[0]     # (4, D)
            z_FE_high = outputs['z_FE_high'].numpy()[0]    # (1, D)

            # Upsample mid and high to frame-level → concatenate along feature dim
            z_combined = np.concatenate([
                z_FE_low,
                np.repeat(z_FE_mid,  SEGMENT_SIZE, axis=0),   # (4,D)→(T,D)
                np.repeat(z_FE_high, T,            axis=0)    # (1,D)→(T,D)
            ], axis=-1)   # (T, 3D)

            dz   = z_combined[1:] - z_combined[:-1]
            feat = np.concatenate([
                dz.mean(0), dz.std(0),
                np.percentile(dz, [25, 50, 75], axis=0).reshape(-1)
            ])
            X.append(feat); Y.append(1); centres_list.append(centre_id)

            # ── NEGATIVE (shuffle in z-space, same as training eval) ──
            perm      = np.random.permutation(len(z_combined))
            z_shuf    = z_combined[perm]
            dz_shuf   = z_shuf[1:] - z_shuf[:-1]
            feat_shuf = np.concatenate([
                dz_shuf.mean(0), dz_shuf.std(0),
                np.percentile(dz_shuf, [25, 50, 75], axis=0).reshape(-1)
            ])
            X.append(feat_shuf); Y.append(0); centres_list.append(centre_id)

        except Exception:
            continue

    return np.array(X), np.array(Y), np.array(centres_list)


# ============================================================
# 6. TOP EVALUATION  
# ============================================================
def evaluate_TOP_loco(X_train_full, Y_train_full,
                      X_test, Y_test,
                      label=""):
    """
    Train a logistic regression classifier on train split,
    evaluate on the LOCO held-out test split.

    Includes:
      - Point accuracy
      - Bootstrap 95% CI
      - Permutation test p-value

    Args:
        X_train_full : features from train centres (used to fit classifier)
        Y_train_full : labels from train centres
        X_test       : features from held-out centre
        Y_test       : labels from held-out centre
        label        : string for print logging

    Returns:
        acc, ci_low, ci_high, p_value
    """
    print(f"\n  [{label}] Train samples: {len(X_train_full)} "
          f"| Test samples: {len(X_test)}")

    # Scale features (fit on train only — no leakage)
    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train_full)
    X_test_sc    = scaler.transform(X_test)

    # Fit classifier
    clf = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=SEED)
    clf.fit(X_train_sc, Y_train_full)

    Y_pred = clf.predict(X_test_sc)
    acc    = accuracy_score(Y_test, Y_pred)

    # Bootstrap 95% CI
    boot_acc = []
    for _ in range(BOOTSTRAP_ITERS):
        X_b, Y_b = resample(X_test_sc, Y_test, random_state=None)
        boot_acc.append(accuracy_score(Y_b, clf.predict(X_b)))
    ci_low, ci_high = np.percentile(boot_acc, [2.5, 97.5])

    # Permutation test
    perm_acc = []
    for _ in range(PERMUTATION_ITERS):
        Y_perm = np.random.permutation(Y_train_full)
        clf_p  = LogisticRegression(max_iter=500, random_state=SEED)
        clf_p.fit(X_train_sc, Y_perm)
        perm_acc.append(accuracy_score(Y_test, clf_p.predict(X_test_sc)))
    p_value = np.mean(np.array(perm_acc) >= acc)

    print(f"  [{label}] Acc: {acc*100:.2f}%  "
          f"95%CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]  "
          f"p={p_value:.5f}")

    return acc, ci_low, ci_high, p_value, perm_acc


# ============================================================
# 7. BUILD MODELS & LOAD WEIGHTS
# ============================================================
def build_and_load_ae(weights_path=AE_WEIGHTS_PATH):
    """Instantiate AutoencoderModel and load trained weights."""
    model = AutoencoderModel(latent_dim=LATENT_DIM, num_centres=NUM_CENTRES)
    # Warm-up: build graph with dummy input so weights can be loaded
    dummy_frames    = tf.zeros((1, T_WIN, IMG_SIZE, IMG_SIZE, 1))
    dummy_centre_id = tf.constant([0], dtype=tf.int32)
    _ = model(dummy_frames, dummy_centre_id, training=False)
    model.load_weights(weights_path)
    print(f"  AE weights loaded from: {weights_path}")
    return model


def build_and_load_hca(weights_path=HCA_WEIGHTS_PATH):
    """Instantiate HCAEncoderModel and load trained weights."""
    model = HCAEncoderModel(latent_dim=LATENT_DIM, num_centres=NUM_CENTRES)
    dummy_frames    = tf.zeros((1, T_WIN, IMG_SIZE, IMG_SIZE, 1))
    dummy_centre_id = tf.constant([0], dtype=tf.int32)
    _ = model(dummy_frames, dummy_centre_id, training=False)
    model.load_weights(weights_path)
    print(f"  HCA weights loaded from: {weights_path}")
    return model


# ============================================================
# 8. PLOT (bar chart + numeric table figure)
# ============================================================
def plot_loco_table(results_df,
                    save_path="/content/loco.png"):
    """
    Visualise as a grouped bar chart with CI error bars.

    results_df columns:
      centre, hca_acc, hca_ci_low, hca_ci_high,
               ae_acc,  ae_ci_low,  ae_ci_high, delta
    """
    centres = results_df['centre'].tolist()
    x       = np.arange(len(centres))
    width   = 0.32

    hca_acc   = results_df['hca_acc'].values * 100
    ae_acc    = results_df['ae_acc'].values  * 100
    hca_err   = np.array([
        hca_acc - results_df['hca_ci_low'].values  * 100,
        results_df['hca_ci_high'].values * 100 - hca_acc
    ])
    ae_err    = np.array([
        ae_acc - results_df['ae_ci_low'].values  * 100,
        results_df['ae_ci_high'].values * 100 - ae_acc
    ])
    delta     = results_df['delta'].values

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_hca = ax.bar(x - width / 2, hca_acc, width,
                      yerr=hca_err, capsize=5,
                      color='steelblue', alpha=0.85, label='HCA Encoder')
    bars_ae  = ax.bar(x + width / 2, ae_acc,  width,
                      yerr=ae_err,  capsize=5,
                      color='darkorange', alpha=0.85, label='Temporal AE')

    # Annotate Δ above each pair
    for i, d in enumerate(delta):
        y_top = max(hca_acc[i], ae_acc[i]) + max(hca_err[1, i], ae_err[1, i]) + 0.5
        ax.text(x[i], y_top, f"+{d:.1f}%",
                ha='center', va='bottom', fontsize=10,
                color='dimgray', fontweight='bold')

    centre_labels = [c.replace('Merkez', 'Center') for c in centres]
    ax.set_xticks(x)
    ax.set_xticklabels(centre_labels, fontsize=12)
    ax.set_ylabel('TOP Accuracy (%)', fontsize=12)
    ax.set_title('Leave-One-Center-Out (LOCO) Validation', fontsize=13)
    ax.set_ylim(85, 105)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nLOCO bar chart saved → {save_path}")


def plot_null_distributions(null_dict,
                             save_path="/content/loco_null_distributions.png"):
    """
    Plot permutation null distributions for all LOCO folds × both models.
    null_dict: {(centre, model_label): (perm_acc_list, observed_acc)}
    """
    n_rows = len(CENTRES)
    n_cols = 2   # HCA | AE
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

    model_labels = ['HCA Encoder', 'Temporal AE']
    model_keys   = ['HCA', 'AE']

    for row_i, centre in enumerate(CENTRES):
        for col_j, (mk, ml) in enumerate(zip(model_keys, model_labels)):
            ax   = axes[row_i, col_j]
            key  = (centre, mk)
            perm_acc, obs_acc = null_dict.get(key, ([], 0.0))

            if perm_acc:
                ax.hist(perm_acc, bins=30, color='steelblue', alpha=0.7,
                        edgecolor='white')
                ax.axvline(obs_acc, color='crimson', lw=2,
                           label=f'Obs = {obs_acc*100:.2f}%')
                ax.legend(fontsize=9)

            ax.set_title(f"{centre.replace('Merkez','Center')} — {ml}",
                         fontsize=10)
            ax.set_xlabel('Accuracy')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)

    plt.suptitle('LOCO Permutation Null Distributions', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Null distribution plots saved → {save_path}")


# ============================================================
# 9. MAIN — LOCO LOOP
# ============================================================
if __name__ == "__main__":

    print("=" * 65)
    print("LOCO VALIDATION")
    print("Test Center | HCA Acc. (%) | Temporal AE Acc. (%) | Δ (%)")
    print("=" * 65)

    # ── Load full metadata ─────────────────────────────────────
    df = create_metadata()

    # ── Load trained models ────────────────────────────────────
    print("\nLoading model weights...")
    ae_model  = build_and_load_ae(AE_WEIGHTS_PATH)
    hca_model = build_and_load_hca(HCA_WEIGHTS_PATH)

    # ── Storage ────────────────────────────────────────────────
    results_rows = []
    null_dict    = {}     # {(centre, 'HCA'|'AE'): (perm_list, obs_acc)}

    # ── LOCO loop — one fold per centre ───────────────────────
    for held_out_centre in CENTRES:
        print(f"\n{'='*65}")
        print(f"FOLD: Held-out = {held_out_centre}")
        print(f"{'='*65}")

        df_train, df_test = loco_split(df, held_out_centre)

        # ── Extract features: AE ────────────────────────────
        print("\n  Extracting AE features (train centres)...")
        X_ae_train, Y_ae_train, _ = extract_features_AE(
            ae_model, df_train, max_clips=MAX_CLIPS)

        print("  Extracting AE features (test centre)...")
        X_ae_test, Y_ae_test, _   = extract_features_AE(
            ae_model, df_test, max_clips=MAX_CLIPS)

        # ── Extract features: HCA ───────────────────────────
        print("\n  Extracting HCA features (train centres)...")
        X_hca_train, Y_hca_train, _ = extract_features_HCA(
            hca_model, df_train, max_clips=MAX_CLIPS)

        print("  Extracting HCA features (test centre)...")
        X_hca_test, Y_hca_test, _   = extract_features_HCA(
            hca_model, df_test, max_clips=MAX_CLIPS)

        # ── Evaluate ────────────────────────────────────────
        print(f"\n  Running TOP evaluation...")

        if len(X_ae_test) == 0 or len(X_hca_test) == 0:
            print(f"  WARNING: No features for {held_out_centre}. Skipping fold.")
            continue

        ae_acc, ae_ci_low, ae_ci_high, ae_p, ae_perm = evaluate_TOP_loco(
            X_ae_train,  Y_ae_train,
            X_ae_test,   Y_ae_test,
            label="Temporal AE"
        )

        hca_acc, hca_ci_low, hca_ci_high, hca_p, hca_perm = evaluate_TOP_loco(
            X_hca_train, Y_hca_train,
            X_hca_test,  Y_hca_test,
            label="HCA Encoder"
        )

        delta = (hca_acc - ae_acc) * 100  # Δ in percentage points

        # Store null distributions for plotting
        null_dict[(held_out_centre, 'AE')]  = (ae_perm,  ae_acc)
        null_dict[(held_out_centre, 'HCA')] = (hca_perm, hca_acc)

        results_rows.append({
            'centre'      : held_out_centre,
            'hca_acc'     : hca_acc,
            'hca_ci_low'  : hca_ci_low,
            'hca_ci_high' : hca_ci_high,
            'hca_p'       : hca_p,
            'ae_acc'      : ae_acc,
            'ae_ci_low'   : ae_ci_low,
            'ae_ci_high'  : ae_ci_high,
            'ae_p'        : ae_p,
            'delta'       : delta
        })

    # ── Build results DataFrame ───────────────────────────────
    results_df = pd.DataFrame(results_rows)

    # Add Mean row
    mean_row = {
        'centre'      : 'Mean',
        'hca_acc'     : results_df['hca_acc'].mean(),
        'hca_ci_low'  : results_df['hca_ci_low'].mean(),
        'hca_ci_high' : results_df['hca_ci_high'].mean(),
        'hca_p'       : float('nan'),
        'ae_acc'      : results_df['ae_acc'].mean(),
        'ae_ci_low'   : results_df['ae_ci_low'].mean(),
        'ae_ci_high'  : results_df['ae_ci_high'].mean(),
        'ae_p'        : float('nan'),
        'delta'       : results_df['delta'].mean()
    }
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row])],
        ignore_index=True
    )

    # ── Print Loco Results ────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("LOCO VALIDATION RESULTS")
    print("=" * 65)
    print(f"{'Test Center':<14} | {'HCA Acc. (%)':<20} | "
          f"{'Temporal AE Acc. (%)':<22} | {'Δ (%)':<8}")
    print("-" * 65)

    for _, row in results_df.iterrows():
        if row['centre'] == 'Mean':
            print("-" * 65)

        centre_label = row['centre'].replace('Merkez', 'Center')

        if row['centre'] == 'Mean':
            hca_str = f"{row['hca_acc']*100:.1f}"
            ae_str  = f"{row['ae_acc']*100:.1f}"
        else:
            hca_str = (f"{row['hca_acc']*100:.1f}  "
                       f"[{row['hca_ci_low']*100:.1f}, {row['hca_ci_high']*100:.1f}]  "
                       f"p={row['hca_p']:.4f}")
            ae_str  = (f"{row['ae_acc']*100:.1f}  "
                       f"[{row['ae_ci_low']*100:.1f}, {row['ae_ci_high']*100:.1f}]  "
                       f"p={row['ae_p']:.4f}")

        delta_str = f"+{row['delta']:.1f}"
        print(f"{centre_label:<14} | {hca_str:<34} | {ae_str:<34} | {delta_str}")

    print("=" * 65)

    # ── Save results as CSV ───────────────────────────────────
    csv_path = "/content/loco_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # ── Plots ────────────────────────────────────────────────
    # Drop mean row for plotting (only per-centre bars)
    plot_df = results_df[results_df['centre'] != 'Mean'].reset_index(drop=True)
    plot_loco_table(plot_df)
    plot_null_distributions(null_dict)

    print("\nLOCO evaluation complete.")