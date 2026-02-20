import os
import time
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
from sklearn.metrics import accuracy_score, silhouette_score
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

# Trained weight paths
AE_WEIGHTS_PATH   = "/content/AE_best_model.weights.h5"
HCA_WEIGHTS_PATH  = "/content/HCA_best_model.weights.h5"
DANN_WEIGHTS_PATH = "/content/DANN_best_model.weights.h5"

# Architecture hyperparameters — identical across all methods
IMG_SIZE     = 128
T_WIN        = 64
LATENT_DIM   = 128
BATCH_SIZE   = 4
EPOCHS       = 10
LR           = 2e-4
NUM_SEGMENTS = 4
SEGMENT_SIZE = T_WIN // NUM_SEGMENTS    # 16

# Loss weights
W_REC        = 1.0
W_SCALE      = 1.0
W_SPEC       = 0.5
W_ADV        = 1.0
W_BATCH      = 1.0
W_KL         = 0.01
W_CROSS_LOW  = 0.3
W_CROSS_MID  = 0.5
W_CROSS_HIGH = 0.7
W_HIERARCHY  = 0.2

# DANN specific
DANN_LAMBDA  = 1.0      # gradient reversal strength

# Evaluation
MAX_CLIPS         = 500
N_RUNS            = 5      # repeated runs for ± std (matches paper's ± values)
BOOTSTRAP_ITERS   = 1000
PERMUTATION_ITERS = 500

# Centre definitions
CENTRE_MAP  = {'Merkez 2': 0, 'Merkez 6': 1, 'Merkez 8': 2}
NUM_CENTRES = len(CENTRE_MAP)

# Method registry — evaluated in this order
METHODS = [
    "raw",
    "combat_ae",
    "dann",
    "temporal_ae",
    "hca",
]

TRAINING_SPEED = {
    "raw"        : "–",
    "combat_ae"  : "Fast",
    "dann"       : "Slow",
    "temporal_ae": "Fast",
    "hca"        : "Moderate",
}

# ============================================================
# 1. METADATA & SPLIT
# ============================================================
def create_metadata():
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
        raise ValueError("No data found! Check DIRS_TO_SCAN.")

    print(f"\nTotal clips: {len(df)}")
    print(df.groupby('centre').size().to_string())
    return df


def split_data(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Subject-level 70/15/15 split — no data leakage across subjects."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    subjects = df['subject'].unique()
    np.random.shuffle(subjects)

    n       = len(subjects)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_subj = subjects[:n_train]
    val_subj   = subjects[n_train : n_train + n_val]
    test_subj  = subjects[n_train + n_val :]

    df_train = df[df['subject'].isin(train_subj)].reset_index(drop=True)
    df_val   = df[df['subject'].isin(val_subj)].reset_index(drop=True)
    df_test  = df[df['subject'].isin(test_subj)].reset_index(drop=True)

    print("\n" + "=" * 60)
    print("DATA SPLIT (by subject)")
    print("=" * 60)
    print(f"Train : {len(train_subj)} subjects, {len(df_train)} clips")
    print(f"Val   : {len(val_subj)} subjects,   {len(df_val)} clips")
    print(f"Test  : {len(test_subj)} subjects,  {len(df_test)} clips")

    for split_name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"\n{split_name} centre distribution:")
        print(split_df.groupby('centre').size())
    print("=" * 60)

    return df_train, df_val, df_test

# ============================================================
# 2. SHARED BUILDING BLOCKS
# ============================================================
def build_cnn_backbone(latent_dim, name="encoder"):
    """5-layer progressive-downsampling CNN. Shared by all methods."""
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
    """Transposed-conv decoder. Shared by all methods."""
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
    def __init__(self, dim, num_heads=4, dropout=0.1, name_prefix=""):
        super().__init__(name=f"{name_prefix}_tattn")
        self.mha  = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout)
        self.norm = layers.LayerNormalization()

    def call(self, z, training=False):
        return self.norm(z + self.mha(z, z, training=training))


class CrossAttention(layers.Layer):
    def __init__(self, dim, num_heads=4, dropout=0.1, name_prefix=""):
        super().__init__(name=f"{name_prefix}_cross")
        self.FE_to_RE = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout)
        self.RE_to_FE = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout)
        self.norm_FE  = layers.LayerNormalization()
        self.norm_RE  = layers.LayerNormalization()
        self.gate_FE  = layers.Dense(dim, activation='sigmoid')
        self.gate_RE  = layers.Dense(dim, activation='sigmoid')

    def call(self, z_FE, z_RE, training=False):
        FE_cross = self.FE_to_RE(z_FE, z_RE, z_RE, training=training)
        z_FE_out = self.norm_FE(z_FE + self.gate_FE(z_FE) * FE_cross)
        RE_cross = self.RE_to_FE(z_RE, z_FE, z_FE, training=training)
        z_RE_out = self.norm_RE(z_RE + self.gate_RE(z_RE) * RE_cross)
        return z_FE_out, z_RE_out


class BatchConditionalNorm(layers.Layer):
    def __init__(self, num_centres, dim):
        super().__init__()
        self.gamma_emb = layers.Embedding(num_centres, dim)
        self.beta_emb  = layers.Embedding(num_centres, dim)

    def call(self, z, centre_id):
        mean   = tf.reduce_mean(z, axis=[1, 2], keepdims=True)
        std    = tf.math.reduce_std(z, axis=[1, 2], keepdims=True)
        z_norm = (z - mean) / (std + 1e-6)
        return (self.gamma_emb(centre_id)[:, None, :] * z_norm +
                self.beta_emb(centre_id)[:, None, :])


class CentreDiscriminator(keras.Model):
    def __init__(self, num_centres, name="discriminator"):
        super().__init__(name=name)
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
    def __init__(self, latent_dim):
        super().__init__(name="bayesian_encoder")
        self.backbone   = build_cnn_backbone(latent_dim * 2)
        self.latent_dim = latent_dim

    def call(self, x):
        h       = self.backbone(x)
        mu      = h[..., :self.latent_dim]
        log_var = h[..., self.latent_dim:]
        z       = mu + tf.random.normal(tf.shape(mu)) * tf.exp(0.5 * log_var)
        return z, mu, log_var

# ============================================================
# 3a. TEMPORAL AUTOENCODER  
# ============================================================
class TemporalAEModel(keras.Model):
    """
    Temporal Autoencoder with scale + spectral regularization
    and adversarial centre discriminator.
    Matches autoencoder.py architecture exactly.
    """
    def __init__(self, latent_dim=LATENT_DIM, num_centres=NUM_CENTRES):
        super().__init__(name="temporal_ae")
        self.encoder           = build_cnn_backbone(latent_dim, "ae_encoder")
        self.temp_attention    = TemporalAttention(latent_dim, name_prefix="ae")
        self.batch_norm        = BatchConditionalNorm(num_centres, latent_dim)
        self.decoder           = build_decoder(latent_dim)
        self.discriminator     = CentreDiscriminator(num_centres)
        self.centre_classifier = layers.Dense(num_centres, name="centre_cls")
        self.latent_dim        = latent_dim
        self.num_centres       = num_centres

    def call(self, frames, centre_id, training=False):
        z_raw  = layers.TimeDistributed(self.encoder)(frames)
        z_attn = self.temp_attention(z_raw, training=training)
        z      = self.batch_norm(z_attn, centre_id)
        x_rec  = layers.TimeDistributed(self.decoder)(z)
        return {'z': z, 'z_raw': z_raw, 'x_rec': x_rec}

# ============================================================
# 3b. HCA ENCODER 
# ============================================================
class HierarchicalLevel(keras.Model):
    def __init__(self, latent_dim, num_centres, level_name):
        super().__init__(name=f"level_{level_name}")
        self.FE_encoder   = build_cnn_backbone(latent_dim, f"FE_{level_name}")
        self.FE_attention = TemporalAttention(latent_dim, name_prefix=f"FE_{level_name}")
        self.RE_encoder   = BayesianEncoder(latent_dim)
        self.RE_norm      = BatchConditionalNorm(num_centres, latent_dim)
        self.cross_attn   = CrossAttention(latent_dim, name_prefix=level_name)

    def call(self, frames, centre_id, training=False):
        if len(frames.shape) == 5:
            z_FE = layers.TimeDistributed(self.FE_encoder)(frames)
            z_FE = self.FE_attention(z_FE, training=training)

            z_RE_list, mu_list, lv_list = [], [], []
            for ft in tf.unstack(frames, axis=1):
                zt, mt, lt = self.RE_encoder(ft)
                z_RE_list.append(zt); mu_list.append(mt); lv_list.append(lt)

            z_RE = tf.stack(z_RE_list, axis=1)
            mu_RE, lv_RE = tf.stack(mu_list, axis=1), tf.stack(lv_list, axis=1)
            z_RE = self.RE_norm(z_RE, centre_id)
        else:
            z_FE  = self.FE_attention(frames, training=training)
            z_RE  = frames
            mu_RE = tf.zeros_like(z_RE)
            lv_RE = tf.zeros_like(z_RE)

        z_FE_out, z_RE_out = self.cross_attn(z_FE, z_RE, training=training)
        return {'z_FE': z_FE_out, 'z_RE': z_RE_out,
                'mu_RE': mu_RE, 'log_var_RE': lv_RE}


class HCAEncoderModel(keras.Model):
    """3-level HCA encoder. Matches hca_encoder.py exactly."""
    def __init__(self, latent_dim=LATENT_DIM, num_centres=NUM_CENTRES):
        super().__init__(name="hca_encoder")
        self.level_low  = HierarchicalLevel(latent_dim, num_centres, "low")
        self.level_mid  = HierarchicalLevel(latent_dim, num_centres, "mid")
        self.level_high = HierarchicalLevel(latent_dim, num_centres, "high")

        self.discriminator_low  = CentreDiscriminator(num_centres, "disc_low")
        self.discriminator_mid  = CentreDiscriminator(num_centres, "disc_mid")
        self.discriminator_high = CentreDiscriminator(num_centres, "disc_high")

        self.RE_classifier_low  = layers.Dense(num_centres, name="RE_cls_low")
        self.RE_classifier_mid  = layers.Dense(num_centres, name="RE_cls_mid")
        self.RE_classifier_high = layers.Dense(num_centres, name="RE_cls_high")

        self.decoder         = build_decoder(latent_dim)
        self.pool_low_to_mid = layers.AveragePooling1D(pool_size=SEGMENT_SIZE)
        self.latent_dim      = latent_dim
        self.num_centres     = num_centres

    def call(self, frames, centre_id, training=False):
        out_low  = self.level_low(frames, centre_id, training=training)
        z_FE_low = out_low['z_FE']
        z_RE_low = out_low['z_RE']

        z_mid_in = (self.pool_low_to_mid(z_FE_low) +
                    self.pool_low_to_mid(z_RE_low)) / 2.0
        out_mid  = self.level_mid(z_mid_in, centre_id, training=training)
        z_FE_mid = out_mid['z_FE']
        z_RE_mid = out_mid['z_RE']

        z_high_in = (tf.reduce_mean(z_FE_mid, axis=1, keepdims=True) +
                     tf.reduce_mean(z_RE_mid, axis=1, keepdims=True)) / 2.0
        out_high  = self.level_high(z_high_in, centre_id, training=training)

        return {
            'z_FE_low'      : z_FE_low,
            'z_RE_low'      : z_RE_low,
            'mu_RE_low'     : out_low['mu_RE'],
            'log_var_RE_low': out_low['log_var_RE'],
            'z_FE_mid'      : out_mid['z_FE'],
            'z_RE_mid'      : out_mid['z_RE'],
            'z_FE_high'     : out_high['z_FE'],
            'z_RE_high'     : out_high['z_RE'],
        }

# ============================================================
# 3c. DANN — Domain-Adversarial Neural Network
#     (Ganin et al., 2016)
#     Gradient reversal layer + domain classifier on top of encoder
# ============================================================
@tf.custom_gradient
def _gradient_reversal_op(x, lambda_val):
    """Forward pass = identity; backward pass = multiply gradient by -lambda."""
    def grad(dy):
        return -lambda_val * dy, None
    return x, grad


class GradientReversalLayer(layers.Layer):
    """Wraps gradient reversal as a Keras layer."""
    def __init__(self, lambda_val=DANN_LAMBDA):
        super().__init__(name="gradient_reversal")
        self.lambda_val = tf.Variable(float(lambda_val), trainable=False,
                                      dtype=tf.float32)

    def call(self, x):
        return _gradient_reversal_op(x, self.lambda_val)


class DANNModel(keras.Model):
    """
    Domain-Adversarial Neural Network for centre harmonization.
    Architecture:
      encoder → feature  → task head (reconstruction)
                         → GRL → domain classifier (centre)
    The encoder is trained to fool the domain classifier via GRL,
    learning centre-invariant representations.
    """
    def __init__(self, latent_dim=LATENT_DIM, num_centres=NUM_CENTRES):
        super().__init__(name="dann")
        self.encoder          = build_cnn_backbone(latent_dim, "dann_encoder")
        self.temp_attention   = TemporalAttention(latent_dim, name_prefix="dann")
        self.decoder          = build_decoder(latent_dim)
        self.grl              = GradientReversalLayer(DANN_LAMBDA)
        self.domain_classifier = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64,  activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_centres)
        ], name="domain_classifier")
        self.latent_dim  = latent_dim
        self.num_centres = num_centres

    def call(self, frames, centre_id=None, training=False):
        # Encode each frame
        z     = layers.TimeDistributed(self.encoder)(frames)   # (B,T,D)
        z_attn = self.temp_attention(z, training=training)     # (B,T,D)

        # Reconstruction head
        x_rec = layers.TimeDistributed(self.decoder)(z_attn)   # (B,T,H,W,1)

        # Domain classification head via GRL
        z_pooled  = tf.reduce_mean(z_attn, axis=1)             # (B,D)
        z_grl     = self.grl(z_pooled)                         # (B,D) — reversed grad
        domain_pred = self.domain_classifier(z_grl, training=training)  # (B,C)

        return {'z': z_attn, 'x_rec': x_rec, 'domain_pred': domain_pred}


def train_dann(model, df_train, df_val,
               epochs=EPOCHS, steps_per_epoch=200, val_steps=50):
    """
    Train DANN with joint reconstruction + domain adversarial loss.
    Single optimizer — gradient reversal handles the adversarial part.
    """
    print("\n" + "=" * 60)
    print("TRAINING: DANN")
    print("=" * 60)

    optimizer     = keras.optimizers.Adam(LR)
    best_val_loss = float('inf')

    dataset_train = _build_dataset(df_train)
    dataset_val   = _build_dataset(df_val)

    @tf.function
    def train_step(frames, centre_id):
        with tf.GradientTape() as tape:
            out      = model(frames, centre_id, training=True)
            z        = out['z']
            x_rec    = out['x_rec']
            d_pred   = out['domain_pred']

            L_rec    = tf.reduce_mean(tf.square(frames - x_rec))
            L_domain = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, d_pred, from_logits=True))
            L_total  = L_rec + DANN_LAMBDA * L_domain

        grads = tape.gradient(L_total, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return L_total, L_rec, L_domain

    @tf.function
    def val_step(frames, centre_id):
        out     = model(frames, centre_id, training=False)
        L_rec   = tf.reduce_mean(tf.square(frames - out['x_rec']))
        L_domain = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, out['domain_pred'], from_logits=True))
        return L_rec + DANN_LAMBDA * L_domain

    for epoch in range(1, epochs + 1):
        tr_losses = [train_step(f, c) for f, c in dataset_train.take(steps_per_epoch)]
        vl_losses = [float(val_step(f, c)) for f, c in dataset_val.take(val_steps)]

        avg_tr  = np.mean([float(l[0]) for l in tr_losses])
        avg_rec = np.mean([float(l[1]) for l in tr_losses])
        avg_dom = np.mean([float(l[2]) for l in tr_losses])
        avg_val = np.mean(vl_losses)

        print(f"  Epoch {epoch}/{epochs} | "
              f"Train: {avg_tr:.4f} (rec={avg_rec:.4f}, dom={avg_dom:.4f}) | "
              f"Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_weights(DANN_WEIGHTS_PATH)
            print(f"  Best DANN saved (val={best_val_loss:.4f})")

    print(f"  DANN training done. Best val: {best_val_loss:.4f}")
    return model

# ============================================================
# 3d. ComBat  (neuroCombat-style, numpy implementation)
#     Applied to clip-level feature vectors extracted by a plain
#     (untrained) CNN encoder, then an autoencoder is trained
#     on the harmonized features.
# ============================================================
def combat_harmonize(X, centres):
    """
    Simplified ComBat harmonization (Johnson et al., 2007).

    Removes additive (beta) and multiplicative (gamma) batch effects
    per centre using an empirical Bayes estimate.

    Args:
        X       : (N, D) feature matrix
        centres : (N,) integer centre labels

    Returns:
        X_harmonized : (N, D) corrected features
    """
    X_harm    = X.copy().astype(np.float64)
    unique_c  = np.unique(centres)

    # Grand mean across all samples
    grand_mean = X.mean(0)                                # (D,)

    # Per-centre additive offset (beta) and multiplicative scale (gamma)
    beta_hat  = {}
    gamma_hat = {}

    for c in unique_c:
        idx          = centres == c
        centre_mean  = X[idx].mean(0)
        centre_std   = X[idx].std(0) + 1e-8

        beta_hat[c]  = centre_mean - grand_mean
        gamma_hat[c] = centre_std

    # Correct each sample
    for c in unique_c:
        idx = centres == c
        X_harm[idx] = (X[idx] - beta_hat[c]) / gamma_hat[c]

    return X_harm.astype(np.float32)


class PlainAEModel(keras.Model):
    """
    Plain autoencoder used after ComBat correction.
    No temporal regularization, no adversarial training.
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__(name="plain_ae")
        self.encoder    = build_cnn_backbone(latent_dim, "plain_encoder")
        self.decoder    = build_decoder(latent_dim)
        self.latent_dim = latent_dim

    def call(self, frames, centre_id=None, training=False):
        z     = layers.TimeDistributed(self.encoder)(frames)
        x_rec = layers.TimeDistributed(self.decoder)(z)
        return {'z': z, 'x_rec': x_rec}


def train_combat_ae(plain_ae_model, df_train, df_val,
                    epochs=EPOCHS, steps_per_epoch=200, val_steps=50):
    """
    1. Extract raw CNN features from all clips (untrained backbone).
    2. Apply ComBat to harmonize across centres.
    3. Train plain AE on harmonized frames.

    Note: We apply ComBat at the pixel distribution level (per-frame
    mean intensity per centre) rather than at latent level, to stay
    consistent with standard ComBat usage in imaging pipelines.
    """
    print("\n" + "=" * 60)
    print("TRAINING: ComBat + Autoencoder")
    print("=" * 60)
    print("  Step 1: Computing per-centre intensity statistics for ComBat...")

    # Compute per-centre mean and std from training clips
    centre_stats = {}
    for _, row in df_train.iterrows():
        A = _load_clip_raw(row)
        if A is None:
            continue
        c = CENTRE_MAP[row['centre']]
        if c not in centre_stats:
            centre_stats[c] = {'sum': 0.0, 'sum_sq': 0.0, 'n': 0}
        centre_stats[c]['sum']    += A.mean()
        centre_stats[c]['sum_sq'] += (A ** 2).mean()
        centre_stats[c]['n']      += 1

    grand_mean_val = np.mean([
        centre_stats[c]['sum'] / centre_stats[c]['n']
        for c in centre_stats
    ])

    centre_beta  = {}
    centre_gamma = {}
    for c, stats in centre_stats.items():
        c_mean  = stats['sum']    / stats['n']
        c_mean2 = stats['sum_sq'] / stats['n']
        c_var   = max(c_mean2 - c_mean ** 2, 1e-8)
        centre_beta[c]  = c_mean - grand_mean_val
        centre_gamma[c] = np.sqrt(c_var)

    print(f"  ComBat stats computed for {len(centre_stats)} centres.")
    print(f"  Grand mean: {grand_mean_val:.4f}")

    optimizer     = keras.optimizers.Adam(LR)
    best_val_loss = float('inf')
    weights_path  = "/content/COMBAT_AE_best_model.weights.h5"

    dataset_train = _build_dataset_combat(df_train, centre_beta, centre_gamma)
    dataset_val   = _build_dataset_combat(df_val,   centre_beta, centre_gamma)

    @tf.function
    def train_step(frames, centre_id):
        with tf.GradientTape() as tape:
            out    = plain_ae_model(frames, training=True)
            L_rec  = tf.reduce_mean(tf.square(frames - out['x_rec']))
        grads = tape.gradient(L_rec, plain_ae_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, plain_ae_model.trainable_variables))
        return L_rec

    @tf.function
    def val_step(frames, centre_id):
        out = plain_ae_model(frames, training=False)
        return tf.reduce_mean(tf.square(frames - out['x_rec']))

    for epoch in range(1, epochs + 1):
        tr_losses = [float(train_step(f, c))
                     for f, c in dataset_train.take(steps_per_epoch)]
        vl_losses = [float(val_step(f, c))
                     for f, c in dataset_val.take(val_steps)]

        avg_tr  = np.mean(tr_losses)
        avg_val = np.mean(vl_losses)
        print(f"  Epoch {epoch}/{epochs} | Train: {avg_tr:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            plain_ae_model.save_weights(weights_path)
            print(f"  Best ComBat AE saved (val={best_val_loss:.4f})")

    plain_ae_model.load_weights(weights_path)
    print(f"  ComBat AE training done. Best val: {best_val_loss:.4f}")
    return plain_ae_model, centre_beta, centre_gamma

# ============================================================
# 4. DATA GENERATORS
# ============================================================
def _build_dataset(df, batch_size=BATCH_SIZE):
    """Standard subject-balanced generator."""
    subj2clips = {}
    for _, r in df.iterrows():
        subj2clips.setdefault((r['centre'], r['subject']), []).append(r)
    subjects = list(subj2clips.keys())

    def load_one():
        for _ in range(100):
            c, s      = random.choice(subjects)
            clip_info = random.choice(subj2clips[(c, s)])
            A = _load_clip_raw(clip_info)
            if A is not None:
                return A.astype(np.float32), CENTRE_MAP[c]
        return np.zeros((T_WIN, IMG_SIZE, IMG_SIZE, 1), np.float32), 0

    def gen():
        while True:
            yield load_one()

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((T_WIN, IMG_SIZE, IMG_SIZE, 1), tf.float32),
            tf.TensorSpec((), tf.int32)
        )
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def _build_dataset_combat(df, centre_beta, centre_gamma, batch_size=BATCH_SIZE):
    """
    Generator that applies ComBat pixel-level correction on-the-fly.
    Normalizes each frame by subtracting centre beta and dividing by centre gamma.
    """
    subj2clips = {}
    for _, r in df.iterrows():
        subj2clips.setdefault((r['centre'], r['subject']), []).append(r)
    subjects = list(subj2clips.keys())

    def load_one():
        for _ in range(100):
            c, s      = random.choice(subjects)
            clip_info = random.choice(subj2clips[(c, s)])
            A = _load_clip_raw(clip_info)
            if A is not None:
                cid   = CENTRE_MAP[c]
                beta  = centre_beta.get(cid,  0.0)
                gamma = centre_gamma.get(cid, 1.0)
                A_corrected = np.clip((A - beta) / (gamma + 1e-8), 0.0, 1.0)
                return A_corrected.astype(np.float32), cid
        return np.zeros((T_WIN, IMG_SIZE, IMG_SIZE, 1), np.float32), 0

    def gen():
        while True:
            yield load_one()

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((T_WIN, IMG_SIZE, IMG_SIZE, 1), tf.float32),
            tf.TensorSpec((), tf.int32)
        )
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def _load_clip_raw(row):
    """Load and preprocess one clip → (T, H, W, 1) in [0,1], or None."""
    A_path = Path(row['full_path_A'])
    V_path = Path(row['full_path_V'])
    if not A_path.exists():
        return None
    try:
        A     = np.load(A_path).astype(np.float32) / 255.0
        valid = (np.load(V_path).astype(bool)
                 if row['valid_exists'] and V_path.exists()
                 else np.ones(len(A), dtype=bool))
        idx   = np.where(valid)[0]
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

# ============================================================
# 5. FEATURE EXTRACTION  (per method)
# ============================================================
def _delta_features(z_seq):
    """Compute delta statistics from a latent sequence (T, D)."""
    dz = z_seq[1:] - z_seq[:-1]
    return np.concatenate([
        dz.mean(0), dz.std(0),
        np.percentile(dz, [25, 50, 75], axis=0).reshape(-1)
    ])


def extract_features_raw(df, max_clips=MAX_CLIPS):
    """
    Raw baseline: random (untrained) CNN encoder.
    Latent space has no learned structure, so TOP ≈ chance.
    """
    backbone = build_cnn_backbone(LATENT_DIM, "raw_backbone")
    X, Y, centres_list = [], [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips), desc="  Raw")
    ):
        if i >= max_clips: break
        A = _load_clip_raw(row)
        if A is None: continue
        try:
            centre_id = CENTRE_MAP[row['centre']]

            # Positive
            z    = backbone(A.reshape(-1, IMG_SIZE, IMG_SIZE, 1),
                            training=False).numpy()          # (T,D)
            feat = _delta_features(z)
            X.append(feat); Y.append(1); centres_list.append(centre_id)

            # Negative
            perm   = np.random.permutation(len(A))
            z_shuf = backbone(A[perm].reshape(-1, IMG_SIZE, IMG_SIZE, 1),
                              training=False).numpy()
            X.append(_delta_features(z_shuf))
            Y.append(0); centres_list.append(centre_id)
        except Exception:
            continue

    return np.array(X), np.array(Y), np.array(centres_list)


def extract_features_dann(model, df, max_clips=MAX_CLIPS):
    """DANN: use z from the encoder output."""
    X, Y, centres_list = [], [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips), desc="  DANN")
    ):
        if i >= max_clips: break
        A = _load_clip_raw(row)
        if A is None: continue
        try:
            centre_id    = CENTRE_MAP[row['centre']]
            centre_id_tf = tf.constant([centre_id], dtype=tf.int32)

            out  = model(A[None, ...], centre_id_tf, training=False)
            z    = out['z'].numpy()[0]     # (T,D)

            X.append(_delta_features(z));    Y.append(1); centres_list.append(centre_id)

            perm  = np.random.permutation(len(A))
            out_s = model(A[perm][None, ...], centre_id_tf, training=False)
            z_s   = out_s['z'].numpy()[0]
            X.append(_delta_features(z_s)); Y.append(0); centres_list.append(centre_id)
        except Exception:
            continue

    return np.array(X), np.array(Y), np.array(centres_list)


def extract_features_combat_ae(model, centre_beta, centre_gamma,
                                df, max_clips=MAX_CLIPS):
    """ComBat + AE: apply ComBat correction first, then encode."""
    X, Y, centres_list = [], [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips), desc="  ComBat+AE")
    ):
        if i >= max_clips: break
        A = _load_clip_raw(row)
        if A is None: continue
        try:
            centre_id = CENTRE_MAP[row['centre']]
            beta      = centre_beta.get(centre_id,  0.0)
            gamma     = centre_gamma.get(centre_id, 1.0)
            A_corr    = np.clip((A - beta) / (gamma + 1e-8), 0.0, 1.0)

            out = model(A_corr[None, ...], training=False)
            z   = out['z'].numpy()[0]

            X.append(_delta_features(z));    Y.append(1); centres_list.append(centre_id)

            perm   = np.random.permutation(len(A_corr))
            out_s  = model(A_corr[perm][None, ...], training=False)
            z_s    = out_s['z'].numpy()[0]
            X.append(_delta_features(z_s)); Y.append(0); centres_list.append(centre_id)
        except Exception:
            continue

    return np.array(X), np.array(Y), np.array(centres_list)


def extract_features_temporal_ae(model, df, max_clips=MAX_CLIPS):
    """Temporal AE: use z from autoencoder."""
    X, Y, centres_list = [], [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips), desc="  Temporal AE")
    ):
        if i >= max_clips: break
        A = _load_clip_raw(row)
        if A is None: continue
        try:
            centre_id    = CENTRE_MAP[row['centre']]
            centre_id_tf = tf.constant([centre_id], dtype=tf.int32)

            out = model(A[None, ...], centre_id_tf, training=False)
            z   = out['z'].numpy()[0]

            X.append(_delta_features(z));    Y.append(1); centres_list.append(centre_id)

            perm  = np.random.permutation(len(A))
            out_s = model(A[perm][None, ...], centre_id_tf, training=False)
            X.append(_delta_features(out_s['z'].numpy()[0]))
            Y.append(0); centres_list.append(centre_id)
        except Exception:
            continue

    return np.array(X), np.array(Y), np.array(centres_list)


def extract_features_hca(model, df, max_clips=MAX_CLIPS):
    """HCA: combine all 3 hierarchical levels."""
    X, Y, centres_list = [], [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips), desc="  HCA")
    ):
        if i >= max_clips: break
        A = _load_clip_raw(row)
        if A is None: continue
        try:
            T            = len(A)
            centre_id    = CENTRE_MAP[row['centre']]
            centre_id_tf = tf.constant([centre_id], dtype=tf.int32)

            out       = model(A[None, ...], centre_id_tf, training=False)
            z_FE_low  = out['z_FE_low'].numpy()[0]
            z_FE_mid  = out['z_FE_mid'].numpy()[0]
            z_FE_high = out['z_FE_high'].numpy()[0]

            z_combined = np.concatenate([
                z_FE_low,
                np.repeat(z_FE_mid,  SEGMENT_SIZE, axis=0),
                np.repeat(z_FE_high, T,            axis=0)
            ], axis=-1)

            X.append(_delta_features(z_combined)); Y.append(1)
            centres_list.append(centre_id)

            perm    = np.random.permutation(T)
            z_shuf  = z_combined[perm]
            X.append(_delta_features(z_shuf)); Y.append(0)
            centres_list.append(centre_id)
        except Exception:
            continue

    return np.array(X), np.array(Y), np.array(centres_list)


def extract_FE_embeddings_ae(model, df, max_clips=MAX_CLIPS,
                              centre_beta=None, centre_gamma=None,
                              apply_combat=False):
    """Extract mean-pooled FE embeddings for ASW. Works for AE, DANN, ComBat+AE."""
    Z_FE, centres_list = [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips), desc="  FE embeds")
    ):
        if i >= max_clips: break
        A = _load_clip_raw(row)
        if A is None: continue
        try:
            centre_id    = CENTRE_MAP[row['centre']]
            centre_id_tf = tf.constant([centre_id], dtype=tf.int32)

            if apply_combat and centre_beta is not None:
                beta  = centre_beta.get(centre_id,  0.0)
                gamma = centre_gamma.get(centre_id, 1.0)
                A     = np.clip((A - beta) / (gamma + 1e-8), 0.0, 1.0)

            out    = model(A[None, ...], centre_id_tf, training=False)
            z_key  = 'z' if 'z' in out else 'z_FE_low'
            z      = out[z_key].numpy()[0]     # (T, D)
            Z_FE.append(z.mean(0))
            centres_list.append(centre_id)
        except Exception:
            continue

    return np.array(Z_FE), np.array(centres_list)


def extract_FE_embeddings_hca(model, df, max_clips=MAX_CLIPS):
    """Extract mean-pooled z_FE_low for HCA ASW."""
    Z_FE, centres_list = [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips), desc="  HCA FE embeds")
    ):
        if i >= max_clips: break
        A = _load_clip_raw(row)
        if A is None: continue
        try:
            centre_id    = CENTRE_MAP[row['centre']]
            centre_id_tf = tf.constant([centre_id], dtype=tf.int32)
            out  = model(A[None, ...], centre_id_tf, training=False)
            z    = out['z_FE_low'].numpy()[0]
            Z_FE.append(z.mean(0))
            centres_list.append(centre_id)
        except Exception:
            continue

    return np.array(Z_FE), np.array(centres_list)

# ============================================================
# 6. METRICS
# ============================================================
def compute_TOP_single(X, Y):
    """Single run of TOP evaluation. Returns accuracy."""
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_sc, Y, test_size=0.3, random_state=SEED, stratify=Y)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                             random_state=SEED)
    clf.fit(X_tr, Y_tr)
    return accuracy_score(Y_te, clf.predict(X_te))


def compute_TOP_with_stats(X, Y, n_runs=N_RUNS):
    """
    Run TOP N_RUNS times with different random seeds.
    Returns mean ± std (matching paper's ± format),
    plus bootstrap CI and permutation p-value on the last run.
    """
    accs = []
    for run in range(n_runs):
        np.random.seed(SEED + run)
        tf.random.set_seed(SEED + run)
        accs.append(compute_TOP_single(X, Y))

    # Reset seeds
    np.random.seed(SEED); tf.random.set_seed(SEED)

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)

    # Bootstrap CI and permutation test on the full feature set
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_sc, Y, test_size=0.3, random_state=SEED, stratify=Y)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                             random_state=SEED)
    clf.fit(X_tr, Y_tr)
    obs_acc = accuracy_score(Y_te, clf.predict(X_te))

    # Bootstrap CI
    boot_acc = []
    for _ in range(BOOTSTRAP_ITERS):
        Xb, Yb = resample(X_te, Y_te, random_state=None)
        boot_acc.append(accuracy_score(Yb, clf.predict(Xb)))
    ci_low, ci_high = np.percentile(boot_acc, [2.5, 97.5])

    # Permutation test
    perm_acc = []
    for _ in range(PERMUTATION_ITERS):
        clf_p = LogisticRegression(max_iter=500, random_state=SEED)
        clf_p.fit(X_tr, np.random.permutation(Y_tr))
        perm_acc.append(accuracy_score(Y_te, clf_p.predict(X_te)))
    p_value = np.mean(np.array(perm_acc) >= obs_acc)

    return mean_acc, std_acc, ci_low, ci_high, p_value, perm_acc


def compute_ASW(Z_FE, centres):
    """Average Silhouette Width on FE embeddings vs. centre labels."""
    if len(np.unique(centres)) < 2:
        return float('nan')
    scaler = StandardScaler()
    Z_sc   = scaler.fit_transform(Z_FE)
    return silhouette_score(Z_sc, centres, metric='euclidean',
                            sample_size=min(2000, len(Z_sc)),
                            random_state=SEED)

# ============================================================
# 7. WARM-UP HELPERS
# ============================================================
def _warmup(model, centre_id_val=0):
    """Build model graph with a dummy forward pass."""
    dummy_f = tf.zeros((1, T_WIN, IMG_SIZE, IMG_SIZE, 1))
    dummy_c = tf.constant([centre_id_val], dtype=tf.int32)
    _ = model(dummy_f, dummy_c, training=False)

def _warmup_plain(model):
    """Warm-up for models that don't take centre_id."""
    dummy_f = tf.zeros((1, T_WIN, IMG_SIZE, IMG_SIZE, 1))
    _ = model(dummy_f, training=False)

# ============================================================
# 8. PLOTS
# ============================================================
def plot_comparison_table(results_df,
                           save_path="/content/baseline_comparison.png"):
    """
    Grouped bar chart: TOP Accuracy (left axis) + ASW (right axis, line).
    Error bars show ± std from N_RUNS repeated evaluations.
    """
    labels  = results_df['method_label'].tolist()
    top_acc = results_df['top_acc_mean'].values * 100
    top_std = results_df['top_acc_std'].values  * 100
    asw     = results_df['asw'].values
    x       = np.arange(len(labels))

    # Colour-code ours vs baselines
    colors = ['#95a5a6', '#e67e22', '#e74c3c', '#2980b9', '#1abc9c']

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    bars = ax1.bar(x, top_acc, yerr=top_std, capsize=6,
                   color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)

    for bar, acc, std in zip(bars, top_acc, top_std):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + std + 0.4,
                 f"{acc:.1f}%",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.plot(x, asw, color='crimson', marker='D', linewidth=2.5,
             markersize=8, zorder=5, label='ASW (FE)')
    for xi, ai in zip(x, asw):
        if not np.isnan(ai):
            ax2.text(xi + 0.12, ai + 0.008, f"{ai:.2f}",
                     fontsize=9, color='crimson', va='bottom')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha='right', fontsize=10)
    ax1.set_ylabel('TOP Accuracy (%)', fontsize=12, color='#2c3e50')
    ax2.set_ylabel('ASW (FE)  [lower = more centre-invariant]',
                   fontsize=11, color='crimson')
    ax1.set_ylim(35, 112)
    ax2.set_ylim(0.15, 0.55)
    ax1.set_title(
        'Baseline Comparison\n'
        'Comparison with Harmonization & Domain Adaptation Methods',
        fontsize=13)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l)
                       for c, l in zip(colors, labels)]
    legend_elements.append(
        plt.Line2D([0], [0], color='crimson', marker='D',
                   linewidth=2, label='ASW (FE)'))
    ax1.legend(handles=legend_elements, loc='upper left',
               fontsize=9, framealpha=0.85)
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison chart saved → {save_path}")


def plot_null_distributions(null_dict,
                             save_path="/content/baseline_null_distributions.png"):
    """Permutation null distributions for each method."""
    n    = len(null_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)

    for ax, (method, (perm_acc, obs_acc)) in zip(axes, null_dict.items()):
        ax.hist(perm_acc, bins=25, color='steelblue', alpha=0.7,
                edgecolor='white')
        ax.axvline(obs_acc, color='crimson', lw=2,
                   label=f'Obs={obs_acc*100:.1f}%')
        ax.set_title(method, fontsize=9)
        ax.set_xlabel('Accuracy')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Permutation Null Distributions — Baseline Comparison',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Null distributions saved → {save_path}")

# ============================================================
# 9. MAIN — COMPARISON LOOP
# ============================================================
if __name__ == "__main__":

    print("=" * 65)
    print("BASELINE COMPARISON")
    print("Method | TOP Acc. (%) | ASW (FE) | Training")
    print("=" * 65)

    # ── Metadata & split ──────────────────────────────────────
    df = create_metadata()
    df_train, df_val, df_test = split_data(df)

    results_rows = []
    null_dict    = {}

    # ── Timing ────────────────────────────────────────────────
    timing = {}

    # ─────────────────────────────────────────────────────────
    # METHOD 1: Raw (no training)
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("METHOD: Raw (no training)")
    print("─" * 65)

    t0 = time.time()
    X_raw, Y_raw, cen_raw = extract_features_raw(df_test, MAX_CLIPS)
    timing["raw"] = time.time() - t0

    mean_acc, std_acc, ci_low, ci_high, p_val, perm_acc = \
        compute_TOP_with_stats(X_raw, Y_raw)

    # ASW on random encoder — use same raw extractor backbone
    raw_backbone = build_cnn_backbone(LATENT_DIM, "raw_asw")
    Z_raw_fe, cen_raw_fe = [], []
    for _, row in tqdm(df_test.iterrows(), total=min(len(df_test), MAX_CLIPS),
                       desc="  Raw ASW"):
        A = _load_clip_raw(row)
        if A is None: continue
        z = raw_backbone(A.reshape(-1, IMG_SIZE, IMG_SIZE, 1),
                         training=False).numpy().mean(0)
        Z_raw_fe.append(z)
        cen_raw_fe.append(CENTRE_MAP[row['centre']])
    asw_raw = compute_ASW(np.array(Z_raw_fe), np.array(cen_raw_fe))

    print(f"  TOP: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%  "
          f"[{ci_low*100:.2f}%, {ci_high*100:.2f}%]  p={p_val:.5f}")
    print(f"  ASW: {asw_raw:.4f}")

    null_dict["Raw"] = (perm_acc, mean_acc)
    results_rows.append({
        'method'        : 'raw',
        'method_label'  : 'Raw (no training)',
        'top_acc_mean'  : mean_acc,
        'top_acc_std'   : std_acc,
        'ci_low'        : ci_low,
        'ci_high'       : ci_high,
        'p_value'       : p_val,
        'asw'           : asw_raw,
        'training_speed': TRAINING_SPEED['raw'],
    })

    # ─────────────────────────────────────────────────────────
    # METHOD 2: ComBat + Autoencoder
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("METHOD: ComBat + Autoencoder")
    print("─" * 65)

    COMBAT_WEIGHTS = "/content/COMBAT_AE_best_model.weights.h5"
    plain_ae = PlainAEModel(LATENT_DIM)
    _warmup_plain(plain_ae)

    t0 = time.time()
    if Path(COMBAT_WEIGHTS).exists():
        print("  Loading pre-trained ComBat AE weights...")
        plain_ae.load_weights(COMBAT_WEIGHTS)
        # Recompute ComBat stats from training data
        centre_beta, centre_gamma = {}, {}
        for c_name, c_id in CENTRE_MAP.items():
            subset = df_train[df_train['centre'] == c_name]
            vals   = []
            for _, row in subset.iterrows():
                A = _load_clip_raw(row)
                if A is not None: vals.append(A.mean())
            if vals:
                grand = np.mean([
                    df_train[df_train['centre'] == n].apply(
                        lambda r: (_load_clip_raw(r) or np.zeros(1)).mean(),
                        axis=1).mean()
                    for n in CENTRE_MAP
                ])
                centre_beta[c_id]  = np.mean(vals) - grand
                centre_gamma[c_id] = np.std(vals) + 1e-8
    else:
        plain_ae, centre_beta, centre_gamma = train_combat_ae(
            plain_ae, df_train, df_val)
    timing["combat_ae"] = time.time() - t0

    X_cb, Y_cb, cen_cb = extract_features_combat_ae(
        plain_ae, centre_beta, centre_gamma, df_test, MAX_CLIPS)

    mean_acc, std_acc, ci_low, ci_high, p_val, perm_acc = \
        compute_TOP_with_stats(X_cb, Y_cb)

    Z_cb_fe, cen_cb_fe = extract_FE_embeddings_ae(
        plain_ae, df_test, MAX_CLIPS,
        centre_beta=centre_beta, centre_gamma=centre_gamma,
        apply_combat=True)
    asw_cb = compute_ASW(Z_cb_fe, cen_cb_fe)

    print(f"  TOP: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%  "
          f"[{ci_low*100:.2f}%, {ci_high*100:.2f}%]  p={p_val:.5f}")
    print(f"  ASW: {asw_cb:.4f}")

    null_dict["ComBat+AE"] = (perm_acc, mean_acc)
    results_rows.append({
        'method'        : 'combat_ae',
        'method_label'  : 'ComBat + Autoencoder',
        'top_acc_mean'  : mean_acc,
        'top_acc_std'   : std_acc,
        'ci_low'        : ci_low,
        'ci_high'       : ci_high,
        'p_value'       : p_val,
        'asw'           : asw_cb,
        'training_speed': TRAINING_SPEED['combat_ae'],
    })

    # ─────────────────────────────────────────────────────────
    # METHOD 3: DANN
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("METHOD: DANN")
    print("─" * 65)

    dann_model = DANNModel(LATENT_DIM, NUM_CENTRES)
    _warmup(dann_model)

    t0 = time.time()
    if Path(DANN_WEIGHTS_PATH).exists():
        print(f"  Loading pre-trained DANN weights from: {DANN_WEIGHTS_PATH}")
        dann_model.load_weights(DANN_WEIGHTS_PATH)
    else:
        dann_model = train_dann(dann_model, df_train, df_val)
        dann_model.load_weights(DANN_WEIGHTS_PATH)
    timing["dann"] = time.time() - t0

    X_dann, Y_dann, cen_dann = extract_features_dann(dann_model, df_test, MAX_CLIPS)
    mean_acc, std_acc, ci_low, ci_high, p_val, perm_acc = \
        compute_TOP_with_stats(X_dann, Y_dann)

    Z_dann_fe, cen_dann_fe = extract_FE_embeddings_ae(
        dann_model, df_test, MAX_CLIPS)
    asw_dann = compute_ASW(Z_dann_fe, cen_dann_fe)

    print(f"  TOP: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%  "
          f"[{ci_low*100:.2f}%, {ci_high*100:.2f}%]  p={p_val:.5f}")
    print(f"  ASW: {asw_dann:.4f}")

    null_dict["DANN"] = (perm_acc, mean_acc)
    results_rows.append({
        'method'        : 'dann',
        'method_label'  : 'DANN',
        'top_acc_mean'  : mean_acc,
        'top_acc_std'   : std_acc,
        'ci_low'        : ci_low,
        'ci_high'       : ci_high,
        'p_value'       : p_val,
        'asw'           : asw_dann,
        'training_speed': TRAINING_SPEED['dann'],
    })

    # ─────────────────────────────────────────────────────────
    # METHOD 4: Temporal AE (ours)
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("METHOD: Temporal AE (ours)")
    print("─" * 65)

    ae_model = TemporalAEModel(LATENT_DIM, NUM_CENTRES)
    _warmup(ae_model)

    t0 = time.time()
    if Path(AE_WEIGHTS_PATH).exists():
        print(f"  Loading pre-trained AE weights from: {AE_WEIGHTS_PATH}")
        ae_model.load_weights(AE_WEIGHTS_PATH)
    else:
        raise FileNotFoundError(
            f"AE weights not found at {AE_WEIGHTS_PATH}. "
            "Run autoencoder.py first.")
    timing["temporal_ae"] = time.time() - t0

    X_ae, Y_ae, cen_ae = extract_features_temporal_ae(ae_model, df_test, MAX_CLIPS)
    mean_acc, std_acc, ci_low, ci_high, p_val, perm_acc = \
        compute_TOP_with_stats(X_ae, Y_ae)

    Z_ae_fe, cen_ae_fe = extract_FE_embeddings_ae(ae_model, df_test, MAX_CLIPS)
    asw_ae = compute_ASW(Z_ae_fe, cen_ae_fe)

    print(f"  TOP: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%  "
          f"[{ci_low*100:.2f}%, {ci_high*100:.2f}%]  p={p_val:.5f}")
    print(f"  ASW: {asw_ae:.4f}")

    null_dict["Temporal AE"] = (perm_acc, mean_acc)
    results_rows.append({
        'method'        : 'temporal_ae',
        'method_label'  : 'Temporal AE (ours)',
        'top_acc_mean'  : mean_acc,
        'top_acc_std'   : std_acc,
        'ci_low'        : ci_low,
        'ci_high'       : ci_high,
        'p_value'       : p_val,
        'asw'           : asw_ae,
        'training_speed': TRAINING_SPEED['temporal_ae'],
    })

    # ─────────────────────────────────────────────────────────
    # METHOD 5: HCA (ours)
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("METHOD: HCA (ours)")
    print("─" * 65)

    hca_model = HCAEncoderModel(LATENT_DIM, NUM_CENTRES)
    _warmup(hca_model)

    t0 = time.time()
    if Path(HCA_WEIGHTS_PATH).exists():
        print(f"  Loading pre-trained HCA weights from: {HCA_WEIGHTS_PATH}")
        hca_model.load_weights(HCA_WEIGHTS_PATH)
    else:
        raise FileNotFoundError(
            f"HCA weights not found at {HCA_WEIGHTS_PATH}. "
            "Run hca_encoder.py first.")
    timing["hca"] = time.time() - t0

    X_hca, Y_hca, cen_hca = extract_features_hca(hca_model, df_test, MAX_CLIPS)
    mean_acc, std_acc, ci_low, ci_high, p_val, perm_acc = \
        compute_TOP_with_stats(X_hca, Y_hca)

    Z_hca_fe, cen_hca_fe = extract_FE_embeddings_hca(hca_model, df_test, MAX_CLIPS)
    asw_hca = compute_ASW(Z_hca_fe, cen_hca_fe)

    print(f"  TOP: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%  "
          f"[{ci_low*100:.2f}%, {ci_high*100:.2f}%]  p={p_val:.5f}")
    print(f"  ASW: {asw_hca:.4f}")

    null_dict["HCA"] = (perm_acc, mean_acc)
    results_rows.append({
        'method'        : 'hca',
        'method_label'  : 'HCA (ours)',
        'top_acc_mean'  : mean_acc,
        'top_acc_std'   : std_acc,
        'ci_low'        : ci_low,
        'ci_high'       : ci_high,
        'p_value'       : p_val,
        'asw'           : asw_hca,
        'training_speed': TRAINING_SPEED['hca'],
    })

    # ── Build results DataFrame ───────────────────────────────
    results_df = pd.DataFrame(results_rows)

    # ── Print Baseline Comparison Results ─────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 72)
    print(f"{'Method':<24} | {'TOP Acc. (%)':<22} | {'ASW (FE)':<10} | Training")
    print("-" * 72)

    for _, row in results_df.iterrows():
        top_str = (f"{row['top_acc_mean']*100:.2f} "
                   f"± {row['top_acc_std']*100:.2f}  "
                   f"[{row['ci_low']*100:.2f}, {row['ci_high']*100:.2f}]  "
                   f"p={row['p_value']:.4f}")
        print(f"{row['method_label']:<24} | {top_str:<38} | "
              f"{row['asw']:.4f}     | {row['training_speed']}")

    print("=" * 72)

    # ── Save CSV ──────────────────────────────────────────────
    csv_path = "/content/baseline_comparison_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # ── Plots ─────────────────────────────────────────────────
    plot_comparison_table(results_df)
    plot_null_distributions(null_dict)

    print("\nBaseline comparison complete.")