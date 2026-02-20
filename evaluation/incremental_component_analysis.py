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

# Trained weight paths — one per ablation configuration
# Each produced by training with the corresponding loss subset
WEIGHTS = {
    "rec_only"  : "/content/ablation_rec_only.weights.h5",
    "rec_scale" : "/content/ablation_rec_scale.weights.h5",
    "rec_spec"  : "/content/ablation_rec_spec.weights.h5",
    "rec_adv"   : "/content/ablation_rec_adv.weights.h5",
    "full_hca"  : "/content/HCA_best_model.weights.h5",
}

# Hyperparameters — identical across all configurations
IMG_SIZE     = 128
T_WIN        = 64
LATENT_DIM   = 128
BATCH_SIZE   = 4
EPOCHS       = 10
LR           = 2e-4
NUM_SEGMENTS = 4
SEGMENT_SIZE = T_WIN // NUM_SEGMENTS    # 16

# Loss weights (same as training scripts)
W_REC        = 0.1
W_SCALE      = 1.0
W_SPEC       = 0.5
W_ADV        = 1.0
W_BATCH      = 1.0
W_KL         = 0.01
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
NUM_CENTRES = len(CENTRE_MAP)

# Ablation configuration labels
ABLATION_CONFIGS = [
    {
        "key"         : "rec_only",
        "label"       : "Reconstruction only (L_rec)",
        "weights_key" : "rec_only",
        "arch"        : "hca",          # architecture to use for this config
        "use_scale"   : False,
        "use_spec"    : False,
        "use_adv"     : False,
        "use_hier"    : False,
    },
    {
        "key"         : "rec_scale",
        "label"       : "+ Scale regularization (L_scale)",
        "weights_key" : "rec_scale",
        "arch"        : "hca",
        "use_scale"   : True,
        "use_spec"    : False,
        "use_adv"     : False,
        "use_hier"    : False,
    },
    {
        "key"         : "rec_spec",
        "label"       : "+ Spectral regularization (L_spec)",
        "weights_key" : "rec_spec",
        "arch"        : "hca",
        "use_scale"   : True,
        "use_spec"    : True,
        "use_adv"     : False,
        "use_hier"    : False,
    },
    {
        "key"         : "rec_adv",
        "label"       : "+ Adversarial loss (L_adv)",
        "weights_key" : "rec_adv",
        "arch"        : "hca",
        "use_scale"   : True,
        "use_spec"    : True,
        "use_adv"     : True,
        "use_hier"    : False,
    },
    {
        "key"         : "full_hca",
        "label"       : "+ Hierarchical aggregation (full HCA)",
        "weights_key" : "full_hca",
        "arch"        : "hca",
        "use_scale"   : True,
        "use_spec"    : True,
        "use_adv"     : True,
        "use_hier"    : True,
    },
]

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
    """Subject-level 70/15/15 split — prevents data leakage."""
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
    print("=" * 60)

    return df_train, df_val, df_test

# ============================================================
# 2. SHARED MODEL BUILDING BLOCKS
# ============================================================
def build_cnn_backbone(latent_dim, name="encoder"):
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
        FE_cross     = self.FE_to_RE(z_FE, z_RE, z_RE, training=training)
        z_FE_out     = self.norm_FE(z_FE + self.gate_FE(z_FE) * FE_cross)
        RE_cross     = self.RE_to_FE(z_RE, z_FE, z_FE, training=training)
        z_RE_out     = self.norm_RE(z_RE + self.gate_RE(z_RE) * RE_cross)
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
        gamma  = self.gamma_emb(centre_id)
        beta   = self.beta_emb(centre_id)
        return gamma[:, None, :] * z_norm + beta[:, None, :]


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
# 3. HCA ENCODER  
# ============================================================
class HierarchicalLevel(keras.Model):
    """Single HCA level: FE encoder + RE encoder + cross-attention."""
    def __init__(self, latent_dim, num_centres, level_name):
        super().__init__(name=f"level_{level_name}")
        self.FE_encoder   = build_cnn_backbone(latent_dim, f"FE_{level_name}")
        self.FE_attention = TemporalAttention(latent_dim, name_prefix=f"FE_{level_name}")
        self.RE_encoder   = BayesianEncoder(latent_dim)
        self.RE_norm      = BatchConditionalNorm(num_centres, latent_dim)
        self.cross_attn   = CrossAttention(latent_dim, name_prefix=level_name)
        self.latent_dim   = latent_dim

    def call(self, frames, centre_id, training=False):
        if len(frames.shape) == 5:   # raw frames (B,T,H,W,1)
            z_FE = layers.TimeDistributed(self.FE_encoder)(frames)
            z_FE = self.FE_attention(z_FE, training=training)

            z_RE_list, mu_list, lv_list = [], [], []
            for ft in tf.unstack(frames, axis=1):
                zt, mt, lt = self.RE_encoder(ft)
                z_RE_list.append(zt); mu_list.append(mt); lv_list.append(lt)

            z_RE     = tf.stack(z_RE_list,  axis=1)
            mu_RE    = tf.stack(mu_list,    axis=1)
            lv_RE    = tf.stack(lv_list,    axis=1)
            z_RE     = self.RE_norm(z_RE, centre_id)
        else:                        # features (B,T,D) from lower level
            z_FE  = self.FE_attention(frames, training=training)
            z_RE  = frames
            mu_RE = tf.zeros_like(z_RE)
            lv_RE = tf.zeros_like(z_RE)

        z_FE_out, z_RE_out = self.cross_attn(z_FE, z_RE, training=training)
        return {'z_FE': z_FE_out, 'z_RE': z_RE_out, 'mu_RE': mu_RE, 'log_var_RE': lv_RE}


class HCAEncoderModel(keras.Model):
    """
    3-level HCA Encoder.
    Used for ALL ablation configs — components are toggled at inference
    time via the ablation config flags (use_scale, use_spec, etc.).
    The architecture is identical; only the training objective differs.
    """
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

        self.latent_dim  = latent_dim
        self.num_centres = num_centres

    def call(self, frames, centre_id, training=False):
        # Level 1 — frame-level
        out_low  = self.level_low(frames, centre_id, training=training)
        z_FE_low = out_low['z_FE']      # (B, T, D)
        z_RE_low = out_low['z_RE']

        # Level 2 — segment-level (pool T→4)
        z_mid_in = (self.pool_low_to_mid(z_FE_low) +
                    self.pool_low_to_mid(z_RE_low)) / 2.0   # (B, 4, D)
        out_mid  = self.level_mid(z_mid_in, centre_id, training=training)
        z_FE_mid = out_mid['z_FE']      # (B, 4, D)
        z_RE_mid = out_mid['z_RE']

        # Level 3 — global (pool 4→1)
        z_high_in  = (tf.reduce_mean(z_FE_mid, axis=1, keepdims=True) +
                      tf.reduce_mean(z_RE_mid, axis=1, keepdims=True)) / 2.0
        out_high   = self.level_high(z_high_in, centre_id, training=training)
        z_FE_high  = out_high['z_FE']   # (B, 1, D)
        z_RE_high  = out_high['z_RE']

        return {
            'z_FE_low'      : z_FE_low,
            'z_RE_low'      : z_RE_low,
            'mu_RE_low'     : out_low['mu_RE'],
            'log_var_RE_low': out_low['log_var_RE'],
            'z_FE_mid'      : z_FE_mid,
            'z_RE_mid'      : z_RE_mid,
            'z_FE_high'     : z_FE_high,
            'z_RE_high'     : z_RE_high,
        }

# ============================================================
# 4. ABLATION TRAINING
#    Each config trains on a subset of the full loss function.
#    Architecture is always the full HCA model —
#    only the loss terms differ.
# ============================================================
def tf_percentile(x, q):
    x = tf.sort(tf.reshape(x, [-1]))
    n = tf.size(x)
    k = tf.cast((q / 100.0) * tf.cast(n - 1, tf.float32), tf.int32)
    return x[k]

def scale_regularization(dz):
    mag = tf.norm(dz, axis=-1)
    med = tf_percentile(mag, 50.0)
    mad = tf_percentile(tf.abs(mag - med), 50.0) + 1e-6
    return tf.reduce_mean(tf.abs(mag - med) / mad)

def spectral_loss(dz):
    mag     = tf.norm(dz, axis=-1)
    mag     = mag - tf.reduce_mean(mag, axis=1, keepdims=True)
    fft     = tf.signal.rfft(mag)
    power   = tf.abs(fft) ** 2
    cutoff  = tf.cast(0.2 * tf.cast(tf.shape(power)[1], tf.float32), tf.int32)
    return tf.reduce_mean(power[:, cutoff:])

def kl_divergence(mu, log_var):
    return -0.5 * tf.reduce_mean(1 + log_var - mu ** 2 - tf.exp(log_var))

def hierarchical_consistency_loss(z_low, z_mid, z_high):
    z_low_pooled = tf.reduce_mean(
        tf.reshape(z_low,
                   [tf.shape(z_low)[0], NUM_SEGMENTS, -1, z_low.shape[-1]]),
        axis=2)
    z_mid_pooled = tf.reduce_mean(z_mid, axis=1, keepdims=True)
    return (tf.reduce_mean(tf.square(z_low_pooled - z_mid)) +
            tf.reduce_mean(tf.square(z_mid_pooled - z_high)))


def train_ablation_config(cfg, df_train, df_val,
                           epochs=EPOCHS,
                           steps_per_epoch=200,
                           val_steps=50):
    """
    Train the HCA model with only the loss components specified in cfg.

    cfg keys used here:
      use_scale  : include L_scale
      use_spec   : include L_spec
      use_adv    : include L_adv (adversarial discriminator on FE)
      use_hier   : include hierarchical consistency + mid/high level losses
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {cfg['label']}")
    print(f"  scale={cfg['use_scale']}  spec={cfg['use_spec']}  "
          f"adv={cfg['use_adv']}  hier={cfg['use_hier']}")
    print(f"{'='*60}")

    # Datasets
    dataset_train = _build_dataset(df_train)
    dataset_val   = _build_dataset(df_val)

    # Model
    model = HCAEncoderModel(LATENT_DIM, NUM_CENTRES)

    opt_enc  = keras.optimizers.Adam(LR)
    opt_disc = keras.optimizers.Adam(LR)

    best_val_loss = float('inf')
    weights_path  = WEIGHTS[cfg['weights_key']]

    @tf.function
    def train_step(frames, centre_id):
        with tf.GradientTape(persistent=True) as tape:
            out = model(frames, centre_id, training=True)

            z_FE_low  = out['z_FE_low']
            z_RE_low  = out['z_RE_low']
            mu_low    = out['mu_RE_low']
            lv_low    = out['log_var_RE_low']
            z_FE_mid  = out['z_FE_mid']
            z_FE_high = out['z_FE_high']

            # ── Reconstruction (always on) ────────────────────
            x_rec = layers.TimeDistributed(model.decoder)(z_FE_low)
            L_rec = tf.reduce_mean(tf.square(frames - x_rec))
            L_KL  = kl_divergence(mu_low, lv_low)

            L_total = W_REC * L_rec + W_KL * L_KL

            # ── Scale regularization ──────────────────────────
            dz_low = z_FE_low[:, 1:] - z_FE_low[:, :-1]
            if cfg['use_scale']:
                L_total += W_SCALE * scale_regularization(dz_low)

            # ── Spectral regularization ───────────────────────
            if cfg['use_spec']:
                L_total += W_SPEC * spectral_loss(dz_low)

            # ── Adversarial (FE discriminator) ────────────────
            pred_FE_low = model.discriminator_low(z_FE_low, training=True)
            L_disc_low  = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_FE_low, from_logits=True))

            if cfg['use_adv']:
                L_adv  = -tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        centre_id, pred_FE_low, from_logits=True))
                # RE classification
                z_RE_pooled = tf.reduce_mean(z_RE_low, axis=1)
                pred_RE_low = model.RE_classifier_low(z_RE_pooled)
                L_batch     = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        centre_id, pred_RE_low, from_logits=True))
                L_total += W_ADV * L_adv + W_BATCH * L_batch

            # ── Hierarchical levels (mid + high + consistency) ─
            if cfg['use_hier']:
                pred_FE_mid  = model.discriminator_mid(z_FE_mid,  training=True)
                pred_FE_high = model.discriminator_high(z_FE_high, training=True)

                L_adv_mid    = -tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        centre_id, pred_FE_mid, from_logits=True))
                L_adv_high   = -tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        centre_id, pred_FE_high, from_logits=True))

                z_RE_mid  = out['z_RE_mid']
                z_RE_high = out['z_RE_high']

                z_RE_mid_pooled  = tf.reduce_mean(z_RE_mid, axis=1)
                z_RE_high_pooled = tf.squeeze(z_RE_high, axis=1)

                pred_RE_mid  = model.RE_classifier_mid(z_RE_mid_pooled)
                pred_RE_high = model.RE_classifier_high(z_RE_high_pooled)

                L_batch_mid  = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        centre_id, pred_RE_mid, from_logits=True))
                L_batch_high = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        centre_id, pred_RE_high, from_logits=True))

                L_hier_FE = hierarchical_consistency_loss(
                    z_FE_low, z_FE_mid, z_FE_high)
                L_hier_RE = hierarchical_consistency_loss(
                    z_RE_low, z_RE_mid, z_RE_high)

                L_total += (W_ADV   * (L_adv_mid + L_adv_high)   +
                            W_BATCH * (L_batch_mid + L_batch_high) +
                            W_HIERARCHY * (L_hier_FE + L_hier_RE))

                # Mid/high discriminators contribute to disc loss
                L_disc_low += (
                    tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(
                            centre_id, pred_FE_mid, from_logits=True)) +
                    tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(
                            centre_id, pred_FE_high, from_logits=True))
                )

        # Encoder update
        enc_vars = (
            model.level_low.trainable_variables  +
            model.level_mid.trainable_variables  +
            model.level_high.trainable_variables +
            model.decoder.trainable_variables    +
            model.RE_classifier_low.trainable_variables  +
            model.RE_classifier_mid.trainable_variables  +
            model.RE_classifier_high.trainable_variables
        )
        grads = tape.gradient(L_total, enc_vars)
        opt_enc.apply_gradients(zip(grads, enc_vars))

        # Discriminator update
        disc_vars = (
            model.discriminator_low.trainable_variables  +
            model.discriminator_mid.trainable_variables  +
            model.discriminator_high.trainable_variables
        )
        grads_d = tape.gradient(L_disc_low, disc_vars)
        opt_disc.apply_gradients(zip(grads_d, disc_vars))

        del tape
        return L_total

    @tf.function
    def val_step(frames, centre_id):
        out      = model(frames, centre_id, training=False)
        z_FE_low = out['z_FE_low']
        mu_low   = out['mu_RE_low']
        lv_low   = out['log_var_RE_low']

        x_rec    = layers.TimeDistributed(model.decoder)(z_FE_low)
        L_rec    = tf.reduce_mean(tf.square(frames - x_rec))
        L_KL     = kl_divergence(mu_low, lv_low)
        L_total  = W_REC * L_rec + W_KL * L_KL

        dz_low = z_FE_low[:, 1:] - z_FE_low[:, :-1]
        if cfg['use_scale']:
            L_total += W_SCALE * scale_regularization(dz_low)
        if cfg['use_spec']:
            L_total += W_SPEC  * spectral_loss(dz_low)
        return L_total

    # ── Training loop ────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        train_losses = []
        for frames, cid in dataset_train.take(steps_per_epoch):
            loss = train_step(frames, cid)
            train_losses.append(float(loss))

        val_losses = []
        for frames, cid in dataset_val.take(val_steps):
            loss = val_step(frames, cid)
            val_losses.append(float(loss))

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)

        print(f"  Epoch {epoch}/{epochs} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_weights(weights_path)
            print(f"  Best model saved (val={best_val_loss:.4f})")

    print(f"  Training done. Best val loss: {best_val_loss:.4f}")
    return model


# ============================================================
# 5. DATA GENERATOR  
# ============================================================
def _build_dataset(df, batch_size=BATCH_SIZE):
    subj2clips = {}
    for _, r in df.iterrows():
        key = (r['centre'], r['subject'])
        subj2clips.setdefault(key, []).append(r)

    subjects = list(subj2clips.keys())

    def load_one():
        for _ in range(100):
            c, s       = random.choice(subjects)
            clip_info  = random.choice(subj2clips[(c, s)])
            A_path     = Path(clip_info['full_path_A'])
            V_path     = Path(clip_info['full_path_V'])
            if not A_path.exists():
                continue
            try:
                A = np.load(A_path)
                valid = (np.load(V_path).astype(bool)
                         if clip_info['valid_exists'] and V_path.exists()
                         else np.ones(len(A), dtype=bool))
                idx = np.where(valid)[0]
                if len(idx) < T_WIN + 2:
                    continue
                s_idx = random.randint(idx[0], idx[-1] - T_WIN + 1)
                seq   = A[s_idx : s_idx + T_WIN]
                seq   = tf.image.resize(seq[..., None],
                                        (IMG_SIZE, IMG_SIZE),
                                        method="area").numpy()
                return (seq / 255.0).astype(np.float32), CENTRE_MAP[c]
            except Exception:
                continue
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


# ============================================================
# 6. CLIP LOADER  
# ============================================================
def _load_clip(row):
    """Load one clip → (T, IMG, IMG, 1) float32 in [0,1], or None."""
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
# 7. FEATURE EXTRACTION
# ============================================================
def extract_features_HCA(model, df, max_clips=MAX_CLIPS):
    """
    Extract delta features from the HCA model (all 3 levels combined).
    Returns X (N, feat_dim), Y (N,), centres (N,).
    """
    X, Y, centres_list = [], [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips),
             desc="  Extracting features")
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
            A_batch      = A[None, ...]                  # (1,T,H,W,1)

            out       = model(A_batch, centre_id_tf, training=False)
            z_FE_low  = out['z_FE_low'].numpy()[0]       # (T, D)
            z_FE_mid  = out['z_FE_mid'].numpy()[0]       # (4, D)
            z_FE_high = out['z_FE_high'].numpy()[0]      # (1, D)

            # Combine all hierarchical levels
            z_combined = np.concatenate([
                z_FE_low,
                np.repeat(z_FE_mid,  SEGMENT_SIZE, axis=0),
                np.repeat(z_FE_high, T,            axis=0)
            ], axis=-1)                                  # (T, 3D)

            # POSITIVE
            dz   = z_combined[1:] - z_combined[:-1]
            feat = np.concatenate([
                dz.mean(0), dz.std(0),
                np.percentile(dz, [25, 50, 75], axis=0).reshape(-1)
            ])
            X.append(feat); Y.append(1); centres_list.append(centre_id)

            # NEGATIVE (shuffle in z-space)
            perm    = np.random.permutation(T)
            z_shuf  = z_combined[perm]
            dz_shuf = z_shuf[1:] - z_shuf[:-1]
            feat_s  = np.concatenate([
                dz_shuf.mean(0), dz_shuf.std(0),
                np.percentile(dz_shuf, [25, 50, 75], axis=0).reshape(-1)
            ])
            X.append(feat_s); Y.append(0); centres_list.append(centre_id)

        except Exception:
            continue

    return np.array(X), np.array(Y), np.array(centres_list)


def extract_FE_embeddings(model, df, max_clips=MAX_CLIPS):
    """
    Extract pooled FE embeddings for ASW computation.

    Returns:
        Z_FE     : (N, D)  — one row per clip (mean-pooled over T)
        centres  : (N,)    — centre label per clip
    """
    Z_FE, centres_list = [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips),
             desc="  FE embeddings")
    ):
        if i >= max_clips:
            break
        A = _load_clip(row)
        if A is None:
            continue

        try:
            centre_id    = CENTRE_MAP[row['centre']]
            centre_id_tf = tf.constant([centre_id], dtype=tf.int32)

            out      = model(A[None, ...], centre_id_tf, training=False)
            z_FE_low = out['z_FE_low'].numpy()[0]        # (T, D)

            # Pool over time to get a single clip-level representation
            z_pooled = z_FE_low.mean(0)                  # (D,)
            Z_FE.append(z_pooled)
            centres_list.append(centre_id)

        except Exception:
            continue

    return np.array(Z_FE), np.array(centres_list)

# ============================================================
# 8. METRICS
# ============================================================
def compute_TOP(X, Y):
    """
    Temporal Order Prediction:
      - 70/30 stratified train/test split
      - Logistic Regression classifier
      - Bootstrap 95% CI (BOOTSTRAP_ITERS)
      - Permutation test p-value (PERMUTATION_ITERS)

    Returns:
        acc, ci_low, ci_high, p_value
    """
    scaler    = StandardScaler()
    X_sc      = scaler.fit_transform(X)

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_sc, Y, test_size=0.3, random_state=SEED, stratify=Y)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                             random_state=SEED)
    clf.fit(X_tr, Y_tr)
    acc = accuracy_score(Y_te, clf.predict(X_te))

    # Bootstrap CI
    boot_acc = []
    for _ in range(BOOTSTRAP_ITERS):
        Xb, Yb = resample(X_te, Y_te, random_state=None)
        boot_acc.append(accuracy_score(Yb, clf.predict(Xb)))
    ci_low, ci_high = np.percentile(boot_acc, [2.5, 97.5])

    # Permutation test
    perm_acc = []
    for _ in range(PERMUTATION_ITERS):
        Yp    = np.random.permutation(Y_tr)
        clf_p = LogisticRegression(max_iter=500, random_state=SEED)
        clf_p.fit(X_tr, Yp)
        perm_acc.append(accuracy_score(Y_te, clf_p.predict(X_te)))
    p_value = np.mean(np.array(perm_acc) >= acc)

    return acc, ci_low, ci_high, p_value, perm_acc


def compute_ASW(Z_FE, centres):
    """
    Average Silhouette Width on FE embeddings w.r.t. centre labels.

    Lower ASW → embeddings are NOT clustered by centre
             → more centre-invariant (better for our goal).

    sklearn's silhouette_score returns values in [-1, 1].
    We report the raw value so lower = more invariant.
    """
    if len(np.unique(centres)) < 2:
        print("  WARNING: Only one centre in this split — ASW undefined.")
        return float('nan')

    scaler = StandardScaler()
    Z_sc   = scaler.fit_transform(Z_FE)
    asw    = silhouette_score(Z_sc, centres, metric='euclidean',
                              sample_size=min(2000, len(Z_sc)),
                              random_state=SEED)
    return asw

# ============================================================
# 9. PLOTS
# ============================================================
def plot_ablation_table(results_df,
                        save_path="/content/ablation_table.png"):
    """
    Dual-axis bar + line chart:
      - Left  axis : TOP Accuracy (%)
      - Right axis : ASW (FE)
    """
    labels   = results_df['label'].tolist()
    top_acc  = results_df['top_acc'].values * 100
    ci_low   = results_df['ci_low'].values  * 100
    ci_high  = results_df['ci_high'].values * 100
    asw      = results_df['asw'].values

    x   = np.arange(len(labels))
    err = np.array([top_acc - ci_low, ci_high - top_acc])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    bars = ax1.bar(x, top_acc, yerr=err, capsize=5,
                   color='steelblue', alpha=0.80,
                   label='TOP Accuracy (%)')

    # Annotate accuracy on top of each bar
    for bar, acc in zip(bars, top_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + err[1, list(top_acc).index(acc)] + 0.3,
                 f"{acc:.1f}%",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.plot(x, asw, color='crimson', marker='o', linewidth=2,
             markersize=7, label='ASW (FE)')
    ax2.set_ylim(0, max(asw) * 1.4)

    # Annotate ASW values
    for xi, ai in zip(x, asw):
        ax2.text(xi, ai + 0.01, f"{ai:.2f}",
                 ha='center', va='bottom', fontsize=9, color='crimson')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha='right', fontsize=9)
    ax1.set_ylabel('TOP Accuracy (%)', fontsize=12, color='steelblue')
    ax2.set_ylabel('ASW (FE)  [lower = more centre-invariant]',
                   fontsize=11, color='crimson')
    ax1.set_title('Ablation Study\n'
                  'Progressive Component Contribution to TOP Acc. & ASW(FE)',
                  fontsize=13)
    ax1.set_ylim(40, 108)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=10)

    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nAblation chart saved → {save_path}")


def plot_null_distributions(null_dict,
                             save_path="/content/ablation_null_distributions.png"):
    """Permutation null distribution for each ablation config."""
    n = len(null_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)

    for ax, (label, (perm_acc, obs_acc)) in zip(axes, null_dict.items()):
        ax.hist(perm_acc, bins=25, color='steelblue', alpha=0.7,
                edgecolor='white')
        ax.axvline(obs_acc, color='crimson', lw=2,
                   label=f'Obs={obs_acc*100:.1f}%')
        ax.set_title(label.replace("+ ", "+\n"), fontsize=8)
        ax.set_xlabel('Accuracy')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Permutation Null Distributions — Ablation Configs',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Null distributions saved → {save_path}")

# ============================================================
# 10. MAIN — ABLATION LOOP
# ============================================================
if __name__ == "__main__":

    print("=" * 65)
    print("ABLATION STUDY")
    print("Configuration | TOP Acc. (%) | ASW (FE)")
    print("=" * 65)

    # ── Metadata & split ──────────────────────────────────────
    df = create_metadata()
    df_train, df_val, df_test = split_data(df)

    results_rows = []
    null_dict    = {}

    # ── Ablation loop ─────────────────────────────────────────
    for cfg in ABLATION_CONFIGS:
        print(f"\n{'─'*65}")
        print(f"CONFIG: {cfg['label']}")
        print(f"{'─'*65}")

        weights_path = WEIGHTS[cfg['weights_key']]

        # ── Option A: use pre-trained weights (skip training) ──
        if Path(weights_path).exists():
            print(f"  Loading existing weights from: {weights_path}")
            model = HCAEncoderModel(LATENT_DIM, NUM_CENTRES)
            # Warm-up to build graph before loading weights
            dummy_f  = tf.zeros((1, T_WIN, IMG_SIZE, IMG_SIZE, 1))
            dummy_c  = tf.constant([0], dtype=tf.int32)
            _        = model(dummy_f, dummy_c, training=False)
            model.load_weights(weights_path)

        # ── Option B: train from scratch ───────────────────────
        else:
            print(f"  Weights not found. Training from scratch...")
            model = train_ablation_config(
                cfg, df_train, df_val,
                epochs=EPOCHS,
                steps_per_epoch=200,
                val_steps=50
            )
            # Load best checkpoint saved during training
            dummy_f = tf.zeros((1, T_WIN, IMG_SIZE, IMG_SIZE, 1))
            dummy_c = tf.constant([0], dtype=tf.int32)
            _       = model(dummy_f, dummy_c, training=False)
            model.load_weights(weights_path)

        # ── Extract TOP features from test set ─────────────────
        print("\n  Extracting TOP features (test set)...")
        X, Y, centres = extract_features_HCA(model, df_test, MAX_CLIPS)

        if len(X) == 0:
            print(f"  WARNING: No features for {cfg['label']}. Skipping.")
            continue

        # ── Compute TOP accuracy ───────────────────────────────
        print(f"  Computing TOP (samples={len(X)})...")
        acc, ci_low, ci_high, p_val, perm_acc = compute_TOP(X, Y)

        # ── Extract FE embeddings + compute ASW ───────────────
        print("  Computing ASW (FE embeddings)...")
        Z_FE, cen = extract_FE_embeddings(model, df_test, MAX_CLIPS)
        asw       = compute_ASW(Z_FE, cen)

        print(f"\n  TOP Acc : {acc*100:.2f}%  "
              f"[{ci_low*100:.2f}%, {ci_high*100:.2f}%]  p={p_val:.5f}")
        print(f"  ASW(FE) : {asw:.4f}")

        # Store
        null_dict[cfg['label']] = (perm_acc, acc)
        results_rows.append({
            'label'   : cfg['label'],
            'top_acc' : acc,
            'ci_low'  : ci_low,
            'ci_high' : ci_high,
            'p_value' : p_val,
            'asw'     : asw,
        })

    # ── Results DataFrame ─────────────────────────────────────
    results_df = pd.DataFrame(results_rows)

    # ── Print Incremental Component Analysis─────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("ABLATION STUDY RESULTS")
    print("=" * 65)
    print(f"{'Configuration':<46} | {'TOP Acc. (%)':<14} | {'ASW (FE)':<8}")
    print("-" * 65)

    for _, row in results_df.iterrows():
        top_str = (f"{row['top_acc']*100:.2f}  "
                   f"[{row['ci_low']*100:.2f}, {row['ci_high']*100:.2f}]  "
                   f"p={row['p_value']:.4f}")
        print(f"{row['label']:<46} | {top_str:<28} | {row['asw']:.4f}")

    print("=" * 65)

    # ── Save CSV ──────────────────────────────────────────────
    csv_path = "/content/ablation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # ── Plots ─────────────────────────────────────────────────
    plot_ablation_table(results_df)
    plot_null_distributions(null_dict)

    print("\nAblation study complete.")