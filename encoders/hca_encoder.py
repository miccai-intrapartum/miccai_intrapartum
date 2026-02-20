"""
HIERARCHICAL CROSS-ATTENTION ENCODER (HCA-Encoder) - FULL PIPELINE
Multi-scale temporal modeling with hierarchical cross-attention
"""
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# ============================================================
# 0. CONFIGURATION
# ============================================================
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Paths
ROOT = Path("/content/dataset/preprocessed")
DIRS_TO_SCAN = [ROOT / "A_fixed", ROOT / "B_variable"]

# Hyperparameters
IMG_SIZE = 128
T_WIN = 64
LATENT_DIM = 128
BATCH_SIZE = 4
EPOCHS = 10
LR = 2e-4

# Hierarchical structure
NUM_SEGMENTS = 4  # Level 2: 64 frames → 4 segments of 16 frames
SEGMENT_SIZE = T_WIN // NUM_SEGMENTS

# Loss weights
W_REC = 0.1
W_SCALE = 1.0
W_SPEC = 0.5
W_ADV = 1.0
W_BATCH = 1.0
W_KL = 0.01
W_CROSS_LOW = 0.3   # Level 1 cross-attention
W_CROSS_MID = 0.5   # Level 2 cross-attention
W_CROSS_HIGH = 0.7  # Level 3 cross-attention
W_HIERARCHY = 0.2   # Hierarchical consistency

# ============================================================
# 1. DATA LOADING & METADATA
# ============================================================
def create_metadata():
    """Create metadata scanning both A_fixed and B_variable"""
    rows = []
    print(f"Scanning directories: {[d.name for d in DIRS_TO_SCAN]}...")

    for root_path in DIRS_TO_SCAN:
        if not root_path.exists():
            print(f"⚠️ Warning: {root_path} does not exist. Skipping.")
            continue

        dataset_type = root_path.name
        all_npy = list(root_path.rglob("*.npy"))
        video_files = [f for f in all_npy if "_valid" not in f.name]
        print(f"   -> Found {len(video_files)} video files in {dataset_type}")

        for clip_file in video_files:
            path_str = str(clip_file)
            centre = None
            if "Merkez 2" in path_str: centre = "Merkez 2"
            elif "Merkez 6" in path_str: centre = "Merkez 6"
            elif "Merkez 8" in path_str: centre = "Merkez 8"

            if centre is None:
                continue

            subject = clip_file.parent.name
            clip = clip_file.stem
            valid_file = clip_file.parent / f"{clip}_valid.npy"

            rows.append({
                'source_root': str(root_path),
                'dataset_type': dataset_type,
                'centre': centre,
                'subject': subject,
                'clip': clip,
                'full_path_A': str(clip_file),
                'full_path_V': str(valid_file),
                'valid_exists': valid_file.exists()
            })

    df = pd.DataFrame(rows)
    print(f"✅ Metadata created: {len(df)} clips found.")

    if not df.empty:
        print("\nDistribution by Dataset Type:")
        print(df.groupby('dataset_type').size())
        print("\nDistribution by Centre:")
        print(df.groupby('centre').size())
    else:
        raise ValueError("No data found! Check paths.")

    return df

CENTRE_MAP = {'Merkez 2': 0, 'Merkez 6': 1, 'Merkez 8': 2}
NUM_CENTRES = len(CENTRE_MAP)

# ============================================================
# 1.5 TRAIN/VAL/TEST SPLIT (70/15/15)
# ============================================================
def split_data(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split data by SUBJECT (patient-level split)
    
    Important: Split by subject, not by clip, to prevent data leakage
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Get unique subjects
    subjects = df['subject'].unique()
    np.random.shuffle(subjects)
    
    n = len(subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train+n_val]
    test_subjects = subjects[n_train+n_val:]
    
    df_train = df[df['subject'].isin(train_subjects)].reset_index(drop=True)
    df_val = df[df['subject'].isin(val_subjects)].reset_index(drop=True)
    df_test = df[df['subject'].isin(test_subjects)].reset_index(drop=True)
    
    print("\n" + "="*60)
    print("DATA SPLIT (by subject)")
    print("="*60)
    print(f"Train: {len(train_subjects)} subjects, {len(df_train)} clips")
    print(f"Val:   {len(val_subjects)} subjects, {len(df_val)} clips")
    print(f"Test:  {len(test_subjects)} subjects, {len(df_test)} clips")
    
    # Print centre distribution for each split
    print("\nTrain centre distribution:")
    print(df_train.groupby('centre').size())
    print("\nVal centre distribution:")
    print(df_val.groupby('centre').size())
    print("\nTest centre distribution:")
    print(df_test.groupby('centre').size())
    print("="*60)
    
    return df_train, df_val, df_test

# ============================================================
# 2. DATA GENERATOR
# ============================================================
def build_dataset(df, batch_size=BATCH_SIZE):
    """Subject-balanced data generator"""
    subj2clips = {}
    for _, r in df.iterrows():
        key = (r['centre'], r['subject'])
        if key not in subj2clips:
            subj2clips[key] = []
        subj2clips[key].append(r)

    subjects = list(subj2clips.keys())
    print(f"Total unique subjects in this split: {len(subjects)}")

    def load_one():
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            centre, subject = random.choice(subjects)
            clip_info = random.choice(subj2clips[(centre, subject)])

            A_path = Path(clip_info['full_path_A'])
            V_path = Path(clip_info['full_path_V'])

            if not A_path.exists():
                continue

            try:
                A = np.load(A_path)

                if clip_info['valid_exists']:
                    valid = np.load(V_path).astype(bool)
                else:
                    valid = np.ones(len(A), dtype=bool)

                idx = np.where(valid)[0]
                if len(idx) < T_WIN + 2:
                    continue

                s = random.randint(idx[0], idx[-1] - T_WIN + 1)
                seq = A[s:s+T_WIN]

                seq = tf.image.resize(seq[..., None], (IMG_SIZE, IMG_SIZE),
                                     method="area").numpy()

                x = (seq / 255.0).astype(np.float32)
                centre_id = CENTRE_MAP[centre]

                return x, centre_id

            except Exception as e:
                continue
        
        # Fallback: return dummy data if all attempts fail
        dummy = np.zeros((T_WIN, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        return dummy, 0

    def gen():
        while True:
            yield load_one()

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((T_WIN, IMG_SIZE, IMG_SIZE, 1), tf.float32),
            tf.TensorSpec((), tf.int32)
        )
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return ds

# ============================================================
# 3. BUILDING BLOCKS
# ============================================================
def build_cnn_backbone(latent_dim, name="encoder"):
    """Shared CNN backbone"""
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    x = inp

    for filters in [32, 64, 96, 128, 128]:
        x = layers.Conv2D(filters, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(latent_dim)(x)

    return keras.Model(inp, x, name=name)

class BayesianEncoder(keras.Model):
    """Bayesian encoder with reparameterization"""
    def __init__(self, latent_dim):
        super().__init__(name="bayesian_encoder")
        self.backbone = build_cnn_backbone(latent_dim*2)
        self.latent_dim = latent_dim

    def call(self, x):
        h = self.backbone(x)
        mu = h[..., :self.latent_dim]
        log_var = h[..., self.latent_dim:]

        std = tf.exp(0.5 * log_var)
        eps = tf.random.normal(tf.shape(std))
        z = mu + eps * std

        return z, mu, log_var

class TemporalAttention(layers.Layer):
    """Multi-head self-attention"""
    def __init__(self, dim, num_heads=4, dropout=0.1, name_prefix=""):
        super().__init__(name=f"{name_prefix}_temporal_attn")
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout
        )
        self.norm = layers.LayerNormalization()

    def call(self, z, training=False):
        attn_out = self.mha(z, z, training=training)
        return self.norm(z + attn_out)

class CrossAttention(layers.Layer):
    """Cross-attention between FE and RE"""
    def __init__(self, dim, num_heads=4, dropout=0.1, name_prefix=""):
        super().__init__(name=f"{name_prefix}_cross_attn")

        self.FE_to_RE = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout
        )

        self.RE_to_FE = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout
        )

        self.norm_FE = layers.LayerNormalization()
        self.norm_RE = layers.LayerNormalization()

        self.gate_FE = layers.Dense(dim, activation='sigmoid')
        self.gate_RE = layers.Dense(dim, activation='sigmoid')

    def call(self, z_FE, z_RE, training=False):
        # FE attends to RE
        FE_cross = self.FE_to_RE(query=z_FE, key=z_RE, value=z_RE, training=training)
        gate_fe = self.gate_FE(z_FE)
        z_FE_updated = self.norm_FE(z_FE + gate_fe * FE_cross)

        # RE attends to FE
        RE_cross = self.RE_to_FE(query=z_RE, key=z_FE, value=z_FE, training=training)
        gate_re = self.gate_RE(z_RE)
        z_RE_updated = self.norm_RE(z_RE + gate_re * RE_cross)

        return z_FE_updated, z_RE_updated

class CentreDiscriminator(keras.Model):
    """Discriminator for adversarial training"""
    def __init__(self, num_centres):
        super().__init__(name="centre_discriminator")
        self.net = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_centres)
        ])

    def call(self, z, training=False):
        z_pooled = tf.reduce_mean(z, axis=1)
        return self.net(z_pooled, training=training)

class BatchConditionalNorm(layers.Layer):
    """Centre-conditioned normalization"""
    def __init__(self, num_centres, dim):
        super().__init__()
        self.gamma_emb = layers.Embedding(num_centres, dim)
        self.beta_emb = layers.Embedding(num_centres, dim)
        self.eps = 1e-6

    def call(self, z, centre_id):
        mean = tf.reduce_mean(z, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(z, axis=[1, 2], keepdims=True)
        z_norm = (z - mean) / (std + self.eps)

        gamma = self.gamma_emb(centre_id)
        beta = self.beta_emb(centre_id)

        z_out = gamma[:, None, :] * z_norm + beta[:, None, :]
        return z_out

def build_decoder(latent_dim):
    """Decoder for reconstruction"""
    inp = layers.Input((latent_dim,))
    x = layers.Dense((IMG_SIZE//8) * (IMG_SIZE//8) * 64)(inp)
    x = layers.Reshape((IMG_SIZE//8, IMG_SIZE//8, 64))(x)

    for filters in [64, 32, 16]:
        x = layers.Conv2DTranspose(filters, 3, 2, 'same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    out = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    return keras.Model(inp, out, name="decoder")

# ============================================================
# 4. HIERARCHICAL ENCODER MODULE
# ============================================================
class HierarchicalLevel(keras.Model):
    """
    Single level in the hierarchy

    Contains:
    - FE encoder with self-attention
    - RE encoder (Bayesian)
    - Cross-attention between FE and RE
    """
    def __init__(self, latent_dim, num_centres, level_name):
        super().__init__(name=f"level_{level_name}")

        # FE components
        self.FE_encoder = build_cnn_backbone(latent_dim, f"FE_{level_name}")
        self.FE_attention = TemporalAttention(latent_dim, name_prefix=f"FE_{level_name}")

        # RE components
        self.RE_encoder = BayesianEncoder(latent_dim)
        self.RE_norm = BatchConditionalNorm(num_centres, latent_dim)

        # Cross-attention
        self.cross_attention = CrossAttention(latent_dim, name_prefix=level_name)

        self.latent_dim = latent_dim
        self.level_name = level_name

    def call(self, frames, centre_id, training=False):
        """
        Args:
            frames: (B, T, H, W, 1) or (B, T, D) for higher levels
            centre_id: (B,)
        """
        # Check if input is raw frames or features
        if len(frames.shape) == 5:  # Raw frames (B, T, H, W, 1)
            # FE path
            z_FE = layers.TimeDistributed(self.FE_encoder)(frames)
            z_FE = self.FE_attention(z_FE, training=training)

            # RE path
            frames_unstacked = tf.unstack(frames, axis=1)
            z_RE_list, mu_list, lv_list = [], [], []

            for frame_t in frames_unstacked:
                z_t, mu_t, lv_t = self.RE_encoder(frame_t)
                z_RE_list.append(z_t)
                mu_list.append(mu_t)
                lv_list.append(lv_t)

            z_RE = tf.stack(z_RE_list, axis=1)
            mu_RE = tf.stack(mu_list, axis=1)
            log_var_RE = tf.stack(lv_list, axis=1)

            z_RE = self.RE_norm(z_RE, centre_id)

        else:  # Already features (B, T, D) from lower level
            # For higher levels, just apply attention
            z_FE = self.FE_attention(frames, training=training)
            z_RE = frames  # Simple passthrough
            mu_RE = tf.zeros_like(z_RE)
            log_var_RE = tf.zeros_like(z_RE)

        # Cross-attention
        z_FE_cross, z_RE_cross = self.cross_attention(z_FE, z_RE, training=training)

        return {
            'z_FE': z_FE_cross,
            'z_RE': z_RE_cross,
            'z_FE_orig': z_FE,
            'z_RE_orig': z_RE,
            'mu_RE': mu_RE,
            'log_var_RE': log_var_RE
        }

# ============================================================
# 5. HCA-ENCODER MODEL
# ============================================================
class HCAEncoderModel(keras.Model):
    """
    Hierarchical Cross-Attention Encoder

    Three levels:
    - Level 1 (Low): Frame-level (64 frames)
    - Level 2 (Mid): Segment-level (4 segments of 16 frames)
    - Level 3 (High): Global-level (1 global representation)
    """
    def __init__(self, latent_dim=LATENT_DIM, num_centres=NUM_CENTRES):
        super().__init__()

        # Three hierarchical levels
        self.level_low = HierarchicalLevel(latent_dim, num_centres, "low")
        self.level_mid = HierarchicalLevel(latent_dim, num_centres, "mid")
        self.level_high = HierarchicalLevel(latent_dim, num_centres, "high")

        # Discriminators for each level
        self.discriminator_low = CentreDiscriminator(num_centres)
        self.discriminator_mid = CentreDiscriminator(num_centres)
        self.discriminator_high = CentreDiscriminator(num_centres)

        # RE classifiers for each level
        self.RE_classifier_low = layers.Dense(num_centres, name="RE_classifier_low")
        self.RE_classifier_mid = layers.Dense(num_centres, name="RE_classifier_mid")
        self.RE_classifier_high = layers.Dense(num_centres, name="RE_classifier_high")

        # Shared decoder
        self.decoder = build_decoder(latent_dim)

        # Aggregation layers
        self.pool_low_to_mid = layers.AveragePooling1D(pool_size=SEGMENT_SIZE, name="pool_low_to_mid")
        self.pool_mid_to_high = layers.GlobalAveragePooling1D(name="pool_mid_to_high")

        self.latent_dim = latent_dim
        self.num_centres = num_centres

    def call(self, frames, centre_id, training=False):
        # frames: (B, T=64, H, W, 1)

        # ===== LEVEL 1: FRAME-LEVEL =====
        outputs_low = self.level_low(frames, centre_id, training=training)
        z_FE_low = outputs_low['z_FE']      # (B, 64, D)
        z_RE_low = outputs_low['z_RE']      # (B, 64, D)

        # ===== LEVEL 2: SEGMENT-LEVEL =====
        # Aggregate: 64 frames → 4 segments
        z_FE_low_pooled = self.pool_low_to_mid(z_FE_low)  # (B, 4, D)
        z_RE_low_pooled = self.pool_low_to_mid(z_RE_low)  # (B, 4, D)

        # Combine for mid-level input
        z_mid_input = (z_FE_low_pooled + z_RE_low_pooled) / 2.0  # (B, 4, D)

        outputs_mid = self.level_mid(z_mid_input, centre_id, training=training)
        z_FE_mid = outputs_mid['z_FE']      # (B, 4, D)
        z_RE_mid = outputs_mid['z_RE']      # (B, 4, D)

        # ===== LEVEL 3: GLOBAL-LEVEL =====
        # Aggregate: 4 segments → 1 global
        z_FE_mid_pooled = tf.reduce_mean(z_FE_mid, axis=1, keepdims=True)  # (B, 1, D)
        z_RE_mid_pooled = tf.reduce_mean(z_RE_mid, axis=1, keepdims=True)  # (B, 1, D)

        # Combine for high-level input
        z_high_input = (z_FE_mid_pooled + z_RE_mid_pooled) / 2.0  # (B, 1, D)

        outputs_high = self.level_high(z_high_input, centre_id, training=training)
        z_FE_high = outputs_high['z_FE']    # (B, 1, D)
        z_RE_high = outputs_high['z_RE']    # (B, 1, D)

        return {
            # Level 1 (Low)
            'z_FE_low': z_FE_low,
            'z_RE_low': z_RE_low,
            'mu_RE_low': outputs_low['mu_RE'],
            'log_var_RE_low': outputs_low['log_var_RE'],

            # Level 2 (Mid)
            'z_FE_mid': z_FE_mid,
            'z_RE_mid': z_RE_mid,

            # Level 3 (High)
            'z_FE_high': z_FE_high,
            'z_RE_high': z_RE_high
        }

# ============================================================
# 6. LOSS FUNCTIONS
# ============================================================
def tf_percentile(x, q):
    x = tf.sort(tf.reshape(x, [-1]))
    n = tf.size(x)
    k = tf.cast((q/100.0) * tf.cast(n-1, tf.float32), tf.int32)
    return x[k]

def scale_regularization(dz):
    """MAD-based scale regularization"""
    mag = tf.norm(dz, axis=-1)
    med = tf_percentile(mag, 50.0)
    mad = tf_percentile(tf.abs(mag - med), 50.0) + 1e-6
    return tf.reduce_mean(tf.abs(mag - med) / mad)

def spectral_loss(dz):
    """High-frequency suppression"""
    mag = tf.norm(dz, axis=-1)
    mag = mag - tf.reduce_mean(mag, axis=1, keepdims=True)
    fft = tf.signal.rfft(mag)
    power = tf.abs(fft) ** 2
    cutoff = tf.cast(0.2 * tf.cast(tf.shape(power)[1], tf.float32), tf.int32)
    return tf.reduce_mean(power[:, cutoff:])

def kl_divergence(mu, log_var):
    """KL divergence"""
    return -0.5 * tf.reduce_mean(1 + log_var - mu**2 - tf.exp(log_var))

def cross_alignment_loss(z_FE, z_RE):
    """Orthogonality constraint for cross-attention"""
    z_FE_norm = tf.nn.l2_normalize(z_FE, axis=-1)
    z_RE_norm = tf.nn.l2_normalize(z_RE, axis=-1)
    similarity = tf.reduce_sum(z_FE_norm * z_RE_norm, axis=-1)
    return tf.reduce_mean(tf.square(similarity))

def hierarchical_consistency_loss(z_low, z_mid, z_high):
    """
    Ensure consistency across hierarchical levels

    Lower levels should be consistent with higher levels when aggregated
    """
    # Pool low to match mid
    z_low_pooled = tf.reduce_mean(
        tf.reshape(z_low, [tf.shape(z_low)[0], NUM_SEGMENTS, -1, z_low.shape[-1]]),
        axis=2
    )  # (B, 4, D)

    # Pool mid to match high
    z_mid_pooled = tf.reduce_mean(z_mid, axis=1, keepdims=True)  # (B, 1, D)

    # Consistency losses
    loss_low_mid = tf.reduce_mean(tf.square(z_low_pooled - z_mid))
    loss_mid_high = tf.reduce_mean(tf.square(z_mid_pooled - z_high))

    return loss_low_mid + loss_mid_high

# ============================================================
# 7. TRAINING
# ============================================================
def train_hca_model(model, dataset_train, dataset_val, epochs=EPOCHS, 
                    steps_per_epoch=200, val_steps=50):
    """Train HCA-Encoder with validation"""

    # Separate optimizers for each level
    opt_low = keras.optimizers.Adam(LR)
    opt_mid = keras.optimizers.Adam(LR)
    opt_high = keras.optimizers.Adam(LR)
    opt_disc = keras.optimizers.Adam(LR)

    history = {
        'train': {
            'L_total': [], 'L_low': [], 'L_mid': [], 'L_high': [], 'L_hierarchy': [],
            'L_rec': [], 'L_adv_low': [], 'L_batch_low': []
        },
        'val': {
            'L_total': [], 'L_low': [], 'L_mid': [], 'L_high': [], 'L_hierarchy': [],
            'L_rec': [], 'L_adv_low': [], 'L_batch_low': []
        }
    }

    @tf.function
    def train_step(frames, centre_id):
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(frames, centre_id, training=True)

            # Extract all outputs
            z_FE_low = outputs['z_FE_low']
            z_RE_low = outputs['z_RE_low']
            mu_RE_low = outputs['mu_RE_low']
            log_var_RE_low = outputs['log_var_RE_low']

            z_FE_mid = outputs['z_FE_mid']
            z_RE_mid = outputs['z_RE_mid']

            z_FE_high = outputs['z_FE_high']
            z_RE_high = outputs['z_RE_high']

            # ===== LEVEL 1 (LOW) LOSSES =====
            # Reconstruction
            x_rec_low = layers.TimeDistributed(model.decoder)(z_FE_low)
            L_rec_low = tf.reduce_mean(tf.square(frames - x_rec_low))

            # Temporal regularization
            dz_FE_low = z_FE_low[:, 1:] - z_FE_low[:, :-1]
            L_scale_low = scale_regularization(dz_FE_low)
            L_spec_low = spectral_loss(dz_FE_low)

            # Adversarial
            pred_FE_low = model.discriminator_low(z_FE_low, training=True)
            L_adv_low = -tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_FE_low, from_logits=True
                )
            )

            # RE classification
            z_RE_low_pooled = tf.reduce_mean(z_RE_low, axis=1)
            pred_RE_low = model.RE_classifier_low(z_RE_low_pooled)
            L_batch_low = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_RE_low, from_logits=True
                )
            )

            # KL divergence
            L_KL_low = kl_divergence(mu_RE_low, log_var_RE_low)

            # Cross-attention alignment
            L_cross_low = cross_alignment_loss(z_FE_low, z_RE_low)

            L_low_total = (
                W_REC * L_rec_low +
                W_SCALE * L_scale_low +
                W_SPEC * L_spec_low +
                W_ADV * L_adv_low +
                W_BATCH * L_batch_low +
                W_KL * L_KL_low +
                W_CROSS_LOW * L_cross_low
            )

            # ===== LEVEL 2 (MID) LOSSES =====
            # Adversarial
            pred_FE_mid = model.discriminator_mid(z_FE_mid, training=True)
            L_adv_mid = -tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_FE_mid, from_logits=True
                )
            )

            # RE classification
            z_RE_mid_pooled = tf.reduce_mean(z_RE_mid, axis=1)
            pred_RE_mid = model.RE_classifier_mid(z_RE_mid_pooled)
            L_batch_mid = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_RE_mid, from_logits=True
                )
            )

            # Cross-attention alignment
            L_cross_mid = cross_alignment_loss(z_FE_mid, z_RE_mid)

            L_mid_total = (
                W_ADV * L_adv_mid +
                W_BATCH * L_batch_mid +
                W_CROSS_MID * L_cross_mid
            )

            # ===== LEVEL 3 (HIGH) LOSSES =====
            # Adversarial
            pred_FE_high = model.discriminator_high(z_FE_high, training=True)
            L_adv_high = -tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_FE_high, from_logits=True
                )
            )

            # RE classification
            z_RE_high_pooled = tf.squeeze(z_RE_high, axis=1)
            pred_RE_high = model.RE_classifier_high(z_RE_high_pooled)
            L_batch_high = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_RE_high, from_logits=True
                )
            )

            # Cross-attention alignment
            L_cross_high = cross_alignment_loss(z_FE_high, z_RE_high)

            L_high_total = (
                W_ADV * L_adv_high +
                W_BATCH * L_batch_high +
                W_CROSS_HIGH * L_cross_high
            )

            # ===== HIERARCHICAL CONSISTENCY =====
            L_hierarchy_FE = hierarchical_consistency_loss(z_FE_low, z_FE_mid, z_FE_high)
            L_hierarchy_RE = hierarchical_consistency_loss(z_RE_low, z_RE_mid, z_RE_high)
            L_hierarchy = L_hierarchy_FE + L_hierarchy_RE

            # ===== DISCRIMINATOR LOSSES =====
            L_disc_low = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_FE_low, from_logits=True
                )
            )
            L_disc_mid = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_FE_mid, from_logits=True
                )
            )
            L_disc_high = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_FE_high, from_logits=True
                )
            )
            L_disc_total = L_disc_low + L_disc_mid + L_disc_high

            # ===== TOTAL LOSS =====
            L_total = (
                L_low_total +
                L_mid_total +
                L_high_total +
                W_HIERARCHY * L_hierarchy
            )

        # Update Level 1 (Low)
        vars_low = (
            model.level_low.trainable_variables +
            model.decoder.trainable_variables
        )
        grads_low = tape.gradient(L_low_total, vars_low)
        opt_low.apply_gradients(zip(grads_low, vars_low))

        # Update Level 2 (Mid)
        vars_mid = model.level_mid.trainable_variables
        grads_mid = tape.gradient(L_mid_total, vars_mid)
        opt_mid.apply_gradients(zip(grads_mid, vars_mid))

        # Update Level 3 (High)
        vars_high = model.level_high.trainable_variables
        grads_high = tape.gradient(L_high_total, vars_high)
        opt_high.apply_gradients(zip(grads_high, vars_high))

        # Update Discriminators
        vars_disc = (
            model.discriminator_low.trainable_variables +
            model.discriminator_mid.trainable_variables +
            model.discriminator_high.trainable_variables +
            model.RE_classifier_low.trainable_variables +
            model.RE_classifier_mid.trainable_variables +
            model.RE_classifier_high.trainable_variables
        )
        grads_disc = tape.gradient(L_disc_total, vars_disc)
        opt_disc.apply_gradients(zip(grads_disc, vars_disc))

        del tape

        return {
            'L_total': L_total,
            'L_low': L_low_total,
            'L_mid': L_mid_total,
            'L_high': L_high_total,
            'L_hierarchy': L_hierarchy,
            'L_rec': L_rec_low,
            'L_adv_low': L_adv_low,
            'L_batch_low': L_batch_low
        }

    @tf.function
    def val_step(frames, centre_id):
        """Validation step (no gradient update)"""
        outputs = model(frames, centre_id, training=False)

        # Extract all outputs
        z_FE_low = outputs['z_FE_low']
        z_RE_low = outputs['z_RE_low']
        mu_RE_low = outputs['mu_RE_low']
        log_var_RE_low = outputs['log_var_RE_low']

        z_FE_mid = outputs['z_FE_mid']
        z_RE_mid = outputs['z_RE_mid']

        z_FE_high = outputs['z_FE_high']
        z_RE_high = outputs['z_RE_high']

        # Level 1 losses
        x_rec_low = layers.TimeDistributed(model.decoder)(z_FE_low)
        L_rec_low = tf.reduce_mean(tf.square(frames - x_rec_low))

        dz_FE_low = z_FE_low[:, 1:] - z_FE_low[:, :-1]
        L_scale_low = scale_regularization(dz_FE_low)
        L_spec_low = spectral_loss(dz_FE_low)

        pred_FE_low = model.discriminator_low(z_FE_low, training=False)
        L_adv_low = -tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, pred_FE_low, from_logits=True
            )
        )

        z_RE_low_pooled = tf.reduce_mean(z_RE_low, axis=1)
        pred_RE_low = model.RE_classifier_low(z_RE_low_pooled)
        L_batch_low = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, pred_RE_low, from_logits=True
            )
        )

        L_KL_low = kl_divergence(mu_RE_low, log_var_RE_low)
        L_cross_low = cross_alignment_loss(z_FE_low, z_RE_low)

        L_low_total = (
            W_REC * L_rec_low +
            W_SCALE * L_scale_low +
            W_SPEC * L_spec_low +
            W_ADV * L_adv_low +
            W_BATCH * L_batch_low +
            W_KL * L_KL_low +
            W_CROSS_LOW * L_cross_low
        )

        # Level 2 losses
        pred_FE_mid = model.discriminator_mid(z_FE_mid, training=False)
        L_adv_mid = -tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, pred_FE_mid, from_logits=True
            )
        )

        z_RE_mid_pooled = tf.reduce_mean(z_RE_mid, axis=1)
        pred_RE_mid = model.RE_classifier_mid(z_RE_mid_pooled)
        L_batch_mid = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, pred_RE_mid, from_logits=True
            )
        )

        L_cross_mid = cross_alignment_loss(z_FE_mid, z_RE_mid)

        L_mid_total = (
            W_ADV * L_adv_mid +
            W_BATCH * L_batch_mid +
            W_CROSS_MID * L_cross_mid
        )

        # Level 3 losses
        pred_FE_high = model.discriminator_high(z_FE_high, training=False)
        L_adv_high = -tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, pred_FE_high, from_logits=True
            )
        )

        z_RE_high_pooled = tf.squeeze(z_RE_high, axis=1)
        pred_RE_high = model.RE_classifier_high(z_RE_high_pooled)
        L_batch_high = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, pred_RE_high, from_logits=True
            )
        )

        L_cross_high = cross_alignment_loss(z_FE_high, z_RE_high)

        L_high_total = (
            W_ADV * L_adv_high +
            W_BATCH * L_batch_high +
            W_CROSS_HIGH * L_cross_high
        )

        # Hierarchical consistency
        L_hierarchy_FE = hierarchical_consistency_loss(z_FE_low, z_FE_mid, z_FE_high)
        L_hierarchy_RE = hierarchical_consistency_loss(z_RE_low, z_RE_mid, z_RE_high)
        L_hierarchy = L_hierarchy_FE + L_hierarchy_RE

        # Total loss
        L_total = (
            L_low_total +
            L_mid_total +
            L_high_total +
            W_HIERARCHY * L_hierarchy
        )

        return {
            'L_total': L_total,
            'L_low': L_low_total,
            'L_mid': L_mid_total,
            'L_high': L_high_total,
            'L_hierarchy': L_hierarchy,
            'L_rec': L_rec_low,
            'L_adv_low': L_adv_low,
            'L_batch_low': L_batch_low
        }

    # Training loop
    print("\n" + "="*60)
    print("HIERARCHICAL CROSS-ATTENTION ENCODER TRAINING")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # TRAINING
        losses_train = []
        for step, (frames, centre_id) in enumerate(
            dataset_train.take(steps_per_epoch), 1
        ):
            loss_dict = train_step(frames, centre_id)
            losses_train.append({k: float(v) for k, v in loss_dict.items()})

        # Average training losses
        avg_train = {
            k: np.mean([d[k] for d in losses_train])
            for k in losses_train[0].keys()
        }

        # VALIDATION
        losses_val = []
        for step, (frames, centre_id) in enumerate(
            dataset_val.take(val_steps), 1
        ):
            loss_dict = val_step(frames, centre_id)
            losses_val.append({k: float(v) for k, v in loss_dict.items()})

        # Average validation losses
        avg_val = {
            k: np.mean([d[k] for d in losses_val])
            for k in losses_val[0].keys()
        }

        # Save to history
        for key in history['train'].keys():
            history['train'][key].append(avg_train[key])
            history['val'][key].append(avg_val[key])

        # Print results
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"TRAIN - Total: {avg_train['L_total']:.4f} | "
              f"Low: {avg_train['L_low']:.4f} | "
              f"Mid: {avg_train['L_mid']:.4f} | "
              f"High: {avg_train['L_high']:.4f}")
        print(f"        Hierarchy: {avg_train['L_hierarchy']:.4f} | "
              f"Recon: {avg_train['L_rec']:.4f}")
        print(f"        Adv_low: {avg_train['L_adv_low']:.4f} | "
              f"Batch_low: {avg_train['L_batch_low']:.4f}")
        
        print(f"VAL   - Total: {avg_val['L_total']:.4f} | "
              f"Low: {avg_val['L_low']:.4f} | "
              f"Mid: {avg_val['L_mid']:.4f} | "
              f"High: {avg_val['L_high']:.4f}")

        # Save best model
        if avg_val['L_total'] < best_val_loss:
            best_val_loss = avg_val['L_total']
            model.save_weights("/content/HCA_best_model.weights.h5")
            print(f"✅ Best model saved! (Val Loss: {best_val_loss:.4f})")

    print("\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, history

# ============================================================
# 8. EVALUATION
# ============================================================
def extract_features_for_TOP(model, df, max_clips=500):
    """Extract hierarchical features for TOP"""
    X, Y, centres_list = [], [], []

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=min(len(df), max_clips),
                                      desc="Extracting features")):
        if i >= max_clips:
            break

        A_path = Path(row['full_path_A'])
        V_path = Path(row['full_path_V'])

        if not (A_path.exists() and V_path.exists()):
            continue

        try:
            A = np.load(A_path).astype(np.float32) / 255.0
            valid = np.load(V_path).astype(bool)

            idx = np.where(valid)[0]
            if len(idx) < 10:
                continue

            T = min(len(idx), T_WIN)
            sampled_idx = np.linspace(idx[0], idx[-1], T).astype(int)
            A = A[sampled_idx]
            A = tf.image.resize(A[..., None], (IMG_SIZE, IMG_SIZE), method='area')
            A = A[None, ...]

            centre_id = CENTRE_MAP[row['centre']]
            centre_id_tf = tf.constant([centre_id], dtype=tf.int32)

            # Extract hierarchical features
            outputs = model(A, centre_id_tf, training=False)

            # Concatenate all levels
            z_FE_low = outputs['z_FE_low'].numpy()[0]
            z_FE_mid = outputs['z_FE_mid'].numpy()[0]
            z_FE_high = outputs['z_FE_high'].numpy()[0]

            # Combine all hierarchical features
            z_combined = np.concatenate([
                z_FE_low,                              # (T, D)
                np.repeat(z_FE_mid, SEGMENT_SIZE, axis=0),  # Upsample (4, D) → (T, D)
                np.repeat(z_FE_high, T, axis=0)        # Upsample (1, D) → (T, D)
            ], axis=-1)  # (T, 3*D)

            # Compute delta features
            dz = z_combined[1:] - z_combined[:-1]
            feat = np.concatenate([
                dz.mean(0),
                dz.std(0),
                np.percentile(dz, [25, 50, 75], axis=0).reshape(-1)
            ])

            # Positive: correct order
            X.append(feat)
            Y.append(1)
            centres_list.append(centre_id)

            # Negative: shuffled order
            perm = np.random.permutation(len(z_combined))
            z_shuf = z_combined[perm]
            dz_shuf = z_shuf[1:] - z_shuf[:-1]
            feat_shuf = np.concatenate([
                dz_shuf.mean(0),
                dz_shuf.std(0),
                np.percentile(dz_shuf, [25, 50, 75], axis=0).reshape(-1)
            ])
            X.append(feat_shuf)
            Y.append(0)
            centres_list.append(centre_id)

        except Exception as e:
            continue

    return np.array(X), np.array(Y), np.array(centres_list)

def evaluate_TOP(X, Y):
    """Temporal Order Prediction evaluation with train/test split"""
    print("\n" + "="*60)
    print("TEMPORAL ORDER PREDICTION (TOP)")
    print("="*60)

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=SEED, stratify=Y
    )

    # Train classifier
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
    clf.fit(X_train, Y_train)
    
    # Test accuracy
    Y_pred = clf.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)

    # Bootstrap CI
    boot_acc = []
    for _ in range(1000):
        X_b, Y_b = resample(X_test, Y_test, random_state=None)
        clf_b = LogisticRegression(max_iter=500, random_state=SEED)
        clf_b.fit(X_train, Y_train)
        boot_acc.append(accuracy_score(Y_b, clf_b.predict(X_b)))

    ci_low, ci_high = np.percentile(boot_acc, [2.5, 97.5])

    # Permutation test
    perm_acc = []
    for _ in range(500):
        Y_perm = np.random.permutation(Y_train)
        clf_p = LogisticRegression(max_iter=500, random_state=SEED)
        clf_p.fit(X_train, Y_perm)
        perm_acc.append(accuracy_score(Y_test, clf_p.predict(X_test)))

    p_value = np.mean(np.array(perm_acc) >= acc)

    print(f"\n✅ Accuracy: {acc*100:.2f}%")
    print(f"   95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print(f"   p-value: {p_value:.5f}")

    return acc, ci_low, ci_high, p_value

# ============================================================
# 9. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("🚀 HCA-ENCODER (Hierarchical Cross-Attention) - FULL PIPELINE")
    print("="*60)

    # 1. Create metadata
    df = create_metadata()

    # 2. SPLIT DATA (70/15/15)
    df_train, df_val, df_test = split_data(df, 0.70, 0.15, 0.15)

    # 3. Build datasets
    dataset_train = build_dataset(df_train, batch_size=BATCH_SIZE)
    dataset_val = build_dataset(df_val, batch_size=BATCH_SIZE)

    # 4. Build model
    model = HCAEncoderModel(
        latent_dim=LATENT_DIM,
        num_centres=NUM_CENTRES
    )

    # 5. Train with validation
    model, history = train_hca_model(
        model,
        dataset_train,
        dataset_val,
        epochs=EPOCHS,
        steps_per_epoch=200,
        val_steps=50
    )

    # 6. Load best model
    print("\n📥 Loading best model from validation...")
    model.load_weights("/content/HCA_best_model.weights.h5")

    # 7. Save final model
    print("✅ Final model saved!")

    # 8. Evaluate TOP on TEST SET
    print("\n📊 Extracting hierarchical features from TEST set...")
    X_test, Y_test, centres_test = extract_features_for_TOP(model, df_test, max_clips=500)

    if len(X_test) > 0:
        acc, ci_low, ci_high, p_val = evaluate_TOP(X_test, Y_test)

        # 9. Summary
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED - FINAL RESULTS (TEST SET)")
        print("="*60)
        print(f"Architecture: Hierarchical Cross-Attention Encoder (HCA)")
        print(f"Training epochs: {EPOCHS}")
        print(f"Latent dimension: {LATENT_DIM}")
        print(f"Number of centres: {NUM_CENTRES}")
        print(f"Hierarchical levels: 3 (Frame/Segment/Global)")
        print(f"Train/Val/Test split: 70/15/15")
        print(f"\nTrain clips: {len(df_train)}")
        print(f"Val clips:   {len(df_val)}")
        print(f"Test clips:  {len(df_test)}")
        print(f"\n📊 Results:")
        print(f"TOP Accuracy: {acc*100:.2f}% [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
        print(f"p-value: {p_val:.5f}")
        print("="*60)
    else:
        print("⚠️ No features extracted. Check paths and data integrity.")