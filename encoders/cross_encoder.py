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

# A ve B klasörlerini tanımlıyoruz
DIRS_TO_SCAN = [ROOT / "A_fixed", ROOT / "B_variable"]

# Hyperparameters
IMG_SIZE = 128
T_WIN = 64
LATENT_DIM = 128
BATCH_SIZE = 4
EPOCHS = 10
LR = 2e-4

# Loss weights
W_REC = 0.1
W_SCALE = 1.0
W_SPEC = 0.5
W_ADV = 1.0     # adversarial (FE)
W_BATCH = 1.0   # batch classification (RE)
W_KL = 0.01     # KL divergence (RE)
W_CROSS = 0.5   # NEW: cross-attention alignment loss

# ============================================================
# 1. DATA LOADING & METADATA
# ============================================================
def create_metadata():
    """Create metadata scanning both A_fixed and B_variable without strict validation checks"""
    rows = []

    print(f"Scanning directories: {[d.name for d in DIRS_TO_SCAN]}...")

    for root_path in DIRS_TO_SCAN:
        if not root_path.exists():
            print(f"⚠️ Warning: {root_path} does not exist. Skipping.")
            continue

        dataset_type = root_path.name # 'A_fixed' or 'B_variable'

        # Klasör yapısını umursamadan TÜM .npy dosyalarını bul (Recursive)
        all_npy = list(root_path.rglob("*.npy"))
        # _valid.npy olmayanları al (Sadece videolar)
        video_files = [f for f in all_npy if "_valid" not in f.name]

        print(f"   -> Found {len(video_files)} video files in {dataset_type}")

        for clip_file in video_files:
            # Merkez bilgisini dosya yolundan çıkar
            path_str = str(clip_file)
            centre = None
            if "Merkez 2" in path_str: centre = "Merkez 2"
            elif "Merkez 6" in path_str: centre = "Merkez 6"
            elif "Merkez 8" in path_str: centre = "Merkez 8"

            if centre is None:
                continue

            subject = clip_file.parent.name
            clip = clip_file.stem

            # Validasyon dosyasının yolu
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

# Map centre names to integers
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
    """Subject-balanced data generator that handles missing valid files"""

    # Group by (centre, subject)
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
            # Rastgele bir hasta seç
            centre, subject = random.choice(subjects)
            # O hastaya ait rastgele bir klip seç
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

                # Rastgele bir pencere seç
                s = random.randint(idx[0], idx[-1] - T_WIN + 1)
                seq = A[s:s+T_WIN]

                # Resize
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
    """Shared CNN backbone for both FE and RE"""
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
    """Bayesian encoder with reparameterization trick"""
    def __init__(self, latent_dim):
        super().__init__(name="bayesian_encoder")
        self.backbone = build_cnn_backbone(latent_dim*2)
        self.latent_dim = latent_dim

    def call(self, x):
        h = self.backbone(x)
        mu = h[..., :self.latent_dim]
        log_var = h[..., self.latent_dim:]

        # Reparameterization
        std = tf.exp(0.5 * log_var)
        eps = tf.random.normal(tf.shape(std))
        z = mu + eps * std

        return z, mu, log_var

class CentreDiscriminator(keras.Model):
    """Adversarial discriminator to predict centre from FE latent"""
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

class TemporalAttentionEncoder(layers.Layer):
    """Temporal self-attention"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout
        )
        self.norm = layers.LayerNormalization()

    def call(self, z, training=False):
        attn_out = self.mha(z, z, training=training)
        z_out = self.norm(z + attn_out)
        return z_out

# ============================================================
# NEW: CROSS-ATTENTION MODULE
# ============================================================
class CrossAttentionModule(layers.Layer):
    """
    Cross-attention between FE and RE streams

    FE attends to RE: "What center-specific info should I be aware of?"
    RE attends to FE: "What center-invariant patterns exist?"
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim

        # FE queries RE
        self.FE_to_RE = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout,
            name="FE_attends_to_RE"
        )

        # RE queries FE
        self.RE_to_FE = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout,
            name="RE_attends_to_FE"
        )

        self.norm_FE = layers.LayerNormalization()
        self.norm_RE = layers.LayerNormalization()

        # Gating mechanism to control cross-attention influence
        self.gate_FE = layers.Dense(dim, activation='sigmoid', name="gate_FE")
        self.gate_RE = layers.Dense(dim, activation='sigmoid', name="gate_RE")

    def call(self, z_FE, z_RE, training=False):
        """
        Args:
            z_FE: (B, T, D) - Center-invariant features
            z_RE: (B, T, D) - Center-specific features

        Returns:
            z_FE_updated: FE enriched with RE info
            z_RE_updated: RE enriched with FE info
        """
        # FE attends to RE
        FE_cross = self.FE_to_RE(
            query=z_FE,
            key=z_RE,
            value=z_RE,
            training=training
        )

        # Gate the cross-attention (control how much RE info affects FE)
        gate_fe = self.gate_FE(z_FE)
        z_FE_updated = self.norm_FE(z_FE + gate_fe * FE_cross)

        # RE attends to FE
        RE_cross = self.RE_to_FE(
            query=z_RE,
            key=z_FE,
            value=z_FE,
            training=training
        )

        # Gate the cross-attention (control how much FE info affects RE)
        gate_re = self.gate_RE(z_RE)
        z_RE_updated = self.norm_RE(z_RE + gate_re * RE_cross)

        return z_FE_updated, z_RE_updated

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
# 4. CROSS ENCODER MODEL
# ============================================================
class CrossEncoderModel(keras.Model):
    """
    Cross Encoder system with bidirectional attention:
    - FE: Centre-invariant (adversarial)
    - RE: Centre-specific (Bayesian)
    - Cross-Attention: FE ↔ RE communication
    """
    def __init__(self, latent_dim=LATENT_DIM, num_centres=NUM_CENTRES):
        super().__init__()

        # Fixed Effects components
        self.FE_encoder = build_cnn_backbone(latent_dim, "FE_encoder")
        self.FE_attention = TemporalAttentionEncoder(latent_dim)
        self.FE_discriminator = CentreDiscriminator(num_centres)

        # Random Effects components
        self.RE_encoder = BayesianEncoder(latent_dim)
        self.RE_batch_classifier = layers.Dense(num_centres, name="RE_classifier")
        self.RE_norm = BatchConditionalNorm(num_centres, latent_dim)

        # NEW: Cross-Attention Module
        self.cross_attention = CrossAttentionModule(latent_dim)

        # Shared decoder
        self.decoder = build_decoder(latent_dim)

        self.latent_dim = latent_dim
        self.num_centres = num_centres

    def call(self, frames, centre_id, training=False):
        # frames: (B,T,H,W,1)

        # ===== FIXED EFFECTS PATH =====
        z_FE = layers.TimeDistributed(self.FE_encoder)(frames)
        z_FE = self.FE_attention(z_FE, training=training)

        # ===== RANDOM EFFECTS PATH =====
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

        # Conditional normalization
        z_RE = self.RE_norm(z_RE, centre_id)

        # ===== NEW: CROSS-ATTENTION =====
        # FE and RE communicate with each other
        z_FE_cross, z_RE_cross = self.cross_attention(z_FE, z_RE, training=training)

        return {
            'z_FE': z_FE_cross,       # Updated with cross-attention
            'z_RE': z_RE_cross,       # Updated with cross-attention
            'z_FE_orig': z_FE,        # Original FE (before cross-attention)
            'z_RE_orig': z_RE,        # Original RE (before cross-attention)
            'mu_RE': mu_RE,
            'log_var_RE': log_var_RE
        }

# ============================================================
# 5. LOSS FUNCTIONS
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
    """KL divergence for Bayesian encoder"""
    return -0.5 * tf.reduce_mean(1 + log_var - mu**2 - tf.exp(log_var))

def cross_alignment_loss(z_FE, z_RE):
    """
    NEW: Alignment loss for cross-attention

    Encourages FE and RE to have complementary information:
    - FE should not duplicate RE
    - RE should not duplicate FE

    Uses orthogonality constraint
    """
    # Normalize features
    z_FE_norm = tf.nn.l2_normalize(z_FE, axis=-1)
    z_RE_norm = tf.nn.l2_normalize(z_RE, axis=-1)

    # Compute cosine similarity (should be low = orthogonal)
    similarity = tf.reduce_sum(z_FE_norm * z_RE_norm, axis=-1)

    # Penalize high similarity (we want them to be different/complementary)
    return tf.reduce_mean(tf.square(similarity))

# ============================================================
# 6. TRAINING
# ============================================================
def train_cross_model(model, dataset_train, dataset_val, epochs=EPOCHS, 
                     steps_per_epoch=200, val_steps=50):
    """Train cross encoder system with validation"""

    # Optimizers
    opt_FE = keras.optimizers.Adam(LR)
    opt_RE = keras.optimizers.Adam(LR)
    opt_disc = keras.optimizers.Adam(LR)
    opt_cross = keras.optimizers.Adam(LR)  # NEW: optimizer for cross-attention

    history = {
        'train': {
            'L_total': [], 'L_FE': [], 'L_RE': [], 'L_cross': [],
            'L_rec_FE': [], 'L_scale': [], 'L_spec': [], 'L_adv': [],
            'L_rec_RE': [], 'L_batch': [], 'L_KL': [], 'L_disc': []
        },
        'val': {
            'L_total': [], 'L_FE': [], 'L_RE': [], 'L_cross': [],
            'L_rec_FE': [], 'L_scale': [], 'L_spec': [], 'L_adv': [],
            'L_rec_RE': [], 'L_batch': [], 'L_KL': [], 'L_disc': []
        }
    }

    @tf.function
    def train_step(frames, centre_id):
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(frames, centre_id, training=True)

            z_FE = outputs['z_FE']
            z_RE = outputs['z_RE']
            z_FE_orig = outputs['z_FE_orig']
            z_RE_orig = outputs['z_RE_orig']
            mu_RE = outputs['mu_RE']
            log_var_RE = outputs['log_var_RE']

            # ===== FIXED EFFECTS LOSSES =====
            # 1. Reconstruction
            x_rec_FE = layers.TimeDistributed(model.decoder)(z_FE)
            L_rec_FE = tf.reduce_mean(tf.square(frames - x_rec_FE))

            # 2. Temporal regularization
            dz_FE = z_FE[:, 1:] - z_FE[:, :-1]
            L_scale_FE = scale_regularization(dz_FE)
            L_spec_FE = spectral_loss(dz_FE)

            # 3. Adversarial loss (NEGATIVE cross-entropy)
            centre_pred_FE = model.FE_discriminator(z_FE, training=True)
            L_adv = -tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, centre_pred_FE, from_logits=True
                )
            )

            L_FE_total = (
                W_REC * L_rec_FE +
                W_SCALE * L_scale_FE +
                W_SPEC * L_spec_FE +
                W_ADV * L_adv
            )

            # ===== RANDOM EFFECTS LOSSES =====
            # 1. Reconstruction
            x_rec_RE = layers.TimeDistributed(model.decoder)(z_RE)
            L_rec_RE = tf.reduce_mean(tf.square(frames - x_rec_RE))

            # 2. Batch classification (POSITIVE)
            z_RE_pooled = tf.reduce_mean(z_RE, axis=1)
            centre_pred_RE = model.RE_batch_classifier(z_RE_pooled)
            L_batch = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, centre_pred_RE, from_logits=True
                )
            )

            # 3. KL divergence
            L_KL = kl_divergence(mu_RE, log_var_RE)

            L_RE_total = (
                W_REC * L_rec_RE +
                W_BATCH * L_batch +
                W_KL * L_KL
            )

            # ===== NEW: CROSS-ATTENTION LOSS =====
            L_cross = cross_alignment_loss(z_FE, z_RE)

            # ===== DISCRIMINATOR LOSS =====
            L_disc = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, centre_pred_FE, from_logits=True
                )
            )

            L_total = L_FE_total + L_RE_total + W_CROSS * L_cross

        # Update FE encoder + attention
        FE_vars = (
            model.FE_encoder.trainable_variables +
            model.FE_attention.trainable_variables +
            model.decoder.trainable_variables
        )
        grads_FE = tape.gradient(L_FE_total, FE_vars)
        opt_FE.apply_gradients(zip(grads_FE, FE_vars))

        # Update RE encoder + classifier + norm
        RE_vars = (
            model.RE_encoder.trainable_variables +
            model.RE_batch_classifier.trainable_variables +
            model.RE_norm.trainable_variables
        )
        grads_RE = tape.gradient(L_RE_total, RE_vars)
        opt_RE.apply_gradients(zip(grads_RE, RE_vars))

        # Update cross-attention module
        cross_vars = model.cross_attention.trainable_variables
        grads_cross = tape.gradient(L_total, cross_vars)
        opt_cross.apply_gradients(zip(grads_cross, cross_vars))

        # Update discriminator
        disc_vars = model.FE_discriminator.trainable_variables
        grads_disc = tape.gradient(L_disc, disc_vars)
        opt_disc.apply_gradients(zip(grads_disc, disc_vars))

        del tape

        return {
            'L_total': L_total,
            'L_FE': L_FE_total,
            'L_RE': L_RE_total,
            'L_cross': L_cross,
            'L_rec_FE': L_rec_FE,
            'L_scale': L_scale_FE,
            'L_spec': L_spec_FE,
            'L_adv': L_adv,
            'L_rec_RE': L_rec_RE,
            'L_batch': L_batch,
            'L_KL': L_KL,
            'L_disc': L_disc
        }

    @tf.function
    def val_step(frames, centre_id):
        """Validation step (no gradient update)"""
        outputs = model(frames, centre_id, training=False)

        z_FE = outputs['z_FE']
        z_RE = outputs['z_RE']
        mu_RE = outputs['mu_RE']
        log_var_RE = outputs['log_var_RE']

        # FE Losses
        x_rec_FE = layers.TimeDistributed(model.decoder)(z_FE)
        L_rec_FE = tf.reduce_mean(tf.square(frames - x_rec_FE))

        dz_FE = z_FE[:, 1:] - z_FE[:, :-1]
        L_scale_FE = scale_regularization(dz_FE)
        L_spec_FE = spectral_loss(dz_FE)

        centre_pred_FE = model.FE_discriminator(z_FE, training=False)
        L_adv = -tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, centre_pred_FE, from_logits=True
            )
        )

        L_FE_total = (
            W_REC * L_rec_FE +
            W_SCALE * L_scale_FE +
            W_SPEC * L_spec_FE +
            W_ADV * L_adv
        )

        # RE Losses
        x_rec_RE = layers.TimeDistributed(model.decoder)(z_RE)
        L_rec_RE = tf.reduce_mean(tf.square(frames - x_rec_RE))

        z_RE_pooled = tf.reduce_mean(z_RE, axis=1)
        centre_pred_RE = model.RE_batch_classifier(z_RE_pooled)
        L_batch = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, centre_pred_RE, from_logits=True
            )
        )

        L_KL = kl_divergence(mu_RE, log_var_RE)

        L_RE_total = (
            W_REC * L_rec_RE +
            W_BATCH * L_batch +
            W_KL * L_KL
        )

        # Cross-attention loss
        L_cross = cross_alignment_loss(z_FE, z_RE)

        # Discriminator loss
        L_disc = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, centre_pred_FE, from_logits=True
            )
        )

        L_total = L_FE_total + L_RE_total + W_CROSS * L_cross

        return {
            'L_total': L_total,
            'L_FE': L_FE_total,
            'L_RE': L_RE_total,
            'L_cross': L_cross,
            'L_rec_FE': L_rec_FE,
            'L_scale': L_scale_FE,
            'L_spec': L_spec_FE,
            'L_adv': L_adv,
            'L_rec_RE': L_rec_RE,
            'L_batch': L_batch,
            'L_KL': L_KL,
            'L_disc': L_disc
        }

    # Training loop
    print("\n" + "="*60)
    print("CROSS ENCODER TEMPORAL TRAINING")
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
              f"FE: {avg_train['L_FE']:.4f} | "
              f"RE: {avg_train['L_RE']:.4f} | "
              f"Cross: {avg_train['L_cross']:.4f}")
        print(f"        L_rec_FE: {avg_train['L_rec_FE']:.4f} | "
              f"L_scale: {avg_train['L_scale']:.4f} | "
              f"L_spec: {avg_train['L_spec']:.4f}")
        print(f"        L_adv: {avg_train['L_adv']:.4f} | "
              f"L_disc: {avg_train['L_disc']:.4f}")
        print(f"        L_batch: {avg_train['L_batch']:.4f} | "
              f"L_KL: {avg_train['L_KL']:.4f}")
        
        print(f"VAL   - Total: {avg_val['L_total']:.4f} | "
              f"FE: {avg_val['L_FE']:.4f} | "
              f"RE: {avg_val['L_RE']:.4f} | "
              f"Cross: {avg_val['L_cross']:.4f}")

        # Save best model
        if avg_val['L_total'] < best_val_loss:
            best_val_loss = avg_val['L_total']
            model.save_weights("/content/CROSS_best_model.weights.h5")
            print(f"✅ Best model saved! (Val Loss: {best_val_loss:.4f})")

    print("\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, history

# ============================================================
# 7. EVALUATION: TEMPORAL ORDER PREDICTION (TOP)
# ============================================================
def extract_features_for_TOP(model, df, max_clips=500):
    """Extract features for Temporal Order Prediction using full paths"""
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

            # Sample frames
            T = min(len(idx), T_WIN)
            sampled_idx = np.linspace(idx[0], idx[-1], T).astype(int)
            A = A[sampled_idx]
            A = tf.image.resize(A[..., None], (IMG_SIZE, IMG_SIZE), method='area')
            A = A[None, ...]

            centre_id = CENTRE_MAP[row['centre']]
            centre_id_tf = tf.constant([centre_id], dtype=tf.int32)

            # Extract features
            outputs = model(A, centre_id_tf, training=False)
            z_FE = outputs['z_FE'].numpy()[0]
            z_RE = outputs['z_RE'].numpy()[0]

            # Combine FE + RE
            z_combined = np.concatenate([z_FE, z_RE], axis=-1)

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
# 8. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("🚀 CROSS ENCODER TEMPORAL SYSTEM - FULL PIPELINE")
    print("="*60)

    # 1. Create metadata
    df = create_metadata()

    # 2. SPLIT DATA (70/15/15)
    df_train, df_val, df_test = split_data(df, 0.70, 0.15, 0.15)

    # 3. Build datasets
    dataset_train = build_dataset(df_train, batch_size=BATCH_SIZE)
    dataset_val = build_dataset(df_val, batch_size=BATCH_SIZE)

    # 4. Build model
    model = CrossEncoderModel(
        latent_dim=LATENT_DIM,
        num_centres=NUM_CENTRES
    )

    # 5. Train with validation
    model, history = train_cross_model(
        model,
        dataset_train,
        dataset_val,
        epochs=EPOCHS,
        steps_per_epoch=200,
        val_steps=50
    )

    # 6. Load best model
    print("\n📥 Loading best model from validation...")
    model.load_weights("/content/CROSS_best_model.weights.h5")

    # 7. Save final models
    model.FE_encoder.save("/content/CROSS_FE_encoder.keras")
    model.RE_encoder.save("/content/CROSS_RE_encoder.keras")
    print("✅ Final models saved!")

    # 8. Evaluate TOP on TEST SET
    print("\n📊 Extracting features from TEST set...")
    X_test, Y_test, centres_test = extract_features_for_TOP(model, df_test, max_clips=500)

    if len(X_test) > 0:
        acc, ci_low, ci_high, p_val = evaluate_TOP(X_test, Y_test)

        # 9. Summary
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED - FINAL RESULTS (TEST SET)")
        print("="*60)
        print(f"Training epochs: {EPOCHS}")
        print(f"Latent dimension: {LATENT_DIM}")
        print(f"Number of centres: {NUM_CENTRES}")
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