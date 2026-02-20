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

# Loss weights
W_REC = 0.1
W_SCALE = 1.0
W_SPEC = 0.5
W_ADV = 1.0
W_BATCH = 1.0
W_KL = 0.01

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

    def call(self, x, training=False):
        h = self.backbone(x, training=training)
        mu = h[..., :self.latent_dim]
        log_var = h[..., self.latent_dim:]

        std = tf.exp(0.5 * log_var)
        eps = tf.random.normal(tf.shape(std))
        z = mu + eps * std

        return z, mu, log_var

class CentreDiscriminator(keras.Model):
    """Adversarial discriminator"""
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
# 4. DUAL TEMPORAL MODEL
# ============================================================
class DualTemporalModel(keras.Model):
    """Dual encoder system"""
    def __init__(self, latent_dim=LATENT_DIM, num_centres=NUM_CENTRES):
        super().__init__()

        self.FE_encoder = build_cnn_backbone(latent_dim, "FE_encoder")
        self.FE_attention = TemporalAttentionEncoder(latent_dim)
        self.FE_discriminator = CentreDiscriminator(num_centres)

        self.RE_encoder = BayesianEncoder(latent_dim)
        self.RE_batch_classifier = layers.Dense(num_centres, name="RE_classifier")
        self.RE_norm = BatchConditionalNorm(num_centres, latent_dim)

        self.decoder = build_decoder(latent_dim)

        self.latent_dim = latent_dim
        self.num_centres = num_centres

    def call(self, frames, centre_id, training=False):
        # frames: (B,T,H,W,1)

        # Fixed Effects
        z_FE = layers.TimeDistributed(self.FE_encoder)(frames, training=training)
        z_FE = self.FE_attention(z_FE, training=training)

        # Random Effects (use tf.unstack)
        frames_list = tf.unstack(frames, axis=1)  # List of (B,H,W,1)

        z_RE_list, mu_list, lv_list = [], [], []
        for frame_t in frames_list:
            z_t, mu_t, lv_t = self.RE_encoder(frame_t, training=training)
            z_RE_list.append(z_t)
            mu_list.append(mu_t)
            lv_list.append(lv_t)

        z_RE = tf.stack(z_RE_list, axis=1)
        mu_RE = tf.stack(mu_list, axis=1)
        log_var_RE = tf.stack(lv_list, axis=1)

        z_RE = self.RE_norm(z_RE, centre_id)

        return {
            'z_FE': z_FE,
            'z_RE': z_RE,
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
    mag = tf.norm(dz, axis=-1)
    med = tf_percentile(mag, 50.0)
    mad = tf_percentile(tf.abs(mag - med), 50.0) + 1e-6
    return tf.reduce_mean(tf.abs(mag - med) / mad)

def spectral_loss(dz):
    mag = tf.norm(dz, axis=-1)
    mag = mag - tf.reduce_mean(mag, axis=1, keepdims=True)

    fft = tf.signal.rfft(mag)
    power = tf.abs(fft) ** 2

    cutoff = tf.cast(0.2 * tf.cast(tf.shape(power)[1], tf.float32), tf.int32)
    return tf.reduce_mean(power[:, cutoff:])

def kl_divergence(mu, log_var):
    return -0.5 * tf.reduce_mean(1 + log_var - mu**2 - tf.exp(log_var))

# ============================================================
# 6. TRAINING
# ============================================================
def train_dual_model(model, dataset_train, dataset_val, epochs=EPOCHS, 
                    steps_per_epoch=200, val_steps=50):
    """Train with validation and history tracking"""

    opt_FE = keras.optimizers.Adam(LR)
    opt_RE = keras.optimizers.Adam(LR)
    opt_disc = keras.optimizers.Adam(LR)

    history = {
        'train': {
            'L_total': [], 'L_FE': [], 'L_RE': [],
            'L_rec_FE': [], 'L_scale': [], 'L_spec': [], 'L_adv': [],
            'L_rec_RE': [], 'L_batch': [], 'L_KL': [], 'L_disc': []
        },
        'val': {
            'L_total': [], 'L_FE': [], 'L_RE': [],
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
            mu_RE = outputs['mu_RE']
            log_var_RE = outputs['log_var_RE']

            # FE Losses
            x_rec_FE = layers.TimeDistributed(model.decoder)(z_FE, training=True)
            L_rec_FE = tf.reduce_mean(tf.square(frames - x_rec_FE))

            dz_FE = z_FE[:, 1:] - z_FE[:, :-1]
            L_scale_FE = scale_regularization(dz_FE)
            L_spec_FE = spectral_loss(dz_FE)

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

            # RE Losses
            x_rec_RE = layers.TimeDistributed(model.decoder)(z_RE, training=True)
            L_rec_RE = tf.reduce_mean(tf.square(frames - x_rec_RE))

            z_RE_pooled = tf.reduce_mean(z_RE, axis=1)
            centre_pred_RE = model.RE_batch_classifier(z_RE_pooled, training=True)
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

            # Discriminator Loss
            L_disc = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, centre_pred_FE, from_logits=True
                )
            )

            L_total = L_FE_total + L_RE_total

        # Updates
        FE_vars = (
            model.FE_encoder.trainable_variables +
            model.FE_attention.trainable_variables +
            model.decoder.trainable_variables
        )
        grads_FE = tape.gradient(L_FE_total, FE_vars)
        opt_FE.apply_gradients(zip(grads_FE, FE_vars))

        RE_vars = (
            model.RE_encoder.trainable_variables +
            model.RE_batch_classifier.trainable_variables +
            model.RE_norm.trainable_variables
        )
        grads_RE = tape.gradient(L_RE_total, RE_vars)
        opt_RE.apply_gradients(zip(grads_RE, RE_vars))

        disc_vars = model.FE_discriminator.trainable_variables
        grads_disc = tape.gradient(L_disc, disc_vars)
        opt_disc.apply_gradients(zip(grads_disc, disc_vars))

        del tape

        return {
            'L_total': L_total, 'L_FE': L_FE_total, 'L_RE': L_RE_total,
            'L_rec_FE': L_rec_FE, 'L_scale': L_scale_FE, 'L_spec': L_spec_FE,
            'L_adv': L_adv, 'L_rec_RE': L_rec_RE, 'L_batch': L_batch,
            'L_KL': L_KL, 'L_disc': L_disc
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
        x_rec_FE = layers.TimeDistributed(model.decoder)(z_FE, training=False)
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
        x_rec_RE = layers.TimeDistributed(model.decoder)(z_RE, training=False)
        L_rec_RE = tf.reduce_mean(tf.square(frames - x_rec_RE))

        z_RE_pooled = tf.reduce_mean(z_RE, axis=1)
        centre_pred_RE = model.RE_batch_classifier(z_RE_pooled, training=False)
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

        # Discriminator Loss
        L_disc = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, centre_pred_FE, from_logits=True
            )
        )

        L_total = L_FE_total + L_RE_total

        return {
            'L_total': L_total, 'L_FE': L_FE_total, 'L_RE': L_RE_total,
            'L_rec_FE': L_rec_FE, 'L_scale': L_scale_FE, 'L_spec': L_spec_FE,
            'L_adv': L_adv, 'L_rec_RE': L_rec_RE, 'L_batch': L_batch,
            'L_KL': L_KL, 'L_disc': L_disc
        }

    print("\n" + "="*60)
    print("DUAL TEMPORAL ENCODER TRAINING")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # TRAINING
        losses_train = []
        for step, (frames, centre_id) in enumerate(dataset_train.take(steps_per_epoch), 1):
            loss_dict = train_step(frames, centre_id)
            losses_train.append({k: float(v) for k, v in loss_dict.items()})

        avg_train = {
            k: np.mean([d[k] for d in losses_train])
            for k in losses_train[0].keys()
        }

        # VALIDATION
        losses_val = []
        for step, (frames, centre_id) in enumerate(dataset_val.take(val_steps), 1):
            loss_dict = val_step(frames, centre_id)
            losses_val.append({k: float(v) for k, v in loss_dict.items()})

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
        print(f"TRAIN - Total: {avg_train['L_total']:.4f} | FE: {avg_train['L_FE']:.4f} | RE: {avg_train['L_RE']:.4f}")
        print(f"        L_rec_FE: {avg_train['L_rec_FE']:.4f} | L_scale: {avg_train['L_scale']:.4f} | L_spec: {avg_train['L_spec']:.4f}")
        print(f"        L_adv: {avg_train['L_adv']:.4f} | L_disc: {avg_train['L_disc']:.4f}")
        print(f"        L_batch: {avg_train['L_batch']:.4f} | L_KL: {avg_train['L_KL']:.4f}")
        
        print(f"VAL   - Total: {avg_val['L_total']:.4f} | FE: {avg_val['L_FE']:.4f} | RE: {avg_val['L_RE']:.4f}")
        print(f"        L_rec_FE: {avg_val['L_rec_FE']:.4f} | L_scale: {avg_val['L_scale']:.4f} | L_spec: {avg_val['L_spec']:.4f}")

        # Save best model
        if avg_val['L_total'] < best_val_loss:
            best_val_loss = avg_val['L_total']
            model.save_weights("/content/DUAL_best_model.weights.h5")
            print(f"✅ Best model saved! (Val Loss: {best_val_loss:.4f})")

    print("\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, history

# ============================================================
# 7. EVALUATION
# ============================================================
def extract_features_for_TOP(model, df, max_clips=500):
    """Extract TOP features"""
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

            outputs = model(A, centre_id_tf, training=False)
            z_FE = outputs['z_FE'].numpy()[0]
            z_RE = outputs['z_RE'].numpy()[0]

            z_combined = np.concatenate([z_FE, z_RE], axis=-1)

            dz = z_combined[1:] - z_combined[:-1]
            feat = np.concatenate([
                dz.mean(0), dz.std(0),
                np.percentile(dz, [25, 50, 75], axis=0).reshape(-1)
            ])

            X.append(feat); Y.append(1); centres_list.append(centre_id)

            perm = np.random.permutation(len(z_combined))
            z_shuf = z_combined[perm]
            dz_shuf = z_shuf[1:] - z_shuf[:-1]
            feat_shuf = np.concatenate([
                dz_shuf.mean(0), dz_shuf.std(0),
                np.percentile(dz_shuf, [25, 50, 75], axis=0).reshape(-1)
            ])
            X.append(feat_shuf); Y.append(0); centres_list.append(centre_id)

        except Exception as e:
            continue

    return np.array(X), np.array(Y), np.array(centres_list)

def evaluate_TOP(X, Y):
    """TOP evaluation with train/test split"""
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
    print("🚀 DUAL TEMPORAL ENCODER - FULL PIPELINE")
    print("="*60)

    RESULTS_DIR = Path("/content/results")
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Metadata
    df = create_metadata()
    
    # 2. SPLIT DATA (70/15/15)
    df_train, df_val, df_test = split_data(df, 0.70, 0.15, 0.15)
    
    # Save split info
    df_train.to_csv(RESULTS_DIR / 'metadata_train.csv', index=False)
    df_val.to_csv(RESULTS_DIR / 'metadata_val.csv', index=False)
    df_test.to_csv(RESULTS_DIR / 'metadata_test.csv', index=False)

    import json
    stats = {
        'total_clips': len(df),
        'train_clips': len(df_train),
        'val_clips': len(df_val),
        'test_clips': len(df_test),
        'A_fixed': len(df[df['dataset_type'] == 'A_fixed']),
        'B_variable': len(df[df['dataset_type'] == 'B_variable']),
        'centres': df.groupby('centre').size().to_dict(),
        'unique_subjects_total': df['subject'].nunique(),
        'unique_subjects_train': df_train['subject'].nunique(),
        'unique_subjects_val': df_val['subject'].nunique(),
        'unique_subjects_test': df_test['subject'].nunique()
    }

    with open(RESULTS_DIR / 'metadata_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # 3. Build datasets
    dataset_train = build_dataset(df_train, batch_size=BATCH_SIZE)
    dataset_val = build_dataset(df_val, batch_size=BATCH_SIZE)

    # 4. Build & Train
    model = DualTemporalModel(latent_dim=LATENT_DIM, num_centres=NUM_CENTRES)
    model, history = train_dual_model(
        model, 
        dataset_train, 
        dataset_val,
        epochs=EPOCHS, 
        steps_per_epoch=200,
        val_steps=50
    )

    # 5. Load best model
    print("\n📥 Loading best model from validation...")
    model.load_weights("/content/DUAL_best_model.weights.h5")

    # 6. Save models
    import pickle
    with open(RESULTS_DIR / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    with open(RESULTS_DIR / 'training_history.json', 'w') as f:
        json.dump({k: {kk: [float(x) for x in vv] for kk, vv in v.items()} 
                  for k, v in history.items()}, f, indent=2)

    model.FE_encoder.save(RESULTS_DIR / "FE_encoder.keras")
    model.RE_encoder.save(RESULTS_DIR / "RE_encoder.keras")
    print("✅ Final models saved!")

    # 7. Evaluate on TEST SET
    print("\n📊 Extracting features from TEST set...")
    X_test, Y_test, centres_test = extract_features_for_TOP(model, df_test, max_clips=500)

    if len(X_test) > 0:
        acc, ci_low, ci_high, p_val = evaluate_TOP(X_test, Y_test)

        top_results = {
            'accuracy': float(acc),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'p_value': float(p_val),
            'n_samples': len(X_test) // 2
        }

        with open(RESULTS_DIR / 'TOP_results.json', 'w') as f:
            json.dump(top_results, f, indent=2)

        np.savez(RESULTS_DIR / 'TOP_features.npz', X=X_test, Y=Y_test, centres=centres_test)

        # 8. Plot training curves
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = range(1, len(history['train']['L_total']) + 1)

        # Overall losses
        axes[0,0].plot(epochs, history['train']['L_total'], 'o-', label='Train Total')
        axes[0,0].plot(epochs, history['val']['L_total'], 's-', label='Val Total')
        axes[0,0].plot(epochs, history['train']['L_FE'], '^-', label='Train FE', alpha=0.7)
        axes[0,0].plot(epochs, history['val']['L_FE'], 'v-', label='Val FE', alpha=0.7)
        axes[0,0].set_title('Overall Loss'); axes[0,0].legend(); axes[0,0].grid(True)

        # FE components
        axes[0,1].plot(epochs, history['train']['L_rec_FE'], 'o-', label='Train Rec')
        axes[0,1].plot(epochs, history['val']['L_rec_FE'], 's-', label='Val Rec')
        axes[0,1].plot(epochs, history['train']['L_scale'], '^-', label='Train Scale')
        axes[0,1].plot(epochs, history['val']['L_scale'], 'v-', label='Val Scale')
        axes[0,1].set_title('FE Reconstruction & Scale'); axes[0,1].legend(); axes[0,1].grid(True)
        axes[0,1].set_yscale('log')

        # Spectral & Adversarial
        axes[0,2].plot(epochs, history['train']['L_spec'], 'o-', label='Train Spec')
        axes[0,2].plot(epochs, history['val']['L_spec'], 's-', label='Val Spec')
        axes[0,2].plot(epochs, np.abs(history['train']['L_adv']), '^-', label='Train |Adv|')
        axes[0,2].plot(epochs, np.abs(history['val']['L_adv']), 'v-', label='Val |Adv|')
        axes[0,2].set_title('Spectral & Adversarial'); axes[0,2].legend(); axes[0,2].grid(True)
        axes[0,2].set_yscale('log')

        # RE components
        axes[1,0].plot(epochs, history['train']['L_rec_RE'], 'o-', label='Train Rec')
        axes[1,0].plot(epochs, history['val']['L_rec_RE'], 's-', label='Val Rec')
        axes[1,0].plot(epochs, history['train']['L_batch'], '^-', label='Train Batch')
        axes[1,0].plot(epochs, history['val']['L_batch'], 'v-', label='Val Batch')
        axes[1,0].set_title('RE Components'); axes[1,0].legend(); axes[1,0].grid(True)
        axes[1,0].set_yscale('log')

        # KL divergence
        axes[1,1].plot(epochs, history['train']['L_KL'], 'o-', label='Train KL')
        axes[1,1].plot(epochs, history['val']['L_KL'], 's-', label='Val KL')
        axes[1,1].set_title('KL Divergence'); axes[1,1].legend(); axes[1,1].grid(True)

        # Discriminator
        axes[1,2].plot(epochs, history['train']['L_disc'], 'o-', label='Train', color='red')
        axes[1,2].plot(epochs, history['val']['L_disc'], 's-', label='Val', color='blue')
        axes[1,2].axhline(y=np.log(NUM_CENTRES), linestyle='--', color='gray',
                         label=f'Random ({NUM_CENTRES} centres)')
        axes[1,2].set_title('Discriminator Loss'); axes[1,2].legend(); axes[1,2].grid(True)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 9. Summary
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED - FINAL RESULTS (TEST SET)")
        print("="*60)
        print(f"Training epochs: {EPOCHS}")
        print(f"Latent dimension: {LATENT_DIM}")
        print(f"Number of centres: {NUM_CENTRES}")
        print(f"Train/Val/Test split: 70/15/15")
        print(f"\nTotal clips: {len(df)}")
        print(f"  - Train: {len(df_train)} clips ({stats['unique_subjects_train']} subjects)")
        print(f"  - Val:   {len(df_val)} clips ({stats['unique_subjects_val']} subjects)")
        print(f"  - Test:  {len(df_test)} clips ({stats['unique_subjects_test']} subjects)")
        print(f"  - A_fixed: {stats['A_fixed']}")
        print(f"  - B_variable: {stats['B_variable']}")
        print(f"\n📊 Results:")
        print(f"TOP Accuracy: {acc*100:.2f}% [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
        print(f"p-value: {p_val:.5f}")
        print(f"\n💾 All results saved to: {RESULTS_DIR}")
        print("="*60)

        # 10. Copy to Drive
        try:
            import shutil
            DRIVE_PATH = Path("/content/drive/MyDrive/intrapartum_results")
            DRIVE_PATH.mkdir(exist_ok=True)

            for file in RESULTS_DIR.glob('*'):
                shutil.copy(file, DRIVE_PATH / file.name)

            print(f"\n✅ All files backed up to Drive: {DRIVE_PATH}")
            print("\n📂 Saved files:")
            for file in sorted(DRIVE_PATH.glob('*')):
                size_kb = file.stat().st_size / 1024
                print(f"  - {file.name} ({size_kb:.1f} KB)")

        except Exception as e:
            print(f"\n⚠️ Could not backup to Drive: {e}")
            print("   Files are saved locally in /content/results/")

    else:
        print("⚠️ No features extracted. Check paths and data integrity.")