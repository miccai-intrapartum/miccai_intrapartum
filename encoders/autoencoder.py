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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
ROOT          = Path("/content/dataset/preprocessed")
DIRS_TO_SCAN  = [ROOT / "A_fixed", ROOT / "B_variable"]
ENCODER_PATH  = "/content/encoder.keras"         # trained autoencoder encoder
WEIGHTS_PATH  = "/content/AE_best_model.weights.h5"

# Hyperparameters (mirror HCA config)
IMG_SIZE    = 128
T_WIN       = 64
LATENT_DIM  = 128
BATCH_SIZE  = 4
EPOCHS      = 10
LR          = 2e-4

# Temporal structure (mirrors HCA segments)
NUM_SEGMENTS = 4
SEGMENT_SIZE = T_WIN // NUM_SEGMENTS   # 16

# Loss weights
W_REC   = 1.0
W_SCALE = 1.0
W_SPEC  = 0.5
W_ADV   = 1.0
W_BATCH = 1.0
W_KL    = 0.01

# Evaluation
MAX_CLIPS       = 500
BOOTSTRAP_ITERS = 1000
PERMUTATION_ITERS = 500

# ============================================================
# 1. DATA LOADING & METADATA
# ============================================================
def create_metadata():
    """
    Scan A_fixed and B_variable directories for .npy clip files.
    Returns a DataFrame with clip paths and centre labels.
    """
    rows = []
    print(f"Scanning directories: {[d.name for d in DIRS_TO_SCAN]}...")

    for root_path in DIRS_TO_SCAN:
        if not root_path.exists():
            print(f"  Warning: {root_path} does not exist. Skipping.")
            continue

        dataset_type = root_path.name
        all_npy      = list(root_path.rglob("*.npy"))
        video_files  = [f for f in all_npy if "_valid" not in f.name]
        print(f"   -> Found {len(video_files)} video files in {dataset_type}")

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
                'source_root' : str(root_path),
                'dataset_type': dataset_type,
                'centre'      : centre,
                'subject'     : subject,
                'clip'        : clip,
                'full_path_A' : str(clip_file),
                'full_path_V' : str(valid_file),
                'valid_exists': valid_file.exists()
            })

    df = pd.DataFrame(rows)
    print(f"Metadata created: {len(df)} clips found.")

    if df.empty:
        raise ValueError("No data found! Check directory paths.")

    print("\nDistribution by Dataset Type:")
    print(df.groupby('dataset_type').size())
    print("\nDistribution by Centre:")
    print(df.groupby('centre').size())

    return df


CENTRE_MAP  = {'Merkez 2': 0, 'Merkez 6': 1, 'Merkez 8': 2}
NUM_CENTRES = len(CENTRE_MAP)

# ============================================================
# 1.5 TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# ============================================================
def split_data(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Patient-level split to prevent data leakage.
    Each subject appears in exactly one partition.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    subjects = df['subject'].unique()
    np.random.shuffle(subjects)

    n       = len(subjects)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_subjects = subjects[:n_train]
    val_subjects   = subjects[n_train : n_train + n_val]
    test_subjects  = subjects[n_train + n_val :]

    df_train = df[df['subject'].isin(train_subjects)].reset_index(drop=True)
    df_val   = df[df['subject'].isin(val_subjects)].reset_index(drop=True)
    df_test  = df[df['subject'].isin(test_subjects)].reset_index(drop=True)

    print("\n" + "=" * 60)
    print("DATA SPLIT (by subject)")
    print("=" * 60)
    print(f"Train : {len(train_subjects)} subjects, {len(df_train)} clips")
    print(f"Val   : {len(val_subjects)} subjects,   {len(df_val)} clips")
    print(f"Test  : {len(test_subjects)} subjects,  {len(df_test)} clips")

    print("\nTrain centre distribution:")
    print(df_train.groupby('centre').size())
    print("\nVal centre distribution:")
    print(df_val.groupby('centre').size())
    print("\nTest centre distribution:")
    print(df_test.groupby('centre').size())
    print("=" * 60)

    return df_train, df_val, df_test

# ============================================================
# 2. DATA GENERATOR
# ============================================================
def build_dataset(df, batch_size=BATCH_SIZE):
    """
    Subject-balanced data generator.
    Mirrors HCA pipeline's build_dataset function.
    """
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
        for attempt in range(max_attempts):
            centre, subject = random.choice(subjects)
            clip_info       = random.choice(subj2clips[(centre, subject)])

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

                s   = random.randint(idx[0], idx[-1] - T_WIN + 1)
                seq = A[s : s + T_WIN]

                seq = tf.image.resize(
                    seq[..., None], (IMG_SIZE, IMG_SIZE), method="area"
                ).numpy()

                x         = (seq / 255.0).astype(np.float32)
                centre_id = CENTRE_MAP[centre]

                return x, centre_id

            except Exception:
                continue

        # Fallback: dummy data
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
    """
    Shared CNN backbone for frame-level encoding.
    5-layer progressive downsampling.
    """
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
    """
    Decoder: latent vector → reconstructed frame.
    Mirrors HCA decoder architecture.
    """
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
    """
    Multi-head self-attention over the temporal dimension.
    Captures long-range frame dependencies.
    """
    def __init__(self, dim, num_heads=4, dropout=0.1, name_prefix=""):
        super().__init__(name=f"{name_prefix}_temporal_attn")
        self.mha  = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout
        )
        self.norm = layers.LayerNormalization()

    def call(self, z, training=False):
        attn_out = self.mha(z, z, training=training)
        return self.norm(z + attn_out)


class CentreDiscriminator(keras.Model):
    """
    Adversarial discriminator: predicts acquisition centre from
    the latent embedding so we can push the encoder to be site-invariant.
    """
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
    """
    Centre-conditioned normalization:
    normalizes per-sample then rescales with centre-specific gamma / beta.
    """
    def __init__(self, num_centres, dim):
        super().__init__()
        self.gamma_emb = layers.Embedding(num_centres, dim)
        self.beta_emb  = layers.Embedding(num_centres, dim)
        self.eps       = 1e-6

    def call(self, z, centre_id):
        mean   = tf.reduce_mean(z, axis=[1, 2], keepdims=True)
        std    = tf.math.reduce_std(z, axis=[1, 2], keepdims=True)
        z_norm = (z - mean) / (std + self.eps)

        gamma  = self.gamma_emb(centre_id)
        beta   = self.beta_emb(centre_id)

        return gamma[:, None, :] * z_norm + beta[:, None, :]

# ============================================================
# 4. AUTOENCODER MODEL
# ============================================================
class AutoencoderModel(keras.Model):
    """
    Frame-level Autoencoder with:
      - Shared CNN encoder backbone
      - Temporal self-attention
      - Centre-conditioned batch normalization (RE path)
      - Adversarial centre discriminator
      - Frame-level decoder for reconstruction
    """
    def __init__(self, latent_dim=LATENT_DIM, num_centres=NUM_CENTRES):
        super().__init__(name="autoencoder")

        # Encoder
        self.encoder       = build_cnn_backbone(latent_dim, name="ae_encoder")
        self.temp_attention = TemporalAttention(latent_dim, name_prefix="ae")
        self.batch_norm    = BatchConditionalNorm(num_centres, latent_dim)

        # Decoder
        self.decoder = build_decoder(latent_dim)

        # Adversarial components
        self.discriminator    = CentreDiscriminator(num_centres)
        self.centre_classifier = layers.Dense(num_centres, name="centre_classifier")

        self.latent_dim  = latent_dim
        self.num_centres = num_centres

    def encode(self, frames, centre_id, training=False):
        """
        Encode a sequence of frames into latent representations.

        Args:
            frames    : (B, T, H, W, 1)
            centre_id : (B,)
        Returns:
            z         : (B, T, D)  centre-normalised latent sequence
            z_raw     : (B, T, D)  raw encoder output (no normalisation)
        """
        # Frame-level encoding
        z_raw = layers.TimeDistributed(self.encoder)(frames)   # (B, T, D)

        # Temporal self-attention
        z_attn = self.temp_attention(z_raw, training=training)  # (B, T, D)

        # Centre-conditioned normalisation
        z = self.batch_norm(z_attn, centre_id)                  # (B, T, D)

        return z, z_raw

    def decode(self, z):
        """
        Decode latent sequence back to frame sequence.

        Args:
            z   : (B, T, D)
        Returns:
            rec : (B, T, H, W, 1)
        """
        return layers.TimeDistributed(self.decoder)(z)

    def call(self, frames, centre_id, training=False):
        z, z_raw = self.encode(frames, centre_id, training=training)
        x_rec    = self.decode(z)

        return {
            'z'      : z,
            'z_raw'  : z_raw,
            'x_rec'  : x_rec
        }

# ============================================================
# 5. LOSS FUNCTIONS
# ============================================================
def tf_percentile(x, q):
    """Differentiable percentile computation."""
    x = tf.sort(tf.reshape(x, [-1]))
    n = tf.size(x)
    k = tf.cast((q / 100.0) * tf.cast(n - 1, tf.float32), tf.int32)
    return x[k]


def scale_regularization(dz):
    """
    MAD-based scale regularization on temporal differences.
    Penalises large or erratic magnitude changes between frames.
    """
    mag = tf.norm(dz, axis=-1)
    med = tf_percentile(mag, 50.0)
    mad = tf_percentile(tf.abs(mag - med), 50.0) + 1e-6
    return tf.reduce_mean(tf.abs(mag - med) / mad)


def spectral_loss(dz):
    """
    High-frequency suppression in the temporal domain.
    Encourages smooth latent trajectories over time.
    """
    mag = tf.norm(dz, axis=-1)
    mag = mag - tf.reduce_mean(mag, axis=1, keepdims=True)
    fft     = tf.signal.rfft(mag)
    power   = tf.abs(fft) ** 2
    cutoff  = tf.cast(0.2 * tf.cast(tf.shape(power)[1], tf.float32), tf.int32)
    return tf.reduce_mean(power[:, cutoff:])


def reconstruction_loss(x_orig, x_rec):
    """Pixel-wise MSE reconstruction loss."""
    return tf.reduce_mean(tf.square(x_orig - x_rec))

# ============================================================
# 6. TRAINING
# ============================================================
def train_autoencoder(model, dataset_train, dataset_val,
                      epochs=EPOCHS, steps_per_epoch=200, val_steps=50):
    """
    Train the autoencoder with:
      - Reconstruction loss
      - Scale + spectral regularization
      - Adversarial centre-invariance
      - Centre classification (RE path)
      - Train / val loss tracking
      - Best model checkpointing
    """
    opt_enc  = keras.optimizers.Adam(LR)
    opt_disc = keras.optimizers.Adam(LR)

    history = {
        'train': {
            'L_total': [], 'L_rec': [], 'L_scale': [],
            'L_spec': [], 'L_adv': [], 'L_batch': []
        },
        'val': {
            'L_total': [], 'L_rec': [], 'L_scale': [],
            'L_spec': [], 'L_adv': [], 'L_batch': []
        }
    }

    @tf.function
    def train_step(frames, centre_id):
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(frames, centre_id, training=True)

            z     = outputs['z']        # (B, T, D)
            x_rec = outputs['x_rec']    # (B, T, H, W, 1)

            # Reconstruction
            L_rec = reconstruction_loss(frames, x_rec)

            # Temporal regularization
            dz      = z[:, 1:] - z[:, :-1]
            L_scale = scale_regularization(dz)
            L_spec  = spectral_loss(dz)

            # Adversarial: encoder tries to fool discriminator
            pred_disc = model.discriminator(z, training=True)
            L_adv     = -tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_disc, from_logits=True
                )
            )

            # Centre classification loss (encourage centre-separable RE path)
            z_pooled = tf.reduce_mean(z, axis=1)
            pred_cls = model.centre_classifier(z_pooled)
            L_batch  = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_cls, from_logits=True
                )
            )

            # Encoder total loss
            L_enc = (
                W_REC   * L_rec   +
                W_SCALE * L_scale +
                W_SPEC  * L_spec  +
                W_ADV   * L_adv   +
                W_BATCH * L_batch
            )

            # Discriminator loss (classify centre correctly)
            L_disc = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    centre_id, pred_disc, from_logits=True
                )
            )

        # Update encoder + decoder
        vars_enc  = (
            model.encoder.trainable_variables +
            model.decoder.trainable_variables +
            model.temp_attention.trainable_variables +
            model.batch_norm.trainable_variables +
            model.centre_classifier.trainable_variables
        )
        grads_enc = tape.gradient(L_enc, vars_enc)
        opt_enc.apply_gradients(zip(grads_enc, vars_enc))

        # Update discriminator
        vars_disc  = model.discriminator.trainable_variables
        grads_disc = tape.gradient(L_disc, vars_disc)
        opt_disc.apply_gradients(zip(grads_disc, vars_disc))

        del tape

        return {
            'L_total': L_enc,
            'L_rec'  : L_rec,
            'L_scale': L_scale,
            'L_spec' : L_spec,
            'L_adv'  : L_adv,
            'L_batch': L_batch
        }

    @tf.function
    def val_step(frames, centre_id):
        """Validation: forward pass only, no gradient update."""
        outputs = model(frames, centre_id, training=False)

        z     = outputs['z']
        x_rec = outputs['x_rec']

        L_rec = reconstruction_loss(frames, x_rec)

        dz      = z[:, 1:] - z[:, :-1]
        L_scale = scale_regularization(dz)
        L_spec  = spectral_loss(dz)

        pred_disc = model.discriminator(z, training=False)
        L_adv     = -tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, pred_disc, from_logits=True
            )
        )

        z_pooled = tf.reduce_mean(z, axis=1)
        pred_cls = model.centre_classifier(z_pooled)
        L_batch  = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                centre_id, pred_cls, from_logits=True
            )
        )

        L_total = (
            W_REC   * L_rec   +
            W_SCALE * L_scale +
            W_SPEC  * L_spec  +
            W_ADV   * L_adv   +
            W_BATCH * L_batch
        )

        return {
            'L_total': L_total,
            'L_rec'  : L_rec,
            'L_scale': L_scale,
            'L_spec' : L_spec,
            'L_adv'  : L_adv,
            'L_batch': L_batch
        }

    # ── Training loop ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AUTOENCODER TRAINING")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):

        # TRAIN
        losses_train = []
        for frames, centre_id in dataset_train.take(steps_per_epoch):
            loss_dict = train_step(frames, centre_id)
            losses_train.append({k: float(v) for k, v in loss_dict.items()})

        avg_train = {
            k: np.mean([d[k] for d in losses_train])
            for k in losses_train[0].keys()
        }

        # VALIDATE
        losses_val = []
        for frames, centre_id in dataset_val.take(val_steps):
            loss_dict = val_step(frames, centre_id)
            losses_val.append({k: float(v) for k, v in loss_dict.items()})

        avg_val = {
            k: np.mean([d[k] for d in losses_val])
            for k in losses_val[0].keys()
        }

        # Record history
        for key in history['train'].keys():
            history['train'][key].append(avg_train[key])
            history['val'][key].append(avg_val[key])

        # Print epoch summary
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"TRAIN - Total: {avg_train['L_total']:.4f} | "
              f"Rec: {avg_train['L_rec']:.4f} | "
              f"Scale: {avg_train['L_scale']:.4f} | "
              f"Spec: {avg_train['L_spec']:.4f}")
        print(f"        Adv: {avg_train['L_adv']:.4f} | "
              f"Batch: {avg_train['L_batch']:.4f}")
        print(f"VAL   - Total: {avg_val['L_total']:.4f} | "
              f"Rec: {avg_val['L_rec']:.4f} | "
              f"Adv: {avg_val['L_adv']:.4f}")

        # Checkpoint
        if avg_val['L_total'] < best_val_loss:
            best_val_loss = avg_val['L_total']
            model.save_weights(WEIGHTS_PATH)
            print(f"  Best model saved! (Val Loss: {best_val_loss:.4f})")

    print(f"\nTraining completed. Best val loss: {best_val_loss:.4f}")
    return model, history


def plot_training_history(history, save_path="/content/ae_training_history.png"):
    """Plot training vs validation loss curves."""
    keys  = list(history['train'].keys())
    n     = len(keys)
    cols  = 3
    rows  = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for i, key in enumerate(keys):
        axes[i].plot(history['train'][key], label='Train', color='steelblue')
        axes[i].plot(history['val'][key],   label='Val',   color='darkorange')
        axes[i].set_title(key)
        axes[i].set_xlabel('Epoch')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Autoencoder Training History", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")

# ============================================================
# 7. FEATURE EXTRACTION FOR TOP
# ============================================================
def extract_TOP_features(model, df, max_clips=MAX_CLIPS):
    """
    Extract delta-based temporal features for TOP evaluation.

    For each clip:
      - POSITIVE sample: z computed from correct temporal order
      - NEGATIVE sample: z computed from randomly shuffled frames

    Feature vector = [mean(dz), std(dz), Q25(dz), Q50(dz), Q75(dz)]
    concatenated across all latent dimensions.

    Args:
        model     : trained AutoencoderModel
        df        : metadata DataFrame
        max_clips : maximum number of clips to process

    Returns:
        X              : (N, feat_dim) feature matrix
        Y              : (N,) labels  (1=correct, 0=shuffled)
        centres_list   : (N,) centre ids for stratification
    """
    X, Y, centres_list = [], [], []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=min(len(df), max_clips),
             desc="Extracting TOP features")
    ):
        if i >= max_clips:
            break

        A_path = Path(row['full_path_A'])
        V_path = Path(row['full_path_V'])

        if not A_path.exists():
            continue

        try:
            A     = np.load(A_path).astype(np.float32) / 255.0
            valid = (
                np.load(V_path).astype(bool)
                if row['valid_exists'] and V_path.exists()
                else np.ones(len(A), dtype=bool)
            )

            idx = np.where(valid)[0]
            if len(idx) < 5:
                continue

            # Uniform temporal sampling → T_WIN frames
            T       = min(len(idx), T_WIN)
            sampled = np.linspace(idx[0], idx[-1], T).astype(int)
            A       = A[sampled]

            # Resize + channel dim → (T, 128, 128, 1)
            A = tf.image.resize(
                A[..., None], (IMG_SIZE, IMG_SIZE), method='area'
            ).numpy()

            centre_id    = CENTRE_MAP[row['centre']]
            centre_id_tf = tf.constant([centre_id], dtype=tf.int32)

            # ── POSITIVE: correct temporal order ──────────────
            A_batch = A[None, ...]                              # (1, T, H, W, 1)
            outputs = model(A_batch, centre_id_tf, training=False)
            z       = outputs['z'].numpy()[0]                   # (T, D)

            dz   = z[1:] - z[:-1]                              # (T-1, D)
            feat = np.concatenate([
                dz.mean(0),
                dz.std(0),
                np.percentile(dz, [25, 50, 75], axis=0).reshape(-1)
            ])
            X.append(feat); Y.append(1); centres_list.append(centre_id)

            # ── NEGATIVE: shuffled temporal order ─────────────
            perm   = np.random.permutation(len(A))
            A_shuf = A[perm][None, ...]                         # (1, T, H, W, 1)

            outputs_s = model(A_shuf, centre_id_tf, training=False)
            z_s       = outputs_s['z'].numpy()[0]               # (T, D)

            dz_s   = z_s[1:] - z_s[:-1]
            feat_s = np.concatenate([
                dz_s.mean(0),
                dz_s.std(0),
                np.percentile(dz_s, [25, 50, 75], axis=0).reshape(-1)
            ])
            X.append(feat_s); Y.append(0); centres_list.append(centre_id)

        except Exception:
            continue

    return np.array(X), np.array(Y), np.array(centres_list)

# ============================================================
# 8. EVALUATION: TEMPORAL ORDER PREDICTION (TOP)
# ============================================================
def evaluate_TOP(X, Y):
    """
    Temporal Order Prediction (TOP) evaluation with:
      - 70 / 30 train-test split
      - Bootstrap 95% confidence interval (1000 iterations)
      - Permutation test p-value (500 iterations)

    Args:
        X : (N, feat_dim) feature matrix
        Y : (N,) binary labels

    Returns:
        acc, ci_low, ci_high, p_value
    """
    print("\n" + "=" * 60)
    print("TEMPORAL ORDER PREDICTION (TOP)")
    print("=" * 60)
    print(f"Samples   : {len(X)}")
    print(f"Pos / Neg : {Y.sum()} / {(1 - Y).sum()}")
    print(f"Feat dim  : {X.shape[1]}")

    # Feature scaling
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train / test split ────────────────────────────────────
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y,
        test_size=0.3,
        random_state=SEED,
        stratify=Y
    )

    # ── Train classifier ──────────────────────────────────────
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=SEED
    )
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)
    acc    = accuracy_score(Y_test, Y_pred)

    print(f"\n  Logistic Regression on held-out test set:")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(Y_test, Y_pred,
                                target_names=['Shuffled', 'Correct'],
                                digits=4))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(Y_test, Y_pred))

    # ── Bootstrap 95% CI ──────────────────────────────────────
    print(f"\n  Running bootstrap CI ({BOOTSTRAP_ITERS} iterations)...")
    boot_acc = []
    for _ in range(BOOTSTRAP_ITERS):
        X_b, Y_b = resample(X_test, Y_test, random_state=None)
        clf_b    = LogisticRegression(max_iter=500, random_state=SEED)
        clf_b.fit(X_train, Y_train)
        boot_acc.append(accuracy_score(Y_b, clf_b.predict(X_b)))

    ci_low, ci_high = np.percentile(boot_acc, [2.5, 97.5])
    print(f"  95% CI: [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]")

    # ── Permutation test ──────────────────────────────────────
    print(f"\n  Running permutation test ({PERMUTATION_ITERS} iterations)...")
    perm_acc = []
    for _ in range(PERMUTATION_ITERS):
        Y_perm = np.random.permutation(Y_train)
        clf_p  = LogisticRegression(max_iter=500, random_state=SEED)
        clf_p.fit(X_train, Y_perm)
        perm_acc.append(accuracy_score(Y_test, clf_p.predict(X_test)))

    p_value = np.mean(np.array(perm_acc) >= acc)
    print(f"  p-value : {p_value:.5f}")

    # ── Null distribution plot ────────────────────────────────
    _plot_permutation_null(perm_acc, acc, ci_low, ci_high)

    return acc, ci_low, ci_high, p_value


def _plot_permutation_null(perm_acc, acc, ci_low, ci_high,
                            save_path="/content/ae_top_null_distribution.png"):
    """Plot null distribution from permutation test."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(perm_acc, bins=30, color='steelblue', alpha=0.7,
            edgecolor='white', label='Null distribution')
    ax.axvline(acc,     color='crimson',   lw=2, linestyle='-',
               label=f'Observed acc = {acc*100:.2f}%')
    ax.axvline(ci_low,  color='darkorange', lw=1.5, linestyle='--',
               label=f'95% CI lower = {ci_low*100:.2f}%')
    ax.axvline(ci_high, color='darkorange', lw=1.5, linestyle=':',
               label=f'95% CI upper = {ci_high*100:.2f}%')

    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Count',    fontsize=12)
    ax.set_title('Permutation Null Distribution – TOP (Autoencoder)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Null distribution plot saved to {save_path}")

# ============================================================
# 9. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("AUTOENCODER – TEMPORAL ORDER PREDICTION FULL PIPELINE")
    print("=" * 60)

    # 1. Create metadata
    df = create_metadata()

    # 2. Split data (70/15/15, subject-level)
    df_train, df_val, df_test = split_data(df, 0.70, 0.15, 0.15)

    # 3. Build tf.data pipelines
    dataset_train = build_dataset(df_train, batch_size=BATCH_SIZE)
    dataset_val   = build_dataset(df_val,   batch_size=BATCH_SIZE)

    # 4. Build autoencoder
    model = AutoencoderModel(
        latent_dim=LATENT_DIM,
        num_centres=NUM_CENTRES
    )

    # 5. Train with validation
    model, history = train_autoencoder(
        model,
        dataset_train,
        dataset_val,
        epochs=EPOCHS,
        steps_per_epoch=200,
        val_steps=50
    )

    # 6. Plot training curves
    plot_training_history(history)

    # 7. Load best checkpoint
    print("\nLoading best model from validation checkpoint...")
    model.load_weights(WEIGHTS_PATH)
    print("Best model loaded.")

    # 8. Extract TOP features from TEST set
    print("\nExtracting TOP features from TEST set...")
    X_test, Y_test, centres_test = extract_TOP_features(
        model, df_test, max_clips=MAX_CLIPS
    )
    print(f"Feature matrix: {X_test.shape}")
    print(f"Label balance : pos={Y_test.sum()}, neg={(1-Y_test).sum()}")

    if len(X_test) == 0:
        print("No features extracted. Check paths and data integrity.")
    else:
        # 9. Evaluate TOP
        acc, ci_low, ci_high, p_val = evaluate_TOP(X_test, Y_test)

        # 10. Final summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS – TEST SET")
        print("=" * 60)
        print(f"Architecture    : Autoencoder (AE)")
        print(f"Latent dim      : {LATENT_DIM}")
        print(f"Training epochs : {EPOCHS}")
        print(f"Number of centres: {NUM_CENTRES}")
        print(f"Train/Val/Test  : 70/15/15 (subject-level)")
        print(f"\nTrain clips : {len(df_train)}")
        print(f"Val clips   : {len(df_val)}")
        print(f"Test clips  : {len(df_test)}")
        print(f"TOP samples : {len(X_test)} (pos+neg pairs)")
        print(f"\nTOP Accuracy : {acc * 100:.2f}%")
        print(f"95% CI       : [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]")
        print(f"p-value      : {p_val:.5f}")
        print("=" * 60)