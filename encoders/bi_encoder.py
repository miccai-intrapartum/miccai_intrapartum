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
BATCH_SIZE = 4  # Will contain 2 pairs per batch
EPOCHS = 10
LR = 2e-4

# Loss weights
W_REC = 0.1
W_CONTRASTIVE = 1.0
W_TRIPLET = 0.5
MARGIN = 0.5  # Margin for contrastive/triplet loss

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
# 2. PAIR GENERATION FOR BI-ENCODER
# ============================================================
def create_pairs_metadata(df):
    """
    Create positive and negative pairs for contrastive learning

    POSITIVE PAIR: Same patient, different dataset (A_fixed vs B_variable)
    NEGATIVE PAIR: Different patients
    """
    # Group by (centre, subject, clip)
    df_A = df[df['dataset_type'] == 'A_fixed'].copy()
    df_B = df[df['dataset_type'] == 'B_variable'].copy()

    # Create lookup dictionaries
    df_A['key'] = df_A['centre'] + '_' + df_A['subject'] + '_' + df_A['clip']
    df_B['key'] = df_B['centre'] + '_' + df_B['subject'] + '_' + df_B['clip']

    A_dict = df_A.set_index('key').to_dict('index')
    B_dict = df_B.set_index('key').to_dict('index')

    # Find matching pairs (same patient, A and B versions exist)
    positive_pairs = []
    for key in A_dict:
        if key in B_dict:
            positive_pairs.append({
                'video_1': A_dict[key]['full_path_A'],
                'valid_1': A_dict[key]['full_path_V'],
                'valid_1_exists': A_dict[key]['valid_exists'],
                'video_2': B_dict[key]['full_path_A'],
                'valid_2': B_dict[key]['full_path_V'],
                'valid_2_exists': B_dict[key]['valid_exists'],
                'centre': A_dict[key]['centre'],
                'subject': A_dict[key]['subject'],
                'label': 1  # Positive pair
            })

    print(f"\n✅ Found {len(positive_pairs)} positive pairs (same patient, A vs B)")

    # Create negative pairs (different patients)
    all_subjects = list(set([p['subject'] for p in positive_pairs]))
    negative_pairs = []

    # Generate same number of negative pairs as positive
    for _ in range(len(positive_pairs)):
        # Randomly select 2 different patients
        if len(all_subjects) < 2:
            break
            
        subj1, subj2 = random.sample(all_subjects, 2)

        # Get random video from each patient
        pairs_subj1 = [p for p in positive_pairs if p['subject'] == subj1]
        pairs_subj2 = [p for p in positive_pairs if p['subject'] == subj2]

        if pairs_subj1 and pairs_subj2:
            p1 = random.choice(pairs_subj1)
            p2 = random.choice(pairs_subj2)

            negative_pairs.append({
                'video_1': p1['video_1'],
                'valid_1': p1['valid_1'],
                'valid_1_exists': p1['valid_1_exists'],
                'video_2': p2['video_2'],
                'valid_2': p2['valid_2'],
                'valid_2_exists': p2['valid_2_exists'],
                'centre': p1['centre'],
                'subject': f"{subj1}_vs_{subj2}",
                'label': 0  # Negative pair
            })

    print(f"✅ Generated {len(negative_pairs)} negative pairs (different patients)")

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    return pd.DataFrame(all_pairs)

# ============================================================
# 3. DATA GENERATOR FOR PAIRS
# ============================================================
def build_pair_dataset(pair_df, batch_size=BATCH_SIZE):
    """Generate batches of video pairs"""

    pairs_list = pair_df.to_dict('records')
    print(f"Total pairs for dataset: {len(pairs_list)}")

    def load_video(video_path, valid_path, valid_exists):
        """Load and preprocess a single video"""
        try:
            A = np.load(video_path)

            if valid_exists:
                valid = np.load(valid_path).astype(bool)
            else:
                valid = np.ones(len(A), dtype=bool)

            idx = np.where(valid)[0]
            if len(idx) < T_WIN + 2:
                return None

            # Random window
            s = random.randint(idx[0], idx[-1] - T_WIN + 1)
            seq = A[s:s+T_WIN]

            # Resize
            seq = tf.image.resize(seq[..., None], (IMG_SIZE, IMG_SIZE),
                                 method="area").numpy()

            x = (seq / 255.0).astype(np.float32)
            return x

        except Exception as e:
            return None

    def load_one_pair():
        """Load one pair of videos"""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            pair = random.choice(pairs_list)

            video1 = load_video(
                pair['video_1'],
                pair['valid_1'],
                pair['valid_1_exists']
            )
            video2 = load_video(
                pair['video_2'],
                pair['valid_2'],
                pair['valid_2_exists']
            )

            if video1 is not None and video2 is not None:
                label = pair['label']
                return video1, video2, label
        
        # Fallback: return dummy data if all attempts fail
        dummy = np.zeros((T_WIN, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        return dummy, dummy, 0

    def gen():
        while True:
            yield load_one_pair()

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((T_WIN, IMG_SIZE, IMG_SIZE, 1), tf.float32),
            tf.TensorSpec((T_WIN, IMG_SIZE, IMG_SIZE, 1), tf.float32),
            tf.TensorSpec((), tf.int32)
        )
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return ds

# ============================================================
# 4. BI-ENCODER ARCHITECTURE
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

class TemporalEncoder(keras.Model):
    """Temporal encoder with self-attention"""
    def __init__(self, latent_dim):
        super().__init__(name="temporal_encoder")
        self.backbone = build_cnn_backbone(latent_dim)
        self.attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=latent_dim // 4,
            dropout=0.1
        )
        self.norm = layers.LayerNormalization()
        self.latent_dim = latent_dim

    def call(self, frames, training=False):
        # frames: (B, T, H, W, 1)

        # Extract frame-level features
        z = layers.TimeDistributed(self.backbone)(frames)  # (B, T, D)

        # Temporal attention
        z_attn = self.attention(z, z, training=training)
        z = self.norm(z + z_attn)

        # Global temporal pooling
        z_global = tf.reduce_mean(z, axis=1)  # (B, D)

        # L2 normalize for similarity computation
        z_normalized = tf.nn.l2_normalize(z_global, axis=-1)

        return z_normalized, z  # normalized for comparison, full for reconstruction

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

class BiEncoderModel(keras.Model):
    """
    Bi-Encoder with Contrastive Learning

    Architecture:
    - Shared encoder for both videos in a pair
    - Contrastive loss: same patient → similar, different → dissimilar
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        # Shared encoder (Siamese architecture)
        self.encoder = TemporalEncoder(latent_dim)

        # Decoder for reconstruction (auxiliary task)
        self.decoder = build_decoder(latent_dim)

        self.latent_dim = latent_dim

    def call(self, video1, video2, training=False):
        """
        Args:
            video1: (B, T, H, W, 1) - First video in pair
            video2: (B, T, H, W, 1) - Second video in pair

        Returns:
            z1_norm: Normalized embedding for video1
            z2_norm: Normalized embedding for video2
            z1_full: Full temporal features for video1
            z2_full: Full temporal features for video2
        """
        # Encode both videos with SHARED encoder
        z1_norm, z1_full = self.encoder(video1, training=training)
        z2_norm, z2_full = self.encoder(video2, training=training)

        return {
            'z1_norm': z1_norm,      # (B, D) - normalized
            'z2_norm': z2_norm,      # (B, D) - normalized
            'z1_full': z1_full,      # (B, T, D) - full temporal
            'z2_full': z2_full       # (B, T, D) - full temporal
        }

# ============================================================
# 5. LOSS FUNCTIONS
# ============================================================
def contrastive_loss(z1, z2, labels, margin=MARGIN):
    """
    Contrastive loss for pairs

    Args:
        z1, z2: Normalized embeddings (B, D)
        labels: 1 if same patient, 0 if different (B,)
        margin: Margin for negative pairs

    Returns:
        loss: Contrastive loss value
    """
    # Euclidean distance
    distance = tf.sqrt(tf.reduce_sum(tf.square(z1 - z2), axis=1))

    # Convert labels to float
    labels = tf.cast(labels, tf.float32)

    # Positive pairs: minimize distance
    loss_positive = labels * tf.square(distance)

    # Negative pairs: maximize distance (up to margin)
    loss_negative = (1 - labels) * tf.square(tf.maximum(0.0, margin - distance))

    return tf.reduce_mean(loss_positive + loss_negative)

def reconstruction_loss(frames, z_full, decoder):
    """
    Reconstruction loss (auxiliary task)

    Helps encoder learn meaningful representations
    """
    # Reconstruct middle frame from temporal features
    z_middle = z_full[:, T_WIN // 2, :]  # (B, D)
    frame_middle = frames[:, T_WIN // 2, :, :, :]  # (B, H, W, 1)

    recon = decoder(z_middle)

    loss = tf.reduce_mean(tf.square(frame_middle - recon))

    return loss

# ============================================================
# 6. TRAINING
# ============================================================
def train_biencoder_model(model, dataset_train, dataset_val, epochs=EPOCHS, 
                         steps_per_epoch=200, val_steps=50):
    """Train bi-encoder with contrastive learning"""

    optimizer = keras.optimizers.Adam(LR)

    @tf.function
    def train_step(video1, video2, labels):
        with tf.GradientTape() as tape:
            outputs = model(video1, video2, training=True)

            z1_norm = outputs['z1_norm']
            z2_norm = outputs['z2_norm']
            z1_full = outputs['z1_full']
            z2_full = outputs['z2_full']

            # Contrastive loss
            L_contrastive = contrastive_loss(z1_norm, z2_norm, labels, MARGIN)

            # Reconstruction loss (auxiliary)
            L_rec1 = reconstruction_loss(video1, z1_full, model.decoder)
            L_rec2 = reconstruction_loss(video2, z2_full, model.decoder)
            L_rec = (L_rec1 + L_rec2) / 2

            # Total loss
            L_total = W_CONTRASTIVE * L_contrastive + W_REC * L_rec

        # Update
        trainable_vars = model.trainable_variables
        grads = tape.gradient(L_total, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))

        # Compute accuracy
        distance = tf.sqrt(tf.reduce_sum(tf.square(z1_norm - z2_norm), axis=1))
        predictions = tf.cast(distance < 0.5, tf.int32)  # threshold at 0.5
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

        return {
            'L_total': L_total,
            'L_contrastive': L_contrastive,
            'L_rec': L_rec,
            'accuracy': accuracy
        }

    @tf.function
    def val_step(video1, video2, labels):
        """Validation step (no gradient update)"""
        outputs = model(video1, video2, training=False)

        z1_norm = outputs['z1_norm']
        z2_norm = outputs['z2_norm']
        z1_full = outputs['z1_full']
        z2_full = outputs['z2_full']

        # Contrastive loss
        L_contrastive = contrastive_loss(z1_norm, z2_norm, labels, MARGIN)

        # Reconstruction loss
        L_rec1 = reconstruction_loss(video1, z1_full, model.decoder)
        L_rec2 = reconstruction_loss(video2, z2_full, model.decoder)
        L_rec = (L_rec1 + L_rec2) / 2

        # Total loss
        L_total = W_CONTRASTIVE * L_contrastive + W_REC * L_rec

        # Accuracy
        distance = tf.sqrt(tf.reduce_sum(tf.square(z1_norm - z2_norm), axis=1))
        predictions = tf.cast(distance < 0.5, tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

        return {
            'L_total': L_total,
            'L_contrastive': L_contrastive,
            'L_rec': L_rec,
            'accuracy': accuracy
        }

    # Training loop
    print("\n" + "="*60)
    print("BI-ENCODER CONTRASTIVE TRAINING")
    print("="*60)

    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # TRAINING
        losses_train = []
        for step, (video1, video2, labels) in enumerate(
            dataset_train.take(steps_per_epoch), 1
        ):
            loss_dict = train_step(video1, video2, labels)
            losses_train.append({k: float(v) for k, v in loss_dict.items()})

        # Average training losses
        avg_train = {
            k: np.mean([d[k] for d in losses_train])
            for k in losses_train[0].keys()
        }

        # VALIDATION
        losses_val = []
        for step, (video1, video2, labels) in enumerate(
            dataset_val.take(val_steps), 1
        ):
            loss_dict = val_step(video1, video2, labels)
            losses_val.append({k: float(v) for k, v in loss_dict.items()})

        # Average validation losses
        avg_val = {
            k: np.mean([d[k] for d in losses_val])
            for k in losses_val[0].keys()
        }

        # Print results
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"TRAIN - Total: {avg_train['L_total']:.4f} | "
              f"Contrastive: {avg_train['L_contrastive']:.4f} | "
              f"Recon: {avg_train['L_rec']:.4f} | "
              f"Acc: {avg_train['accuracy']*100:.2f}%")
        print(f"VAL   - Total: {avg_val['L_total']:.4f} | "
              f"Contrastive: {avg_val['L_contrastive']:.4f} | "
              f"Recon: {avg_val['L_rec']:.4f} | "
              f"Acc: {avg_val['accuracy']*100:.2f}%")

        # Save best model
        if avg_val['accuracy'] > best_val_acc:
            best_val_acc = avg_val['accuracy']
            model.save_weights("/content/BIENCODER_best_model.weights.h5")
            print(f"✅ Best model saved! (Val Acc: {best_val_acc*100:.2f}%)")

    print("\n✅ Training completed!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    
    return model

# ============================================================
# 7. EVALUATION: TEMPORAL ORDER PREDICTION
# ============================================================
def extract_features_for_TOP(model, df, max_clips=500):
    """Extract features for Temporal Order Prediction"""
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

            # Extract features using encoder
            z_norm, z_full = model.encoder(A, training=False)
            z_full = z_full.numpy()[0]  # (T, D)

            # Compute delta features
            dz = z_full[1:] - z_full[:-1]
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
            perm = np.random.permutation(len(z_full))
            z_shuf = z_full[perm]
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
    """Temporal Order Prediction evaluation"""
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
    print("🚀 BI-ENCODER CONTRASTIVE SYSTEM - FULL PIPELINE")
    print("="*60)

    # 1. Create metadata
    df = create_metadata()

    # 2. SPLIT DATA (70/15/15)
    df_train, df_val, df_test = split_data(df, 0.70, 0.15, 0.15)

    # 3. Create pairs for TRAIN and VAL
    pair_df_train = create_pairs_metadata(df_train)
    pair_df_val = create_pairs_metadata(df_val)

    # 4. Build datasets
    dataset_train = build_pair_dataset(pair_df_train, batch_size=BATCH_SIZE)
    dataset_val = build_pair_dataset(pair_df_val, batch_size=BATCH_SIZE)

    # 5. Build model
    model = BiEncoderModel(latent_dim=LATENT_DIM)

    # 6. Train with validation
    model = train_biencoder_model(
        model,
        dataset_train,
        dataset_val,
        epochs=EPOCHS,
        steps_per_epoch=200,
        val_steps=50
    )

    # 7. Load best model
    print("\n📥 Loading best model from validation...")
    model.load_weights("/content/BIENCODER_best_model.weights.h5")

    # 8. Save final models
    model.encoder.save("/content/BIENCODER_encoder.keras")
    model.decoder.save("/content/BIENCODER_decoder.keras")
    print("✅ Final models saved!")

    # 9. Evaluate TOP on TEST SET
    print("\n📊 Extracting features from TEST set...")
    X_test, Y_test, centres_test = extract_features_for_TOP(
        model, df_test, max_clips=500
    )

    if len(X_test) > 0:
        acc, ci_low, ci_high, p_val = evaluate_TOP(X_test, Y_test)

        # 10. Summary
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED - FINAL RESULTS (TEST SET)")
        print("="*60)
        print(f"Training epochs: {EPOCHS}")
        print(f"Latent dimension: {LATENT_DIM}")
        print(f"Number of centres: {NUM_CENTRES}")
        print(f"Train/Val/Test split: 70/15/15")
        print(f"\nTOP Accuracy: {acc*100:.2f}% [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
        print(f"p-value: {p_val:.5f}")
        print("="*60)
    else:
        print("⚠️ No features extracted. Check paths and data integrity.")