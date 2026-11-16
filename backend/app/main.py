from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

HEIGHT = 128
WIDTH = 384
CHANNELS = 3

BACKGROUND_SIGMA = 25
USE_CLAHE = False
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID = (8, 8)
CROP_MARGIN = 8

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}

@tf.keras.utils.register_keras_serializable(package="writer")
class SnakeToSquare(tf.keras.layers.Layer):
    def call(self, x):
        t = HEIGHT

        t1 = x[:, :, :t, :]
        t2 = x[:, :, t:2 * t, :]
        t3 = x[:, :, 2 * t:3 * t, :]

        blank = tf.ones_like(t1) * 255.0

        top = tf.concat([t1, t2], axis=2)
        bottom = tf.concat([blank, t3], axis=2)

        return tf.concat([top, bottom], axis=1)


def load_models():
    models = []

    for fold in range(1, 6):
        model_path = MODELS_DIR / f"dino_fold_{fold}" / "final_model.keras"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        models.append(model)

        print(f"Loaded DINO fold {fold}: {model.input_shape}")

    return models


models = load_models()

def _load_gray(image):
    if isinstance(image, (str, Path)):
        arr = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Gagal membaca gambar: {image}")
    elif isinstance(image, bytes):
        arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("Bytes gambar tidak valid.")
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        arr = image.copy()
    else:
        raise TypeError("Input harus berupa path, bytes, PIL Image, atau NumPy array.")

    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # Alpha compositing ke putih sebelum grayscale.
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # Array NumPy diasumsikan BGR (seperti OpenCV).
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Dimensi gambar tidak didukung: {arr.shape}")

    return np.clip(gray, 0, 255).astype(np.uint8)


def ink_centric_enhancement(gray):
    h, w = gray.shape
    sigma = max(10, min(BACKGROUND_SIGMA, min(h, w) / 3))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    background = np.maximum(background, 1)
    normalized = cv2.divide(gray, background, scale=255)
    if USE_CLAHE:
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID
        )
        normalized = clahe.apply(normalized)
    return normalized


def get_ink_bbox(gray, margin=CROP_MARGIN):
    # Otsu hanya dipakai untuk menentukan bbox, bukan mengganti citra grayscale.
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # Hilangkan komponen noise sangat kecil tanpa threshold agresif.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= 2:
            clean[labels == i] = 255

    points = cv2.findNonZero(clean)
    if points is None:
        return 0, 0, gray.shape[1], gray.shape[0], mask

    x, y, w, h = cv2.boundingRect(points)
    H, W = gray.shape
    x1, y1 = max(0, x-margin), max(0, y-margin)
    x2, y2 = min(W, x+w+margin), min(H, y+h+margin)
    return x1, y1, x2, y2, mask


def smart_line_stack(
    cropped,
    target_height=HEIGHT,
    target_width=WIDTH,
    min_aspect_ratio=2.5,
    max_rows=4,
    row_gap_ratio=0.18,
    fill_value=FILL_VALUE
):
    """
    Memotong baris tulisan tangan yang sangat lebar di celah spasi antar-kata/huruf
    dan menumpuknya secara vertikal agar resolusi stroke tetap besar saat di-resize ke persegi.
    """
    h, w = cropped.shape[:2]
    if h < 1 or w < 1 or (w / h) < min_aspect_ratio:
        return cropped

    # Mask tinta sementara untuk mencari lembah spasi (tidak mengubah nilai piksel asli)
    _, mask = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_ink = (mask > 0).sum(axis=0).astype(np.float32)

    # Haluskan profil proyeksi vertikal agar celah kecil di dalam huruf tidak terdeteksi sebagai spasi
    k = max(3, int(round(h * 0.12)) | 1)
    smooth_ink = np.convolve(col_ink, np.ones(k) / k, mode="same")
    ink_threshold = max(1.0, 0.02 * h)

    best_chunks = None
    best_scale = min(target_width / w, target_height / h)
    gap = max(4, int(round(h * row_gap_ratio)))

    for n_rows in range(2, max_rows + 1):
        # Jangan tumpuk terlalu banyak baris jika tinggi sudah mendekati lebar
        if (w / n_rows) < (h * 0.85):
            break

        split_points = [0]
        valid_split = True

        for r in range(1, n_rows):
            ideal_x = int(round(w * r / n_rows))
            search_radius = max(10, int(w / (n_rows * 2.5)))
            left = max(split_points[-1] + max(8, h // 3), ideal_x - search_radius)
            right = min(w - max(8, h // 3), ideal_x + search_radius)

            if left >= right:
                valid_split = False
                break

            window = smooth_ink[left:right]
            # Cari area kosong/spasi yang paling dekat dengan titik bagi ideal
            zero_indices = np.where(window <= ink_threshold)[0]
            if len(zero_indices) > 0:
                candidates = left + zero_indices
                best_x = int(candidates[np.argmin(np.abs(candidates - ideal_x))])
            else:
                # Fallback: jangan potong jika tinta di titik terendah masih terlalu padat (>15% tinggi)
                min_idx = int(np.argmin(window))
                if window[min_idx] > 0.15 * h:
                    valid_split = False
                    break
                best_x = left + min_idx

            split_points.append(best_x)

        if not valid_split:
            continue

        split_points.append(w)

        # Potong dan trim whitespace horizontal berlebih di ujung tiap baris
        chunks = []
        for i in range(len(split_points) - 1):
            c_left, c_right = split_points[i], split_points[i + 1]
            sub_mask = mask[:, c_left:c_right]
            nz = np.where(sub_mask > 0)[1]
            if len(nz) > 0:
                pad = max(2, CROP_MARGIN // 2)
                t_left = max(c_left, c_left + int(nz.min()) - pad)
                t_right = min(c_right, c_left + int(nz.max()) + 1 + pad)
                chunks.append(cropped[:, t_left:t_right])
            else:
                chunks.append(cropped[:, c_left:c_right])

        stacked_w = max(c.shape[1] for c in chunks)
        stacked_h = sum(c.shape[0] for c in chunks) + gap * (len(chunks) - 1)
        scale = min(target_width / stacked_w, target_height / stacked_h)

        # Pilih jumlah baris yang menghasilkan skala stroke paling besar (minimal 15% lebih besar)
        if scale > best_scale * 1.15:
            best_scale = scale
            best_chunks = chunks

    if best_chunks is None:
        return cropped

    stacked_w = max(c.shape[1] for c in best_chunks)
    stacked_h = sum(c.shape[0] for c in best_chunks) + gap * (len(best_chunks) - 1)
    canvas = np.full((stacked_h, stacked_w), fill_value, dtype=np.uint8)

    curr_y = 0
    for chunk in best_chunks:
        ch, cw = chunk.shape[:2]
        # Rata kiri (natural untuk teks tulisan tangan)
        canvas[curr_y:curr_y + ch, 0:cw] = chunk
        curr_y += ch + gap

    return canvas


def resize_with_white_padding(gray, height=HEIGHT, width=WIDTH):
    h, w = gray.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("Gambar kosong setelah crop.")
    scale = min(width / w, height / h)
    new_w = max(1, min(width, int(round(w * scale))))
    new_h = max(1, min(height, int(round(h * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas


def preprocess_handwriting_image(image, height=HEIGHT, width=WIDTH, return_debug=False):
    original = _load_gray(image)
    enhanced = ink_centric_enhancement(original)
    x1, y1, x2, y2, mask = get_ink_bbox(enhanced)
    cropped = enhanced[y1:y2, x1:x2]
    resized = resize_with_white_padding(cropped, height, width)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    if return_debug:
        return rgb, {
            "original": original, "enhanced": enhanced, "ink_mask": mask,
            "cropped": cropped, "resized": resized, "final_rgb": rgb
        }
    return rgb
class PredictionValue(BaseModel):
    Label: int
    Probability: list[float]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": len(models)
    }


@app.post("/predict", response_model=PredictionValue)
async def predict(file: UploadFile = File(...)):
    extension = Path(file.filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and JPEG images are supported."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    try:
        processed = preprocess_handwriting_image(image_bytes)

        image_batch = np.expand_dims(
            processed.astype(np.float32),
            axis=0
        )

        fold_probabilities =np.stack([
            model.predict(image_batch, verbose=0)[0]
            for model in models
            ])

        probabilities = np.mean(
            fold_probabilities,
            axis=0
        )

        predicted_label = int(np.argmax(probabilities))

        return {
            "Label": predicted_label,
            "Probability": probabilities.astype(float).tolist()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
