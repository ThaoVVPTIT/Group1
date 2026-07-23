import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. CẤU HÌNH
# ============================================================

SEED = 42
BATCH_SIZE = 128
EPOCHS = 20

NUM_CLASSES = 47
INPUT_SHAPE = (28, 28, 1)

# EMNIST Balanced:
# 10 chữ số + 37 lớp chữ cái đã gộp các ký tự hoa/thường dễ nhầm = 47 lớp
DATASET_NAME = "emnist/balanced"

OUTPUT_DIR = "lenet5_emnist_3x3_output"
MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "lenet5_emnist_3x3.keras",
)
BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_lenet5_emnist_3x3.keras",
)

FPGA_EXPORT_DIR = os.path.join(OUTPUT_DIR, "fpga_export")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FPGA_EXPORT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("TensorFlow Datasets version:", tfds.__version__)


# ============================================================
# 2. ĐỌC BỘ DỮ LIỆU EMNIST BALANCED
# ============================================================

(train_raw, test_raw), dataset_info = tfds.load(
    DATASET_NAME,
    split=["train", "test"],
    as_supervised=True,
    with_info=True,
    shuffle_files=True,
)

train_size = dataset_info.splits["train"].num_examples
test_size = dataset_info.splits["test"].num_examples

BALANCED_ASCII_CODES = (
    list(range(ord("0"), ord("9") + 1))
    + list(range(ord("A"), ord("Z") + 1))
    + [
        ord("a"),
        ord("b"),
        ord("d"),
        ord("e"),
        ord("f"),
        ord("g"),
        ord("h"),
        ord("n"),
        ord("q"),
        ord("r"),
        ord("t"),
    ]
)

class_names = [
    chr(ascii_code)
    for ascii_code in BALANCED_ASCII_CODES
]

if len(class_names) != NUM_CLASSES:
    raise RuntimeError(
        f"Số tên lớp nhận được là {len(class_names)}, "
        f"nhưng mô hình yêu cầu {NUM_CLASSES} lớp."
    )

print("\nThông tin EMNIST Balanced:")
print("Số ảnh train:", train_size)
print("Số ảnh test :", test_size)
print("Số lớp      :", len(class_names))
print("Các lớp     :", class_names)


# ============================================================
# 3. HÀM TIỀN XỬ LÝ
# ============================================================

def correct_emnist_orientation(image):
    return tf.transpose(image, perm=[1, 0, 2])


def preprocess_image(image, label):
    image = correct_emnist_orientation(image)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return image, label


AUTOTUNE = tf.data.AUTOTUNE

test_dataset = (
    test_raw
    .map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


# ============================================================
# 6. XÂY DỰNG MÔ HÌNH LENET-5
# ============================================================

def create_lenet5():
    model = Sequential([
        Input(shape=INPUT_SHAPE),

        Conv2D(
            filters=6,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
            name="conv1",
        ),

        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name="pool1",
        ),

        Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
            name="conv2",
        ),

        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name="pool2",
        ),

        Flatten(name="flatten"),

        Dense(
            units=120,
            activation="relu",
            name="fc1",
        ),

        Dense(
            units=84,
            activation="relu",
            name="fc2",
        ),

        Dense(
            units=NUM_CLASSES,
            activation="softmax",
            name="output",
        ),
    ])
    return model


model = create_lenet5()
model.summary()


# ============================================================
# 7. BIÊN DỊCH MÔ HÌNH
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# 8. CALLBACK
# ============================================================

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),

    ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
]


# ============================================================
# 9. HUẤN LUYỆN MÔ HÌNH
# ============================================================

train_fit_raw, validation_raw = tfds.load(
    DATASET_NAME,
    split=["train[:90%]", "train[90%:]"],
    as_supervised=True,
    shuffle_files=True,
)

train_fit_dataset = (
    train_fit_raw
    .map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )
    .shuffle(
        buffer_size=20000,
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

validation_dataset = (
    validation_raw
    .map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

history = model.fit(
    train_fit_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# 9b. VẼ BIỂU ĐỒ QUÁ TRÌNH HUẤN LUYỆN
# ============================================================

epochs_ran = np.arange(1, len(history.history["accuracy"]) + 1)

# Biểu đồ Accuracy
plt.figure(figsize=(9, 6))
plt.plot(
    epochs_ran,
    history.history["accuracy"],
    marker="o",
    label="Training Accuracy",
)
plt.plot(
    epochs_ran,
    history.history["val_accuracy"],
    marker="o",
    label="Validation Accuracy",
)
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs_ran)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

ACCURACY_CURVE_PATH = os.path.join(
    OUTPUT_DIR,
    "training_validation_accuracy.png",
)
plt.savefig(
    ACCURACY_CURVE_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Biểu đồ Loss
plt.figure(figsize=(9, 6))
plt.plot(
    epochs_ran,
    history.history["loss"],
    marker="o",
    label="Training Loss",
)
plt.plot(
    epochs_ran,
    history.history["val_loss"],
    marker="o",
    label="Validation Loss",
)
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs_ran)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

LOSS_CURVE_PATH = os.path.join(
    OUTPUT_DIR,
    "training_validation_loss.png",
)
plt.savefig(
    LOSS_CURVE_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("\nĐã lưu biểu đồ Accuracy tại:", ACCURACY_CURVE_PATH)
print("Đã lưu biểu đồ Loss tại:", LOSS_CURVE_PATH)


# ============================================================
# 10. LƯU MÔ HÌNH
# ============================================================

model.save(MODEL_PATH)
print("\nĐã lưu mô hình cuối tại:", MODEL_PATH)
print("Đã lưu mô hình tốt nhất tại:", BEST_MODEL_PATH)


# ============================================================
# 11. ĐÁNH GIÁ TRÊN TẬP KIỂM TRA
# ============================================================

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=2,
)

print(f"Test loss    : {test_loss:.6f}")
print(f"Test accuracy: {test_accuracy:.6f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# ============================================================
# 12. CONFUSION MATRIX VÀ CLASSIFICATION REPORT
# ============================================================

y_true = []
y_pred = []

for image_batch, label_batch in test_dataset:
    probabilities = model.predict(image_batch, verbose=0)
    predicted_batch = np.argmax(probabilities, axis=1)

    y_true.extend(label_batch.numpy().astype(np.int32).tolist())
    y_pred.extend(predicted_batch.astype(np.int32).tolist())

y_true = np.asarray(y_true, dtype=np.int32)
y_pred = np.asarray(y_pred, dtype=np.int32)

conf_matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES),
)

CONFUSION_MATRIX_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv",
)
np.savetxt(
    CONFUSION_MATRIX_CSV_PATH,
    conf_matrix,
    fmt="%d",
    delimiter=",",
)

plt.figure(figsize=(18, 16))
plt.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix - EMNIST Balanced")
plt.xlabel("Nhãn dự đoán")
plt.ylabel("Nhãn thực tế")
plt.colorbar()

tick_positions = np.arange(NUM_CLASSES)
plt.xticks(tick_positions, class_names, rotation=90, fontsize=7)
plt.yticks(tick_positions, class_names, fontsize=7)
plt.tight_layout()

CONFUSION_MATRIX_PNG_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png",
)
plt.savefig(
    CONFUSION_MATRIX_PNG_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close()

classification_text = classification_report(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=class_names,
    digits=4,
    zero_division=0,
)

CLASSIFICATION_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt",
)
with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as file:
    file.write(classification_text)

print("\nClassification report:")
print(classification_text)
print("Đã lưu confusion matrix CSV tại:", CONFUSION_MATRIX_CSV_PATH)
print("Đã lưu confusion matrix PNG tại:", CONFUSION_MATRIX_PNG_PATH)
print("Đã lưu classification report tại:", CLASSIFICATION_REPORT_PATH)


# ============================================================
# 22b. CALIBRATE SCALE ĐẦU RA THỰC TẾ CỦA TỪNG LAYER (post-ReLU)
# ============================================================

CALIB_NUM_IMAGES = 1000

calib_model = tf.keras.Model(
    inputs=model.inputs,    
    outputs=[
        model.get_layer("conv1").output, 
        model.get_layer("conv2").output,
        model.get_layer("fc1").output,
        model.get_layer("fc2").output,
    ],
)

calib_images = []
for image, _ in train_raw.take(CALIB_NUM_IMAGES):
    img, _ = preprocess_image(image, 0)
    calib_images.append(img.numpy())
calib_images = np.stack(calib_images, axis=0)

conv1_act, conv2_act, fc1_act, fc2_act = calib_model.predict(
    calib_images, batch_size=128, verbose=0
)


def calc_scale(activation):
    """Scale = 127 / max(abs(activation)), tránh chia 0."""
    return 127.0 / max(np.max(np.abs(activation)), 1e-8)


S_in_conv1 = 127.0  
S_out_conv1 = calc_scale(conv1_act)
S_out_conv2 = calc_scale(conv2_act)
S_out_fc1 = calc_scale(fc1_act)
S_out_fc2 = calc_scale(fc2_act)

print("\n--- SCALE CALIBRATE TỪNG LAYER (post-ReLU, thay cho shift=8 cố định) ---")
print(f"S_in_conv1 : {S_in_conv1:.4f}")
print(f"S_out_conv1: {S_out_conv1:.4f}")
print(f"S_out_conv2: {S_out_conv2:.4f}")
print(f"S_out_fc1  : {S_out_fc1:.4f}")
print(f"S_out_fc2  : {S_out_fc2:.4f}")


# ============================================================
# 23. LƯỢNG TỬ HÓA PER-CHANNEL (WEIGHT) + MULTIPLY-SHIFT (REQUANT)
# ============================================================

print("\n--- ĐANG XUẤT TRỌNG SỐ PER-CHANNEL CHO RTL ---")
WEIGHTS_EXPORT_DIR = os.path.join(FPGA_EXPORT_DIR, "weights_hex")
os.makedirs(WEIGHTS_EXPORT_DIR, exist_ok=True)


def quantize_multiplier(real_multiplier, mantissa_bits=15):
    if real_multiplier <= 0:
        return 0, 0

    m = real_multiplier
    shift = 0
    while m >= 1.0:
        m /= 2.0
        shift -= 1
    while m < 0.5:
        m *= 2.0
        shift += 1

    mantissa = int(round(m * (1 << mantissa_bits)))
    if mantissa == (1 << mantissa_bits):
        mantissa //= 2
        shift -= 1

    total_shift = shift + mantissa_bits
    return mantissa, total_shift


W_c1, b_c1 = model.get_layer("conv1").get_weights()
W_c2, b_c2 = model.get_layer("conv2").get_weights()
W_f1, b_f1 = model.get_layer("fc1").get_weights()
W_f2, b_f2 = model.get_layer("fc2").get_weights()
W_f3, b_f3 = model.get_layer("output").get_weights()

# 1. CHUYỂN VỊ MA TRẬN TỪ KERAS (NHWC) SANG VERILOG (NCHW)
W_c1 = np.transpose(W_c1, (3, 2, 0, 1))  # (6, 1, 3, 3)
W_c2 = np.transpose(W_c2, (3, 2, 0, 1))  # (16, 6, 3, 3)

# Lớp FC1: Khôi phục HWC -> Chuyển thành CHW -> Dẹt lại 400
W_f1 = np.reshape(W_f1, (5, 5, 16, 120))
W_f1 = np.transpose(W_f1, (2, 0, 1, 3))
W_f1 = np.reshape(W_f1, (400, 120))
W_f1 = np.transpose(W_f1, (1, 0))  # (120, 400)

W_f2 = np.transpose(W_f2, (1, 0))  # (84, 120)
W_f3 = np.transpose(W_f3, (1, 0))  # (47, 84)


def per_channel_scale(W):
    """Scale riêng cho từng kênh output = axis 0 của W (đã transpose)."""
    axes = tuple(range(1, W.ndim))
    max_abs = np.maximum(np.max(np.abs(W), axis=axes), 1e-8)
    return 127.0 / max_abs


def quantize_weights_per_channel(W, scale_per_channel):
    shape = [-1] + [1] * (W.ndim - 1)
    scale = scale_per_channel.reshape(shape)
    return np.clip(np.round(W * scale), -128, 127).astype(np.int64)


# ---- Per-channel weight scale ----
c1_w_scale = per_channel_scale(W_c1)  # (6,)
c2_w_scale = per_channel_scale(W_c2)  # (16,)
f1_w_scale = per_channel_scale(W_f1)  # (120,)
f2_w_scale = per_channel_scale(W_f2)  # (84,)
f3_s = 127.0 / max(np.max(np.abs(W_f3)), 1e-8)  # FC3: per-tensor (giữ nguyên)

# ---- Quantize weights ----
c1_w_int = quantize_weights_per_channel(W_c1, c1_w_scale)
c2_w_int = quantize_weights_per_channel(W_c2, c2_w_scale)
f1_w_int = quantize_weights_per_channel(W_f1, f1_w_scale)
f2_w_int = quantize_weights_per_channel(W_f2, f2_w_scale)
f3_w_int = np.clip(np.round(W_f3 * f3_s), -128, 127).astype(np.int64)

# ---- Quantize bias: scale = S_in_layer * S_w[c] (per-channel) ----
c1_b_int = np.round(b_c1 * S_in_conv1 * c1_w_scale).astype(np.int64)
c2_b_int = np.round(b_c2 * S_out_conv1 * c2_w_scale).astype(np.int64)
f1_b_int = np.round(b_f1 * S_out_conv2 * f1_w_scale).astype(np.int64)
f2_b_int = np.round(b_f2 * S_out_fc1 * f2_w_scale).astype(np.int64)
f3_b_int = np.round(b_f3 * S_out_fc2 * f3_s).astype(np.int64)  # FC3: input scale = S_out_fc2

# ---- Multiplier + shift per-channel (requant) ----
def compute_mult_shift_array(S_in_layer, S_out_layer, w_scale_array):
    n = len(w_scale_array)
    mult = np.zeros(n, dtype=np.int64)
    shift = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        real_m = S_out_layer / (S_in_layer * w_scale_array[idx])
        m, s = quantize_multiplier(real_m)
        mult[idx] = m
        shift[idx] = s
    return mult, shift


c1_mult, c1_shift = compute_mult_shift_array(S_in_conv1, S_out_conv1, c1_w_scale)
c2_mult, c2_shift = compute_mult_shift_array(S_out_conv1, S_out_conv2, c2_w_scale)
f1_mult, f1_shift = compute_mult_shift_array(S_out_conv2, S_out_fc1, f1_w_scale)
f2_mult, f2_shift = compute_mult_shift_array(S_out_fc1, S_out_fc2, f2_w_scale)
# FC3: không cần multiplier/shift - giữ raw accumulator cho argmax

print("\nVí dụ multiplier/shift kênh 0 mỗi layer (kiểm tra sơ bộ):")
print(f"conv1[0]: mult={c1_mult[0]}, shift={c1_shift[0]}")
print(f"conv2[0]: mult={c2_mult[0]}, shift={c2_shift[0]}")
print(f"fc1[0]  : mult={f1_mult[0]}, shift={f1_shift[0]}")
print(f"fc2[0]  : mult={f2_mult[0]}, shift={f2_shift[0]}")

print("\nPhạm vi shift từng layer (để chọn độ rộng thanh ghi RTL cho an toàn):")
print(f"conv1 shift: min={c1_shift.min()}, max={c1_shift.max()}")
print(f"conv2 shift: min={c2_shift.min()}, max={c2_shift.max()}")
print(f"fc1   shift: min={f1_shift.min()}, max={f1_shift.max()}")
print(f"fc2   shift: min={f2_shift.min()}, max={f2_shift.max()}")


# ---- Ghi file hex ----
def write_hex(name, values, n_bits):
    mask = (1 << n_bits) - 1
    hex_digits = (n_bits + 3) // 4
    lines = [f"{(int(v) & mask):0{hex_digits}x}" for v in np.asarray(values).flatten()]
    with open(os.path.join(WEIGHTS_EXPORT_DIR, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    

# Trọng số (8-bit)
write_hex("conv1_kernel.hex", c1_w_int, 8)
write_hex("conv2_kernel.hex", c2_w_int, 8)
write_hex("fc1_kernel.hex", f1_w_int, 8)
write_hex("fc2_kernel.hex", f2_w_int, 8)
write_hex("fc3_kernel.hex", f3_w_int, 8)

# Bias (32-bit)
write_hex("conv1_bias.hex", c1_b_int, 32)
write_hex("conv2_bias.hex", c2_b_int, 32)
write_hex("fc1_bias.hex", f1_b_int, 32)
write_hex("fc2_bias.hex", f2_b_int, 32)
write_hex("fc3_bias.hex", f3_b_int, 32)

# Multiplier (16-bit) + Shift (8-bit)
write_hex("conv1_multiplier.hex", c1_mult, 16)
write_hex("conv1_shift.hex", c1_shift, 8)
write_hex("conv2_multiplier.hex", c2_mult, 16)
write_hex("conv2_shift.hex", c2_shift, 8)
write_hex("fc1_multiplier.hex", f1_mult, 16)
write_hex("fc1_shift.hex", f1_shift, 8)
write_hex("fc2_multiplier.hex", f2_mult, 16)
write_hex("fc2_shift.hex", f2_shift, 8)

print("\nĐã xuất trọng số + bias + multiplier/shift per-channel (kiểu TFLite) cho RTL.")


# ============================================================
# 25. BÁO CÁO TÀI NGUYÊN ƯỚC TÍNH (RESOURCE BUDGET)
# ============================================================

resource_report_lines = ["layer_name\tparams\tMACs_estimate"]
total_params = 0
total_macs = 0

conv_output_shapes = {
    "conv1": (26, 26, 6),
    "conv2": (11, 11, 16),
}

for layer in model.layers:
    layer_weights = layer.get_weights()
    if not layer_weights:
        continue
    kernel = layer_weights[0]
    num_params = int(np.prod(kernel.shape))
    total_params += num_params

    if layer.name in conv_output_shapes:
        out_h, out_w, out_c = conv_output_shapes[layer.name]
        kh, kw, in_c, out_c_k = kernel.shape
        macs = out_h * out_w * out_c * kh * kw * in_c
    else:
        macs = num_params  # Dense: 1 MAC / trọng số / ảnh

    total_macs += macs
    resource_report_lines.append(f"{layer.name}\t{num_params}\t{macs}")

resource_report_lines.append(f"TOTAL\t{total_params}\t{total_macs}")

RESOURCE_REPORT_PATH = os.path.join(
    FPGA_EXPORT_DIR,
    "resource_budget_estimate.tsv",
)
with open(RESOURCE_REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("\n".join(resource_report_lines))

print("\nĐã lưu ước tính tài nguyên (params/MACs) tại:", RESOURCE_REPORT_PATH)
print(f"Tổng số tham số : {total_params:,}")
print(f"Tổng số MAC/ảnh : {total_macs:,}")

print("\nHoàn thành huấn luyện và đánh giá LeNet-5.")
print("Bộ dữ liệu: EMNIST Balanced, 47 lớp.")
print("Lượng tử hóa: PER-CHANNEL multiply-shift (kiểu TFLite), thay cho shift=8 cố định.")
print("Các kết quả được lưu trong thư mục:", OUTPUT_DIR)
print("Các sản phẩm phục vụ FPGA (Nhóm 1) lưu tại:", FPGA_EXPORT_DIR)