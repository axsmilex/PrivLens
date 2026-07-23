# batch_infer_tflite.py  (robust to binary or 2-class outputs)
import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)  # [1,H,W,3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--labels", default="non_sensitive,sensitive")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--files", nargs="+", required=True)  # basenames like 9 21 30 ...
    ap.add_argument("--num-threads", type=int, default=4)
    args = ap.parse_args()

    class_names = [s.strip() for s in args.labels.split(",")]

    interpreter = tf.lite.Interpreter(model_path=args.model, num_threads=args.num_threads)
    interpreter.allocate_tensors()
    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    is_int8 = in_det["dtype"] in (np.uint8, np.int8)
    in_scale, in_zero = (1.0, 0)
    if "quantization" in in_det and in_det["quantization"][0] not in (0.0, None):
        in_scale, in_zero = in_det["quantization"]

    print(f"Model: {args.model}")
    print(f"Classes: {class_names}")
    print("-"*60)

    for base in args.files:
        candidates = [os.path.join(args.imgdir, f"{base}.jpg"),
                      os.path.join(args.imgdir, f"{base}.jpeg")]
        img_path = next((p for p in candidates if os.path.exists(p)), None)
        if not img_path:
            print(f"[MISS] {base}: not found (.jpg/.jpeg)")
            continue

        x = load_image(img_path, args.size)  # float32 [0..1]
        xin = x
        if is_int8:
            xin = (x / in_scale + in_zero).round().astype(in_det["dtype"])

        interpreter.set_tensor(in_det["index"], xin)
        interpreter.invoke()
        y = interpreter.get_tensor(out_det["index"]).squeeze()

        # Dequantize output if needed
        if "quantization" in out_det and out_det["quantization"][0] not in (0.0, None):
            s, z = out_det["quantization"]
            y = (y.astype(np.float32) - z) * s

        # Normalize to class probabilities
        if np.ndim(y) == 0 or (np.ndim(y) == 1 and y.shape[0] == 1):
            # Binary head: y is prob of "sensitive"
            p = float(np.array(y).reshape(()))
            p = max(0.0, min(1.0, p))  # clamp
            probs = np.array([1.0 - p, p], dtype=np.float32)
        elif np.ndim(y) == 1 and y.shape[0] == 2:
            # Two logits or two probs
            y = y.astype(np.float32)
            # If not already probabilities, softmax is safe
            e = np.exp(y - np.max(y))
            probs = e / np.sum(e)
        else:
            # Fallback: take argmax over whatever came out
            y = np.array(y, dtype=np.float32).ravel()
            e = np.exp(y - np.max(y))
            probs = e / np.sum(e)
            if probs.size < 2:
                probs = np.array([1.0 - float(probs[0]), float(probs[0])], dtype=np.float32)

        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        pred_score = float(probs[pred_idx])
        is_sensitive = (pred_label == "sensitive") and (pred_score >= args.threshold)

        print(f"[{base}] {os.path.basename(img_path)}")
        print(f"  probs={probs.tolist()}")
        print(f"  pred={pred_label}  score={pred_score:.3f}  sensitive@{args.threshold}={is_sensitive}")
        print("-"*60)

if __name__ == "__main__":
    main()
