# infer_tflite.py
import argparse, os, numpy as np
from PIL import Image
import tensorflow as tf

def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [0,1]
    arr = np.expand_dims(arr, 0)  # [1,H,W,3]
    return arr

def prepare_input(arr, input_details):
    """Match model's expected dtype, including quantization."""
    dtype = input_details[0]["dtype"]
    if np.issubdtype(dtype, np.floating):
        # If model expects float, ensure float32
        return arr.astype(np.float32)
    else:
        # Quantized path (int8 or uint8)
        scale = input_details[0]["quantization_parameters"]["scales"]
        zp = input_details[0]["quantization_parameters"]["zero_points"]
        # Fallback for older TF fields
        if (scale is None or len(scale) == 0): 
            scale = input_details[0]["quantization"][0]
            zp = input_details[0]["quantization"][1]
        else:
            scale = float(scale[0]); zp = int(zp[0])

        q = arr / float(scale) + float(zp)
        if dtype == np.int8:
            q = np.clip(np.round(q), -128, 127).astype(np.int8)
        elif dtype == np.uint8:
            q = np.clip(np.round(q), 0, 255).astype(np.uint8)
        else:
            raise RuntimeError(f"Unsupported quantized dtype: {dtype}")
        return q

def dequantize_output(logits, output_details):
    dtype = output_details[0]["dtype"]
    if np.issubdtype(dtype, np.floating):
        return logits.astype(np.float32)
    # dequantize if needed
    scale = output_details[0]["quantization_parameters"]["scales"]
    zp = output_details[0]["quantization_parameters"]["zero_points"]
    if (scale is None or len(scale) == 0): 
        scale = output_details[0]["quantization"][0]
        zp = output_details[0]["quantization"][1]
    else:
        scale = float(scale[0]); zp = int(zp[0])
    return (logits.astype(np.float32) - float(zp)) * float(scale)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--labels", default="non_sensitive,sensitive")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--num-threads", type=int, default=4)
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(",")]
    assert len(labels) >= 2, "Expect at least 2 labels (non_sensitive,sensitive)."

    # Use tf.lite.Interpreter (works with your TensorFlow install)
    interpreter = tf.lite.Interpreter(model_path=args.model, num_threads=args.num_threads)
    interpreter.allocate_tensors()
    in_details  = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    arr = load_image(args.image, args.size)
    inp = prepare_input(arr, in_details)

    interpreter.set_tensor(in_details[0]["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(out_details[0]["index"])
    out = dequantize_output(out, out_details)[0]  # [C]

    # Softmax for 2-class logits if float; for quantized often already "logits-ish"
    # Safe softmax:
    exp = np.exp(out - np.max(out))
    probs = exp / np.sum(exp)

    pred_idx = int(np.argmax(probs))
    prob_sensitive = float(probs[1]) if len(probs) >= 2 else float(probs[-1])
    pred_label = labels[pred_idx]
    is_sensitive = (prob_sensitive >= args.threshold)

    print(f"Image: {args.image}")
    print(f"logits: {out.tolist()}")
    print(f"probs: {probs.tolist()}")
    print(f"pred: {pred_label}  prob_sensitive={prob_sensitive:.4f}  threshold={args.threshold}  -> is_sensitive={is_sensitive}")

if __name__ == "__main__":
    main()

# py infer_tflite.py   --model "C:/RIT/GA/BIV-Priv-Seg/artifacts/export/mobilenetv3_privacy_fp32.tflite"   --image "C:/RIT/GA/BIV-Priv-Seg/BIV-Priv-Seg/some_image.jpg"   --size 224   --labels "non_sensitive,sensitive"   --threshold 0.5   --num-threads 4   