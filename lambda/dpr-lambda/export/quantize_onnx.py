import os
from onnxruntime.quantization import quantize_dynamic, QuantType

IN_DIR = os.environ.get("IN_DIR", "app/models")

def main():
    in_path = os.path.join(IN_DIR, "model.onnx")
    out_path = os.path.join(IN_DIR, "model.int8.onnx")

    quantize_dynamic(
        model_input=in_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
    )
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
