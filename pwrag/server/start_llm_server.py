import subprocess
import sys

import hydra
from pwrag.args.args import AppConfig

@hydra.main(config_path="../config", config_name="dev_config", version_base=None)  # dev_config.yaml, config.yaml
def start_server(cfg: AppConfig) -> None:

    model_path = cfg.generator.model_path
    port = "8000"

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", port,
        "--gpu-memory-utilization", "0.95",
        "--max-model-len", "100000",
    ]

    print("Starting vLLM server...")
    subprocess.run(cmd)


if __name__ == "__main__":
    start_server()

    
