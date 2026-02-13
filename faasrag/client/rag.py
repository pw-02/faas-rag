
from dataclasses import dataclass
import argparse
import asyncio
import logging
import random
import time


def run_rag(args):
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse RAG benchmark configurations.")
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset path")
    parser.add_argument("--start-index", type=int, default=0, help="Start index of the workload")
    parser.add_argument( "--end-index", type=int, default=-1, help="End index of the workload")
    parser.add_argument("--shuffle", action="store_true", help="Random shuffle")
    parser.add_argument("--system-prompt", type=str, default="", help="System prompt")
    parser.add_argument("--separator", type=str, default="", help="Separator")
    parser.add_argument("--query-prompt", type=str, default="", help="Query prompt")

    parser.add_argument(
        "--prompt-build-method",
        type=str,
        required=True,
        help="Prompt build method",
        )
   
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name (ended with csv or txt) for the summary csv and txt",
    )

    parser.add_argument("--warmup", action="store_true", help="Whether to enable warmup")

    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help="The total running time in seconds",
    )


    parser.add_argument(
        "--step-interval", type=float, default=0.02, help="Step interval"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    run_rag(args)



if __name__ == "__main__":
    main()
