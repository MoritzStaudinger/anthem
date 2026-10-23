#!/usr/bin/env python3

import os
import time
import argparse
import subprocess
import atexit
import requests
from grobid_client import Client

def main():
    parser = argparse.ArgumentParser(description="Runs GROBID server and client to process PDFs in a folder.")
    parser.add_argument("input_path", help="Path to the input PDF folder.")
    parser.add_argument("output_path", help="Path to the output XML folder.")
    parser.add_argument("--workers", type=int, default=32, help="Number of workers for the client.")
    args = parser.parse_args()

    # The GROBID server executable is located at this path
    grobid_executable = "/opt/grobid/grobid-home/bin/grobid.sh"
    grobid_port = 8070
    grobid_host = "localhost"

    # -----------------
    # Start GROBID Service
    # -----------------
    print("Starting GROBID service in the background...")
    try:
        grobid_process = subprocess.Popen(
            [
                grobid_executable,
                "server",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        atexit.register(grobid_process.terminate)

    except FileNotFoundError:
        print("Error: The GROBID executable was not found. Check the path.")
        exit(1)

    # -----------------
    # Wait for the service to be ready
    # -----------------
    print("Waiting for GROBID service to become available...")
    max_retries = 60
    for i in range(max_retries):
        try:
            response = requests.get(f"http://{grobid_host}:{grobid_port}/api/isalive", timeout=5)
            if response.status_code == 200:
                print("GROBID service is up!")
                break
        except requests.exceptions.ConnectionError:
            pass  # Ignore connection errors and retry

        print(f"Attempt {i + 1}/{max_retries}: Service not ready. Retrying in 5 seconds...")
        time.sleep(5)
    else:
        print("Error: GROBID service did not become available after multiple retries.")
        exit(1)

    # -----------------
    # Run GROBID Python Client
    # -----------------
    print("Running GROBID client to process PDFs...")
    client = Client(grobid_host, grobid_port)

    try:
        client.process(
            service="processFulltextDocument",
            input_path=args.input_path,
            output=args.output_path,
            n=args.workers,
            consolidate_header=True,
            consolidate_citations=True
        )
    except Exception as e:
        print(f"An error occurred during client processing: {e}")
        exit(1)

    print("GROBID parsing completed.")


if __name__ == "__main__":
    main()