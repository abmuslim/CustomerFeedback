import requests
import json
import csv
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
INPUT_FILE = "feedback_input_text_unique_variations.json"
OUTPUT_CSV = "inference_latency_log.csv"
URL = "http://172.22.174.42:31818/feedback/analyse"

# === LOAD FEEDBACK DATA ===
with open(INPUT_FILE, "r") as file:
    feedback_data = json.load(file)
print(f"[INFO] Loaded {len(feedback_data)} feedback entries.")

# === INIT VARIABLES ===
inference_times = []
elapsed_times = []
start_time = time.time()

# === RUN INFERENCE ===
try:
    with open(OUTPUT_CSV, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['elapsed_time_sec', 'inference_time_ms'])

        for idx, feedback in enumerate(feedback_data):
            try:
                response = requests.post(URL, json=feedback, timeout=100)
                response.raise_for_status()
                result = response.json()

                latency = result['inference_time']
                elapsed = time.time() - start_time

                inference_times.append(latency)
                elapsed_times.append(elapsed)

                print(f"[{idx+1}] {elapsed:.2f}s | {latency:.2f} ms")
                writer.writerow([round(elapsed, 2), round(latency, 2)])

            except Exception as e:
                print(f"[ERROR] Failed at entry {idx+1}: {e}")
            time.sleep(0.5)

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Saving plots...")

# === PLOTTING ===
if inference_times:
    average_latency = sum(inference_times) / len(inference_times)

    # Line Plot - trend with cumulative average
    cumulative_avg = [sum(inference_times[:i+1]) / (i+1) for i in range(len(inference_times))]

    plt.figure(figsize=(14, 6))
    plt.plot(elapsed_times, inference_times, marker='o', linestyle='', alpha=0.4, label='Latency Samples')
    plt.plot(elapsed_times, cumulative_avg, color='red', linewidth=2, label='Cumulative Avg Latency')

    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Latency (ms)")
    plt.title("Inference Latency Over Time with Cumulative Average")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("latency_lineplot.png")
    print("[✓] Line plot saved as latency_lineplot.png")


    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(inference_times, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Inference Latency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("latency_histogram.png")
    print("[✓] Histogram saved as latency_histogram.png")

    # CDF
    sorted_latencies = np.sort(inference_times)
    cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_latencies, cdf, marker='.', color='green')
    plt.xlabel("Latency (ms)")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Inference Latency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("latency_cdf.png")
    print("[✓] CDF plot saved as latency_cdf.png")

    # Horizontal Box Plot
    plt.figure(figsize=(10, 2.5))
    plt.boxplot(inference_times, vert=False, patch_artist=True,
                boxprops=dict(facecolor="lightblue"))
    plt.xlabel("Latency (ms)")
    plt.title("Box Plot of Inference Latency")
    plt.tight_layout()
    plt.savefig("latency_boxplot.png")
    print("[✓] Box plot saved as latency_boxplot.png")

else:
    print("[!] No data recorded. Skipping plots.")

