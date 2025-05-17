# Scritp to compare times between two different implementations of the same algorithm

import class_SeamCarver as sc
import numpy as np

import time

# --- Version NON parallélisée ---
start = time.time()
e1 = compute_hog_custom(gray_image)
print(f"⏱️ Version non parallélisée : {time.time() - start:.3f} s")

# --- Version PARALLÉLISÉE ---
start = time.time()
e2 = compute_hog_custom_parallel(gray_image)
print(f"⚡ Version parallélisée     : {time.time() - start:.3f} s")
