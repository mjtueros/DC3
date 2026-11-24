from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
FILE_NUMBER = 7

matches = list((BASE_DIR / "res_horizon").glob(f"max_visible_zenith_{FILE_NUMBER}_*.csv"))

if len(matches) != 1:
    raise FileNotFoundError(
        f"Expected exactly one CSV for ID {FILE_NUMBER}, found: {matches}"
    )

CSV_FILE = matches[0]

df = pd.read_csv(CSV_FILE)

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)

for antenna_id, group in df.groupby("Antenna_ID"):
    group = group.sort_values("Azimuth")

    ax.plot(group["Azimuth"], group["Max_Zenith"], linewidth=1.2)

ax.set_xlabel("Azimuth (°)")
ax.set_ylabel("Maximum visible zenith (°)")
ax.set_title(f"Maximum visible zenith - File {FILE_NUMBER}")

ax.set_xlim(0, 360)
ax.set_ylim(85, 90)
ax.set_xticks(np.arange(0, 361, 45))
ax.set_yticks(np.arange(85, 90.1, 0.5))

ax.grid(True, linewidth=0.5, alpha=0.3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()