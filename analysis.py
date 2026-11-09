"""
Smart Room Environment Monitoring - Data Analytics Project
Dataset: hourly temperature / humidity / light readings for 7 days (168 rows),
as a room sensor node (e.g. ESP32 + DHT22 + LDR) would log them.
Steps: 1) generate dataset  2) computation  3) visualization
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- 1. DATASET ----------------
rng = np.random.default_rng(42)          # fixed seed -> same data every run
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
rows = []
for d, day in enumerate(days):
    base = 29 + rng.normal(0, 0.8)       # day-to-day weather variation
    for hour in range(24):
        temp = base + 4 * np.sin((hour - 9) * np.pi / 12) + rng.normal(0, 0.6)
        hum = 75 - 1.8 * (temp - 29) + rng.normal(0, 2.5)
        light = max(0, 650 * np.sin((hour - 6) * np.pi / 12)) if 6 <= hour <= 18 else 0
        light += rng.normal(0, 15) if light > 0 else 0
        rows.append([day, hour, round(temp, 1), round(hum, 1), round(max(light, 0), 0)])

df = pd.DataFrame(rows, columns=["Day", "Hour", "Temperature_C", "Humidity_pct", "Light_lux"])

def comfort(t):
    if t < 28: return "Cool"
    if t <= 31: return "Comfortable"
    return "Hot"
df["Comfort"] = df["Temperature_C"].apply(comfort)
df.to_csv("data.csv", index=False)

# ---------------- 2. COMPUTATION ----------------
lines = []
def log(s=""):
    print(s); lines.append(s)

log("=== BASIC STATISTICS ===")
log(df[["Temperature_C", "Humidity_pct", "Light_lux"]].describe().round(2).to_string())
log()
log(f"Mean temperature   : {df.Temperature_C.mean():.2f} C")
log(f"Median temperature : {df.Temperature_C.median():.2f} C")
log(f"Std deviation      : {df.Temperature_C.std():.2f}")
log(f"Max / Min temp     : {df.Temperature_C.max()} / {df.Temperature_C.min()} C")
log(f"Hottest hour (avg) : {df.groupby('Hour').Temperature_C.mean().idxmax()}:00")
log(f"Coolest hour (avg) : {df.groupby('Hour').Temperature_C.mean().idxmin()}:00")
log()
log("=== CORRELATION ===")
log(df[["Temperature_C", "Humidity_pct", "Light_lux"]].corr().round(3).to_string())
log()
log("=== AVERAGE TEMPERATURE PER DAY ===")
daily = df.groupby("Day", sort=False).Temperature_C.mean().round(2)
log(daily.to_string())
log()
log("=== COMFORT LEVEL COUNT ===")
comfort_counts = df.Comfort.value_counts()
log(comfort_counts.to_string())
open("results.txt", "w").write("\n".join(lines))

# ---------------- 3. VISUALIZATION ----------------
plt.style.use("seaborn-v0_8-whitegrid")

# Bar chart: average temperature per day
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(daily.index, daily.values, color="#2a7fba")
ax.bar_label(bars, fmt="%.1f", padding=2)
ax.set_title("Average Room Temperature per Day"); ax.set_xlabel("Day"); ax.set_ylabel("Temperature (°C)")
ax.set_ylim(daily.min() - 2, daily.max() + 1)
fig.tight_layout(); fig.savefig("bar_chart.png", dpi=150); plt.close(fig)

# Line chart: hourly temperature, humidity (avg over week)
hourly = df.groupby("Hour")[["Temperature_C", "Humidity_pct"]].mean()
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(hourly.index, hourly.Temperature_C, marker="o", color="#d9531e", label="Temperature (°C)")
ax1.set_xlabel("Hour of day"); ax1.set_ylabel("Temperature (°C)", color="#d9531e")
ax2 = ax1.twinx()
ax2.plot(hourly.index, hourly.Humidity_pct, marker="s", color="#2a7fba", label="Humidity (%)")
ax2.set_ylabel("Humidity (%)", color="#2a7fba"); ax2.grid(False)
ax1.set_title("Average Temperature & Humidity by Hour of Day")
ax1.set_xticks(range(0, 24, 2))
fig.tight_layout(); fig.savefig("line_chart.png", dpi=150); plt.close(fig)

# Pie chart: comfort level share
order = ["Cool", "Comfortable", "Hot"]
vals = [comfort_counts.get(k, 0) for k in order]
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(vals, labels=order, autopct="%1.1f%%", startangle=90,
       colors=["#7fb8e6", "#6cc08b", "#e8745a"], wedgeprops={"edgecolor": "white"})
ax.set_title("Share of Readings by Comfort Level")
fig.tight_layout(); fig.savefig("pie_chart.png", dpi=150); plt.close(fig)

# Histogram: temperature distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df.Temperature_C, bins=12, color="#8e6bbf", edgecolor="black")
ax.axvline(df.Temperature_C.mean(), color="red", linestyle="--", label=f"Mean = {df.Temperature_C.mean():.1f} °C")
ax.set_title("Distribution of Temperature Readings"); ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Frequency")
ax.legend(); fig.tight_layout(); fig.savefig("histogram.png", dpi=150); plt.close(fig)

print("\nDone: data.csv, results.txt, bar_chart.png, line_chart.png, pie_chart.png, histogram.png")
