import yfinance as yf
from scipy import stats

# Download VIX data
vix = yf.download("^VIX", period="max", auto_adjust=False)
close = vix["Close"]
row_count = len(close)

# Statistics calculation
current_vix = close.iloc[-1]
mean_vix = close.mean()
median_vix = close.median()
std_dev_vix = close.std()

# Calculate mode (rounded to nearest 0.5 for more meaningful results)
rounded_close = round(close * 2) / 2  # Round to nearest 0.5
mode_result = stats.mode(rounded_close)
mode_vix = mode_result.mode[0]
mode_count = mode_result.count[0]

# Calculate z-score for current VIX value
z_score = (current_vix - mean_vix) / std_dev_vix

# Convert all potential Series objects to float values
if hasattr(current_vix, 'item'):
    current_vix = current_vix.item()
if hasattr(mean_vix, 'item'):
    mean_vix = mean_vix.item()
if hasattr(median_vix, 'item'):
    median_vix = median_vix.item()
if hasattr(std_dev_vix, 'item'):
    std_dev_vix = std_dev_vix.item()
if hasattr(mode_vix, 'item'):
    mode_vix = mode_vix.item()
if hasattr(mode_count, 'item'):
    mode_count = mode_count.item()
if hasattr(z_score, 'item'):
    z_score = z_score.item()

print(f"Sample Size: {row_count}")
print(f"Current: {current_vix:.2f}")
print(f"Mean: {mean_vix:.2f}")
print(f"Median: {median_vix:.2f}")
print(f"Mode: {mode_vix:.2f} (occurs {mode_count} times)")
print(f"Std Dev: {std_dev_vix:.2f}")
print(f"Z-Score: {z_score:.2f}")

print("bye")
