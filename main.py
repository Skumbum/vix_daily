#main.py

from stats_vix import VixStats

def main():
    vix_stats = VixStats()
    vix_stats.download_data()

    print(f"Row Count: {vix_stats.get_row_count}")
    print(f"Current Vix: {vix_stats.get_current_vix}")
    print(f"Mean: {vix_stats.get_mean}")
    print(f"Median: {vix_stats.get_median}")
    print(f"Mode: {vix_stats.get_mode}")
    print(f"Std Dev: {vix_stats.get_std_dev}")
    print(f"Z-Score: {vix_stats.get_z_score}\n")
    print(f"Rolling 7 Day Mean: {vix_stats.get_rolling_mean7}")
    print(f"Rolling 30 Day Mean: {vix_stats.get_rolling_mean30}")
    print(f"RSI: {round(vix_stats.get_rsi.iloc[-1], 2)}")

if __name__ == "__main__":
    main()