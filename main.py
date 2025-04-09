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
    print(f"Z-Score: {vix_stats.get_z_score}")

if __name__ == "__main__":
    main()