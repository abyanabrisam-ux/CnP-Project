import pandas as pd

def get_esports_data():
    esports_data = {
        'Player_ID': [f'P{i:02d}' for i in range(1, 31)],
        'Average_Survival_Time_Min': [
            8, 6, 7, 5, 9, 8,        # Aggressive Carries
            28, 26, 27, 29, 25, 26,  # Passive Survivors
            22, 24, 21, 25, 23, 22,  # Support / Utility
            4, 5, 3, 6, 5, 4,        # Novices / Low Performers
            15, 14, 16, 17, 13, 15   # Mid-tier All-Rounders
        ],
        'Average_Damage_Dealt': [
            1500, 1750, 1600, 1850, 1400, 1650, # Carries
            300, 250, 400, 200, 350, 280,       # Survivors
            600, 750, 500, 700, 650, 550,       # Supports
            150, 200, 100, 250, 180, 120,       # Novices
            800, 950, 850, 900, 750, 880        # All-Rounders
        ],
        'Kills_Per_Match': [
            8.2, 9.5, 7.8, 10.1, 6.9, 8.8,      # Carries
            0.8, 0.5, 1.2, 0.2, 1.0, 0.6,       # Survivors
            2.1, 2.5, 1.8, 3.0, 2.2, 1.9,       # Supports
            0.4, 0.6, 0.1, 0.8, 0.5, 0.2,       # Novices
            4.1, 4.8, 3.9, 4.5, 3.5, 4.2        # All-Rounders
        ],
        'Assists_Per_Match': [
            1.2, 0.8, 1.5, 0.5, 2.1, 1.1,       # Carries
            0.5, 1.1, 0.2, 0.4, 0.8, 0.7,       # Survivors
            7.5, 8.2, 6.8, 9.1, 7.9, 8.0,       # Supports
            0.5, 0.8, 0.2, 1.1, 0.4, 0.6,       # Novices
            3.2, 2.8, 3.5, 3.1, 2.9, 3.4        # All-Rounders
        ]
    }
    return pd.DataFrame(esports_data)

if __name__ == "__main__":
    df = get_esports_data()
    print(f"Dataset loaded successfully with {df.shape[0]} players and {df.shape[1]} features.")
