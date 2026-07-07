"""Step 2: Load saved PV data and plot curves for 8Small and 24All"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='sans-serif'; matplotlib.rcParams['axes.unicode_minus']=False; matplotlib.rcParams['font.size']=12

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pv_data")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots")

# Load dates
df = pd.read_csv(f"{DATA_DIR}/bert_pred_24stocks_delta_G.csv"); df["Date"] = pd.to_datetime(df["Date"])
df_c = df.pivot_table(index='Date',columns='Stock_symbol',values='close').ffill().bfill()
dates = df_c.index
VE = dates.get_indexer([pd.to_datetime('2019-12-31')],method='pad')[0]
DAYS_ = dates.get_indexer([pd.to_datetime('2021-12-31')],method='pad')[0]
TEST_R = range(VE, DAYS_)

# ============ COLORS & STYLES ============
COLORS = {
    'EW':     '#000000',   # black
    'SB-P':   '#FF6D00',   # bright orange
    'S-BERT': '#8E24AA',   # bright purple
    'SBA':    '#00ACC1',   # bright teal
    'SBC':    '#43A047',   # bright green
    'SBCA':   '#E53935',   # bright red
    'PPO':    '#1E88E5',   # bright blue
}

STYLES = {
    'EW':     (1.0, '-'),
    'SB-P':   (1.0, '-'),
    'S-BERT': (1.0, '-'),
    'SBA':    (1.0, '-'),
    'SBC':    (1.0, '-'),
    'SBCA':   (1.0, '-'),
    'PPO':    (1.0, '-'),
}

LABELS = {
    'EW': 'Equal Weight', 'SB-P': 'SB-P (no text, no AC)', 'S-BERT': 'S-BERT (text, no AC)',
    'SBA': 'SBA (text + AC)', 'SBC': 'SBC (CM, no AC)', 'SBCA': 'SBCA (CM + AC)', 'PPO': 'PPO'
}

ORDER = ['EW','SB-P','S-BERT','SBC','SBA','PPO','SBCA']

for group_name in ['8Small', '24All']:
    fig, ax = plt.subplots(figsize=(14, 6))

    xd = dates[list(TEST_R)]
    max_len = 99999

    for mname in ORDER:
        fname = f"{SAVE_DIR}/pv_{group_name}_{mname}.npy"
        pv = np.load(fname)
        ml = min(len(xd), len(pv)-1)
        lw, ls = STYLES[mname]
        ax.plot(xd[:ml], pv[1:ml+1], color=COLORS[mname], linewidth=lw,
                linestyle=ls, label=LABELS[mname], alpha=1.0)

    # COVID crash shading + recovery line
    all_pvs = [np.load(f"{SAVE_DIR}/pv_{group_name}_{m}.npy") for m in ORDER]
    ymax = max(p[1:min(len(xd)+1, len(p))].max() for p in all_pvs)
    ax.set_ylim(0.5, ymax * 1.1)
    ax.axvspan(pd.to_datetime('2020-02-19'), pd.to_datetime('2020-03-23'),
               alpha=0.15, color='gray', label='COVID Crash')
    ax.axvline(x=pd.to_datetime('2020-07-01'), color='black', linestyle='--',
               linewidth=1.2, alpha=0.5, label='Recovery')

    ax.set_title(f'{group_name} Stocks: Portfolio Value (2020--2021)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Portfolio Value (initial = 1.0)')
    ax.legend(fontsize=10, ncol=4, loc='upper left')
    ax.grid(alpha=0.25)
    for l in ax.get_xticklabels(): l.set_rotation(30)
    plt.tight_layout(pad=1.5)
    plt.savefig(f'{OUT_DIR}/portfolio_value_{group_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved portfolio_value_{group_name}.png")

print("Done! If you want to change colors/styles, edit COLORS/STYLES above and re-run.")
