"""
Traffic Visualisations - Notebook-Friendly (Full)
=================================================
Generates all dashboards inline from data/traffic CSVs
"""

import os, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

plt.ioff()
sns.set_style("whitegrid")
sns.set_palette("husl")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_data():
    files = glob.glob("data/traffic/*.csv")
    if not files:
        raise FileNotFoundError("No traffic CSVs in data/traffic/")

    dfs = []
    for f in files:
        df_temp = pd.read_csv(f)
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    # Parse dates
    df["Report Date"] = pd.to_datetime(df["Report Date"], dayfirst=True, errors="coerce")
    df["Time Period Ending Parsed"] = pd.to_datetime(df["Time Period Ending"], dayfirst=True, errors="coerce")
    df["Hour"] = df["Time Period Ending Parsed"].dt.hour
    df["Weekday"] = df["Report Date"].dt.dayofweek
    df["Date"] = df["Report Date"].dt.date
    df["YearMonth"] = df["Report Date"].dt.to_period('M')
    df["Year"] = df["Report Date"].dt.year
    df["Month"] = df["Report Date"].dt.month
    df["IsWeekend"] = df["Weekday"].isin([5, 6])

    # Convert numeric columns
    size_cols = ["0 - 520 cm", "521  - 660 cm", "661 - 1160 cm", "1160+ cm"]
    for c in size_cols + ["Avg mph", "Total Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter valid data
    df = df[df["Total Volume"].notna() & (df["Total Volume"] > 0)].copy()

    # Site handling
    if "Site Name" in df.columns:
        df["Site"] = df["Site Name"]
    else:
        df["Site"] = "Unknown"

    df["Total Volume Clipped"] = df["Total Volume"].clip(upper=df["Total Volume"].quantile(0.95))

    return df

def plot_traffic_inline(show_plots=True, save_plots=False, out_root="plots/traffic"):
    """
    Reads CSVs from data/traffic, processes them, and generates all dashboards inline.

    Args:
        show_plots (bool): Display plots inline
        save_plots (bool): Also save plots to out_root
        out_root (str): Folder for saved plots
    """
    if save_plots:
        ensure_dir(out_root)

    df = load_data()
    sites = df["Site"].unique()

    # ============================================
    # DASHBOARD 1: SITE COMPARISON
    # ============================================
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Daily trends
    ax1 = fig.add_subplot(gs[0, :2])
    daily = df.groupby(['Date', 'Site'])['Total Volume'].sum().reset_index()
    for s in sites:
        d = daily[daily['Site'] == s].sort_values('Date')
        ax1.plot(d['Date'], d['Total Volume'], label=s, lw=2, alpha=0.8)
    ax1.set_title('Daily Traffic Volume by Site', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Vehicles')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Statistics table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    txt = ""
    for s in sites:
        sd = df[df['Site'] == s]
        total = sd['Total Volume'].sum()
        davg = sd.groupby('Date')['Total Volume'].sum().mean()
        peak = sd.groupby('Hour')['Total Volume'].mean().idxmax()
        spd_data = sd[sd['Avg mph'].notna()]
        spd = spd_data['Avg mph'].mean() if len(spd_data) > 0 else 0
        txt += f"{s}:\n  Total: {total:,.0f}\n  Daily: {davg:,.0f}\n  Peak: {peak:02d}:00\n"
        if spd > 0: txt += f"  Speed: {spd:.1f}mph\n"
        txt += "\n"
    ax2.text(0.05, 0.95, txt, fontsize=10, family='monospace', va='top', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', fc='lightblue', alpha=0.8))

    # 3. Hourly pattern
    ax3 = fig.add_subplot(gs[1, 0])
    hrly = df.groupby(['Hour', 'Site'])['Total Volume'].mean().reset_index()
    for s in sites:
        d = hrly[hrly['Site'] == s]
        ax3.plot(d['Hour'], d['Total Volume'], marker='o', label=s, lw=2.5, ms=6)
    ax3.set_title('Hourly Pattern', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Avg Vehicles')
    ax3.set_xticks(range(0,24,3))
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Weekday pattern
    ax4 = fig.add_subplot(gs[1, 1])
    wkd = df.groupby(['Weekday', 'Site'])['Total Volume'].mean().reset_index()
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for s in sites:
        d = wkd[wkd['Site'] == s]
        ax4.plot(d['Weekday'], d['Total Volume'], marker='s', label=s, lw=2.5, ms=7)
    ax4.set_title('Weekly Pattern', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Avg Vehicles')
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(days)
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    for s in sites:
        d = df[df['Site'] == s]['Total Volume']
        ax5.hist(d, bins=40, alpha=0.6, label=s, density=True, ec='black')
    ax5.set_title('Volume Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Vehicles')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. Monthly trends
    ax6 = fig.add_subplot(gs[2, 0])
    mnth = df.groupby(['YearMonth', 'Site'])['Total Volume'].sum().reset_index()
    for s in sites:
        d = mnth[mnth['Site'] == s]
        ax6.plot(range(len(d)), d['Total Volume'], marker='o', label=s, lw=2)
    ax6.set_title('Monthly Trends', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Month Index')
    ax6.set_ylabel('Total')
    ax6.legend()
    ax6.grid(alpha=0.3)

    # 7. Weekday vs Weekend
    ax7 = fig.add_subplot(gs[2, 1])
    wkcomp = df.groupby(['Site', 'IsWeekend'])['Total Volume'].mean().reset_index()
    x = np.arange(len(sites))
    w = 0.35
    wkday = [wkcomp[(wkcomp['Site']==s) & (~wkcomp['IsWeekend'])]['Total Volume'].values[0] for s in sites]
    wkend = [wkcomp[(wkcomp['Site']==s) & (wkcomp['IsWeekend'])]['Total Volume'].values[0] for s in sites]
    ax7.bar(x-w/2, wkday, w, label='Weekday', color='skyblue', ec='black')
    ax7.bar(x+w/2, wkend, w, label='Weekend', color='salmon', ec='black')
    ax7.set_title('Weekday vs Weekend', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Avg Volume')
    ax7.set_xticks(x)
    ax7.set_xticklabels(sites, fontsize=9)
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)

    # 8. Heatmap (first site)
    ax8 = fig.add_subplot(gs[2, 2])
    pivot = df[df['Site'] == sites[0]].pivot_table(
        index="Hour", columns="Weekday", values="Total Volume", aggfunc="mean"
    )
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax8, cbar_kws={'label':'Vehicles'}, linewidths=0.5, fmt='.0f', annot=False)
    ax8.set_title(f'Hour×Day\n({sites[0]})', fontsize=12, fontweight='bold')
    ax8.set_xticklabels(days)

    plt.suptitle('SITE COMPARISON DASHBOARD', fontsize=18, fontweight='bold', y=0.995)
    if show_plots:
        plt.show()
    if save_plots:
        plt.savefig(os.path.join(out_root, "01_site_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ============================================
    # DASHBOARD 2: SPEED ANALYSIS
    # ============================================
    dfs = df[df["Avg mph"].notna()].copy()
    if len(dfs) >= 100:
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, hspace=0.35, wspace=0.3)

        # 1. Speed trends
        ax1 = fig.add_subplot(gs[0, :2])
        for s in sites:
            d = dfs[dfs['Site'] == s].sort_values('Report Date')
            if len(d) > 50:
                roll = d.set_index('Report Date')['Avg mph'].rolling(50, min_periods=10).mean()
                ax1.plot(roll.index, roll.values, label=s, lw=2, alpha=0.8)
        ax1.set_title('Speed Trends (50-period rolling)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('mph')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Speed statistics table
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        txt = ""
        for s in sites:
            sp = dfs[dfs['Site'] == s]['Avg mph']
            if len(sp) > 0:
                txt += f"{s}:\n  Mean: {sp.mean():.1f}\n  Med: {sp.median():.1f}\n"
                txt += f"  Std: {sp.std():.1f}\n  Min: {sp.min():.1f}\n  Max: {sp.max():.1f}\n\n"
        ax2.text(0.05, 0.95, txt, fontsize=10, family='monospace', va='top', transform=ax2.transAxes,
                 bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

        # 3. Hourly speed
        ax3 = fig.add_subplot(gs[1, 0])
        hrsp = dfs.groupby(['Hour', 'Site'])['Avg mph'].mean().reset_index()
        for s in sites:
            d = hrsp[hrsp['Site'] == s]
            ax3.plot(d['Hour'], d['Avg mph'], marker='o', label=s, lw=2.5, ms=6)
        ax3.set_title('Speed by Hour', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Hour')
        ax3.set_ylabel('mph')
        ax3.set_xticks(range(0,24,3))
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Speed distribution
        ax4 = fig.add_subplot(gs[1, 1])
        for s in sites:
            sp = dfs[dfs['Site'] == s]['Avg mph']
            ax4.hist(sp, bins=30, alpha=0.6, label=s, density=True, ec='black')
        ax4.set_title('Speed Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('mph')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(alpha=0.3)

        # 5. Speed vs Volume
        ax5 = fig.add_subplot(gs[1, 2])
        for s in sites:
            d = dfs[dfs['Site'] == s].sample(min(1000, len(dfs[dfs['Site'] == s])))
            ax5.scatter(d['Total Volume'], d['Avg mph'], alpha=0.4, label=s, s=30)
        ax5.set_title('Speed vs Volume', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Volume')
        ax5.set_ylabel('mph')
        ax5.legend()
        ax5.grid(alpha=0.3)

        # 6. Binned analysis
        ax6 = fig.add_subplot(gs[2, 0])
        for s in sites:
            d = dfs[dfs['Site'] == s][['Avg mph', 'Total Volume']].dropna()
            if len(d) > 50:
                bins = np.linspace(d['Total Volume'].min(), d['Total Volume'].max(), 8)
                d['VB'] = pd.cut(d['Total Volume'], bins=bins)
                bn = d.groupby('VB')['Avg mph'].median()
                ax6.plot(range(len(bn)), bn.values, marker='o', label=s, lw=2)
        ax6.set_title('Speed vs Volume (Binned)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Volume Bin')
        ax6.set_ylabel('Median mph')
        ax6.legend()
        ax6.grid(alpha=0.3)

        # 7. Weekday speed
        ax7 = fig.add_subplot(gs[2, 1])
        wksp = dfs.groupby(['Weekday', 'Site'])['Avg mph'].mean().reset_index()
        for s in sites:
            d = wksp[wksp['Site'] == s]
            ax7.plot(d['Weekday'], d['Avg mph'], marker='s', label=s, lw=2.5, ms=7)
        ax7.set_title('Speed by Day', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Day')
        ax7.set_ylabel('mph')
        ax7.set_xticks(range(7))
        ax7.set_xticklabels(days)
        ax7.legend()
        ax7.grid(alpha=0.3)

        # 8. Speed variability
        ax8 = fig.add_subplot(gs[2, 2])
        var = dfs.groupby('Hour')['Avg mph'].agg(['mean', 'std']).reset_index()
        ax8.plot(var['Hour'], var['mean'], marker='o', lw=2, color='blue', label='Mean')
        ax8.fill_between(var['Hour'], var['mean']-var['std'], var['mean']+var['std'], alpha=0.3, color='blue')
        ax8.set_title('Speed Variability', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Hour')
        ax8.set_ylabel('mph')
        ax8.set_xticks(range(0,24,3))
        ax8.legend()
        ax8.grid(alpha=0.3)

        plt.suptitle('SPEED ANALYSIS DASHBOARD', fontsize=18, fontweight='bold', y=0.995)
        if show_plots: plt.show()
        if save_plots: plt.savefig(os.path.join(out_root, "02_speed_analysis.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # ============================================
    # DASHBOARD 3: TEMPORAL PATTERNS
    # ============================================
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Monthly trends with trendline
    ax1 = fig.add_subplot(gs[0, :])
    mnth = df.groupby('YearMonth')['Total Volume'].sum()
    ax1.plot(range(len(mnth)), mnth.values, marker='o', lw=2.5, ms=8, color='purple', label='Actual')
    x = np.arange(len(mnth))
    z = np.polyfit(x, mnth.values, 1)
    p = np.poly1d(z)
    ax1.plot(x, p(x), 'r--', lw=3, label=f'Trend: {z[0]:+,.0f}/mo')
    ax1.set_title('Monthly Volume & Growth', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # 2. Weekday vs Weekend
    ax2 = fig.add_subplot(gs[1, 0])
    wkp = df.groupby(['Hour', 'IsWeekend'])['Total Volume'].mean().reset_index()
    wkp['Type'] = wkp['IsWeekend'].map({True: 'Weekend', False: 'Weekday'})
    for t in ['Weekday', 'Weekend']:
        d = wkp[wkp['Type'] == t]
        col = 'blue' if t == 'Weekday' else 'red'
        ax2.plot(d['Hour'], d['Total Volume'], marker='o', label=t, lw=3, color=col, ms=6)
    ax2.set_title('Weekday vs Weekend', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Avg Volume')
    ax2.set_xticks(range(0,24,2))
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Peak hours
    ax3 = fig.add_subplot(gs[1, 1])
    havg = df.groupby('Hour')['Total Volume'].mean()
    t75 = havg.quantile(0.75)
    t90 = havg.quantile(0.90)
    cols = ['red' if x > t90 else 'orange' if x > t75 else 'steelblue' for x in havg.values]
    ax3.bar(havg.index, havg.values, color=cols, ec='black', lw=1.5)
    ax3.axhline(t75, color='orange', ls='--', lw=2, label='75th')
    ax3.axhline(t90, color='red', ls='--', lw=2, label='90th')
    ax3.set_title('Peak Hours', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Avg Volume')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. Day of week
    ax4 = fig.add_subplot(gs[1, 2])
    davg = df.groupby('Weekday')['Total Volume'].mean()
    cols_d = ['skyblue']*5 + ['salmon']*2
    ax4.bar(range(7), davg.values, color=cols_d, ec='black', lw=1.5)
    ax4.set_title('By Day of Week', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Avg Volume')
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(days)
    ax4.grid(axis='y', alpha=0.3)

    # 5. Top 10 days
    ax5 = fig.add_subplot(gs[2, 0])
    top = df.groupby('Date')['Total Volume'].sum().sort_values(ascending=False).head(10)
    ax5.barh(range(len(top)), top.values, color='coral', ec='black', lw=1.5)
    ax5.set_yticks(range(len(top)))
    ax5.set_yticklabels([str(d) for d in top.index], fontsize=9)
    ax5.set_xlabel('Total')
    ax5.set_title('Top 10 Days', fontsize=12, fontweight='bold')
    ax5.invert_yaxis()
    ax5.grid(axis='x', alpha=0.3)

    # 6. Variability
    ax6 = fig.add_subplot(gs[2, 1])
    hst = df.groupby('Hour')['Total Volume'].agg(['mean', 'std']).reset_index()
    hst['cv'] = hst['std'] / hst['mean']
    ax6t = ax6.twinx()
    ax6.bar(hst['Hour'], hst['mean'], alpha=0.6, color='skyblue', ec='black')
    ax6t.plot(hst['Hour'], hst['cv'], color='red', marker='o', lw=2.5, ms=6)
    ax6.set_xlabel('Hour')
    ax6.set_ylabel('Mean', color='skyblue')
    ax6t.set_ylabel('CV', color='red')
    ax6.set_title('Volume & Variability', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='y', labelcolor='skyblue')
    ax6t.tick_params(axis='y', labelcolor='red')
    ax6.set_xticks(range(0,24,2))
    ax6.grid(alpha=0.3)

    # 7. Year-over-year
    ax7 = fig.add_subplot(gs[2, 2])
    yly = df.groupby(['Year', 'Month'])['Total Volume'].mean().reset_index()
    for y in sorted(df['Year'].unique()):
        d = yly[yly['Year'] == y]
        ax7.plot(d['Month'], d['Total Volume'], marker='o', label=str(y), lw=2.5, ms=6)
    ax7.set_title('Year-over-Year', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Month')
    ax7.set_ylabel('Avg Volume')
    ax7.set_xticks(range(1,13))
    ax7.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax7.legend()
    ax7.grid(alpha=0.3)

    plt.suptitle('TEMPORAL PATTERNS DASHBOARD', fontsize=18, fontweight='bold', y=0.995)
    if show_plots: plt.show()
    if save_plots: plt.savefig(os.path.join(out_root, "03_temporal_patterns.png"), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_traffic_inline(show_plots=True, save_plots=False)
