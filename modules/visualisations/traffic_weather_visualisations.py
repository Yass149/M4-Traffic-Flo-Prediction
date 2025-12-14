"""
Traffic & Weather Analysis - Notebook Friendly
==============================================
Call `run_traffic_weather_analysis()` to generate all plots inline
"""

import os, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
plt.ion()  # interactive plotting for notebooks

def load_traffic():
    dfs = [pd.read_csv(f) for f in glob.glob("data/traffic/*.csv")]
    df = pd.concat(dfs, ignore_index=True)
    df["Report Date"] = pd.to_datetime(df["Report Date"], dayfirst=True, errors="coerce")
    df["Hour"] = pd.to_datetime(df["Time Period Ending"], dayfirst=True, errors="coerce").dt.hour
    df["Date"] = df["Report Date"].dt.date
    df["Weekday"] = df["Report Date"].dt.dayofweek
    df["Month"] = df["Report Date"].dt.to_period('M')
    for c in ["0 - 520 cm", "521  - 660 cm", "661 - 1160 cm", "1160+ cm", "Avg mph", "Total Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["Total Volume"].notna() & (df["Total Volume"] > 0)]
    df["Site"] = df["Site Name"] if "Site Name" in df.columns else "Unknown"
    return df

def load_weather():
    dfs = [pd.read_csv(f) for f in glob.glob("data/weather/*.csv")]
    df = pd.concat(dfs, ignore_index=True)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["Date"] = df["DATE"].dt.date
    if "TMP" in df.columns:
        df["Temp_C"] = pd.to_numeric(df["TMP"].str.split(",").str[0], errors="coerce") / 10
    if "WND" in df.columns:
        df["Wind_ms"] = pd.to_numeric(df["WND"].str.split(",").str[3], errors="coerce") / 10
    if "VIS" in df.columns:
        df["Vis_m"] = pd.to_numeric(df["VIS"].str.split(",").str[0], errors="coerce")
    return df[["Date", "Temp_C", "Wind_ms", "Vis_m"]].groupby("Date").mean().reset_index()

def run_traffic_weather_analysis(save_plots=False, out_dir="plots/traffic_weather"):
    """
    Runs the traffic & weather analysis and returns a dictionary of figures.
    If save_plots=True, figures are saved as PNGs in out_dir.
    """
    if save_plots:
        os.makedirs(out_dir, exist_ok=True)
    
    df_t = load_traffic()
    df_w = load_weather()
    sites = df_t["Site"].unique()
    
    figs = {}  # store all figures
    
    # -----------------------------
    # 1. SITE COMPARISON
    # -----------------------------
    fig, axes = plt.subplots(2,2,figsize=(16,10))
    
    # Daily volume
    daily = df_t.groupby(['Date','Site'])['Total Volume'].sum().reset_index()
    for site in sites:
        d = daily[daily['Site']==site]
        axes[0,0].plot(d['Date'], d['Total Volume'], label=site, alpha=0.8)
    axes[0,0].set_title('Daily Volume: Site Comparison', fontweight='bold', fontsize=13)
    axes[0,0].set_ylabel('Total Vehicles')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    
    # Hourly patterns
    hourly = df_t.groupby(['Hour','Site'])['Total Volume'].mean().reset_index()
    for site in sites:
        d = hourly[hourly['Site']==site]
        axes[0,1].plot(d['Hour'], d['Total Volume'], marker='o', label=site, linewidth=2.5)
    axes[0,1].set_title('Peak Hours by Site', fontweight='bold', fontsize=13)
    axes[0,1].set_xlabel('Hour')
    axes[0,1].set_ylabel('Avg Vehicles')
    axes[0,1].legend()
    axes[0,1].set_xticks(range(0,24,2))
    axes[0,1].grid(alpha=0.3)
    
    # Speed comparison
    df_speed = df_t[df_t['Avg mph'].notna()]
    if len(df_speed) > 100:
        speed_h = df_speed.groupby(['Hour','Site'])['Avg mph'].mean().reset_index()
        for site in sites:
            d = speed_h[speed_h['Site']==site]
            axes[1,0].plot(d['Hour'], d['Avg mph'], marker='s', label=site, linewidth=2.5)
        axes[1,0].set_title('Average Speed by Hour', fontweight='bold', fontsize=13)
        axes[1,0].set_xlabel('Hour')
        axes[1,0].set_ylabel('mph')
        axes[1,0].legend()
        axes[1,0].set_xticks(range(0,24,2))
        axes[1,0].grid(alpha=0.3)
    
    # Summary stats
    axes[1,1].axis('off')
    stats_text = "SITE STATISTICS\n" + "="*40 + "\n\n"
    for site in sites:
        sd = df_t[df_t['Site']==site]
        total = sd['Total Volume'].sum()
        avg_daily = sd.groupby('Date')['Total Volume'].sum().mean()
        peak = sd.groupby('Hour')['Total Volume'].mean().idxmax()
        speed = sd['Avg mph'].mean() if len(sd[sd['Avg mph'].notna()])>0 else 0
        stats_text += f"{site}:\n  Total: {total:,.0f} vehicles\n  Daily avg: {avg_daily:,.0f}\n"
        stats_text += f"  Peak hour: {peak:02d}:00\n  Avg speed: {speed:.1f} mph\n\n"
    axes[1,1].text(0.1, 0.9, stats_text, fontsize=11, family='monospace', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.suptitle('1. SITE COMPARISON ANALYSIS', fontsize=16, fontweight='bold')
    figs['site_comparison'] = fig
    if save_plots:
        fig.savefig(os.path.join(out_dir,'01_site_comparison.png'), dpi=150, bbox_inches='tight')
    
    # -----------------------------
    # 2. WEATHER IMPACT
    # -----------------------------
    fig, axes = plt.subplots(2,3,figsize=(18,10))
    dt = df_t.groupby('Date').agg({'Total Volume':'sum','Avg mph':'mean'}).reset_index()
    merged = pd.merge(dt, df_w, on='Date', how='inner')
    
    if len(merged)>30:
        # Temp vs volume
        ax = axes[0,0]
        ax.scatter(merged['Temp_C'], merged['Total Volume'], alpha=0.5, s=40)
        z = np.polyfit(merged['Temp_C'].dropna(), merged.loc[merged['Temp_C'].notna(),'Total Volume'],1)
        p = np.poly1d(z)
        ax.plot(sorted(merged['Temp_C']), p(sorted(merged['Temp_C'])), 'r--', lw=2)
        corr = merged[['Temp_C','Total Volume']].corr().iloc[0,1]
        ax.set_title(f'Temperature Impact (r={corr:.3f})', fontweight='bold')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Total Volume')
        ax.grid(alpha=0.3)
        
        # Wind vs volume
        ax = axes[0,1]
        wind_data = merged[merged['Wind_ms'].notna()]
        if len(wind_data) > 10:
            ax.scatter(wind_data['Wind_ms'], wind_data['Total Volume'], alpha=0.5, s=40, color='green')
            corr = wind_data[['Wind_ms','Total Volume']].corr().iloc[0,1]
            ax.set_title(f'Wind Impact (r={corr:.3f})', fontweight='bold')
            ax.set_xlabel('Wind Speed (m/s)')
            ax.set_ylabel('Total Volume')
            ax.grid(alpha=0.3)
        
        # Temp categories
        ax = axes[0,2]
        merged['TempCat'] = pd.cut(merged['Temp_C'], bins=[-20,5,15,25,40],
                                   labels=['Cold <5°C','Cool 5-15°C','Mild 15-25°C','Warm >25°C'])
        temp_vol = merged.groupby('TempCat')['Total Volume'].mean()
        bars = ax.bar(range(len(temp_vol)), temp_vol.values, color=['blue','lightblue','orange','red'], edgecolor='black')
        ax.set_xticks(range(len(temp_vol)))
        ax.set_xticklabels(temp_vol.index, rotation=15, ha='right')
        ax.set_title('Volume by Temperature Range', fontweight='bold')
        ax.set_ylabel('Avg Total Volume')
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2,h,f'{h:,.0f}',ha='center',va='bottom',fontsize=10)
    
    plt.suptitle('2. WEATHER IMPACT ON TRAFFIC', fontsize=16, fontweight='bold')
    figs['weather_impact'] = fig
    if save_plots:
        fig.savefig(os.path.join(out_dir,'02_weather_impact.png'), dpi=150, bbox_inches='tight')
    
    # -----------------------------
    # 3. TEMPORAL INSIGHTS
    # -----------------------------
    fig, axes = plt.subplots(2,2,figsize=(16,10))
    df_t['IsWeekend'] = df_t['Weekday'].isin([5,6])
    wkd = df_t.groupby(['Hour','IsWeekend'])['Total Volume'].mean().reset_index()
    wkd['Type'] = wkd['IsWeekend'].map({True:'Weekend',False:'Weekday'})
    for t in ['Weekday','Weekend']:
        d = wkd[wkd['Type']==t]
        axes[0,0].plot(d['Hour'], d['Total Volume'], marker='o', label=t, linewidth=2.5)
    axes[0,0].set_title('Weekday vs Weekend Patterns', fontweight='bold', fontsize=13)
    axes[0,0].set_xlabel('Hour')
    axes[0,0].set_ylabel('Avg Volume')
    axes[0,0].legend()
    axes[0,0].set_xticks(range(0,24,2))
    axes[0,0].grid(alpha=0.3)
    
    monthly = df_t.groupby('Month')['Total Volume'].sum()
    axes[0,1].plot(range(len(monthly)), monthly.values, marker='o', linewidth=2, color='purple')
    axes[0,1].set_title('Monthly Traffic Trend', fontweight='bold', fontsize=13)
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Total Volume')
    axes[0,1].grid(alpha=0.3)
    x = np.arange(len(monthly))
    z = np.polyfit(x, monthly.values,1)
    p = np.poly1d(z)
    axes[0,1].plot(x,p(x),'r--', linewidth=2,label=f'Trend: {z[0]:+.0f}/mo')
    axes[0,1].legend()
    
    hourly_avg = df_t.groupby('Hour')['Total Volume'].mean()
    threshold = hourly_avg.quantile(0.75)
    colors = ['red' if x>threshold else 'steelblue' for x in hourly_avg]
    axes[1,0].bar(hourly_avg.index, hourly_avg.values,color=colors,edgecolor='black')
    axes[1,0].axhline(threshold,color='red',linestyle='--',linewidth=2,label=f'75th percentile: {threshold:.0f}')
    axes[1,0].set_title('Peak Hour Identification', fontweight='bold', fontsize=13)
    axes[1,0].set_xlabel('Hour')
    axes[1,0].set_ylabel('Avg Volume')
    axes[1,0].legend()
    axes[1,0].set_xticks(range(0,24,2))
    axes[1,0].grid(axis='y',alpha=0.3)
    
    top_days = df_t.groupby('Date')['Total Volume'].sum().sort_values(ascending=False).head(10)
    axes[1,1].barh(range(len(top_days)), top_days.values,color='coral',edgecolor='black')
    axes[1,1].set_yticks(range(len(top_days)))
    axes[1,1].set_yticklabels([str(d) for d in top_days.index],fontsize=9)
    axes[1,1].set_xlabel('Total Volume')
    axes[1,1].set_title('Top 10 Busiest Days', fontweight='bold', fontsize=13)
    axes[1,1].grid(axis='x', alpha=0.3)
    
    plt.suptitle('3. TEMPORAL PATTERNS & TRENDS', fontsize=16, fontweight='bold')
    figs['temporal_insights'] = fig
    if save_plots:
        fig.savefig(os.path.join(out_dir,'03_temporal_insights.png'), dpi=150, bbox_inches='tight')
    
    return figs
