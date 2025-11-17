import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
import geopandas as gpd
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

# Set black-and-white research paper style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("gray")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.color'] = 'gray'

# Regulation dates
REGULATIONS = {
    'GDPR': datetime(2018, 5, 25),
    'DSA': datetime(2022, 11, 16),
    'DMA': datetime(2023, 5, 2),
    'AI Act': datetime(2025, 2, 2)
}

REGULATION_COLORS = {
    'GDPR': '#000000',
    'DSA': '#444444',
    'DMA': '#666666',
    'AI Act': '#888888'
}

# COVID-19 pandemic period
COVID_START = datetime(2020, 3, 11)
COVID_END = datetime(2023, 5, 1)

def add_regulation_lines(ax, start_year=2010):
    """Add vertical lines for regulations with text labels next to them and COVID-19 shaded region"""
    # Add COVID-19 shaded region
    covid_start_x = COVID_START.year + (COVID_START.month - 1) / 12
    covid_end_x = COVID_END.year + (COVID_END.month - 1) / 12
    ax.axvspan(covid_start_x, covid_end_x, alpha=0.15, color='gray', label='COVID-19 Pandemic')
    
    for name, date in REGULATIONS.items():
        if date.year >= start_year:
            x_pos = date.year + (date.month - 1) / 12
            # Draw darker vertical line
            ax.axvline(x=x_pos,
                      color='#222222',
                      linestyle=':',
                      linewidth=1.5,
                      alpha=0.8,
                      label='_nolegend_')
            # Add text annotation next to the line
            y_pos = ax.get_ylim()[1] * 0.95  # Position near top of plot
            ax.text(x_pos + 0.1, y_pos, name, 
                   rotation=90, verticalalignment='top',
                   fontsize=8, alpha=0.7, color='#222222')

def load_data():
    """Load all CSV files"""
    data = {}
    
    # Broadband prices
    data['broadband_prices'] = pd.read_csv('data/broadband prices.csv')
    
    # Individual cloud service use
    data['individual_cloud'] = pd.read_csv('data/Individual Cloud Service Use.csv')
    
    # GERD OECD
    data['gerd'] = pd.read_csv('data/GERD OECD.csv')
    
    # Broadband traffic
    data['broadband_traffic'] = pd.read_csv('data/broadband traffic.csv')
    
    # Cloud computing services
    data['cloud_services'] = pd.read_csv('data/Cloud Computing Services.csv')
    
    # BRED OECD
    data['bred'] = pd.read_csv('data/BRED OECD.csv')
    
    # Internet usage
    data['internet_usage'] = pd.read_csv('data/Internet Usage _ UN DATA.csv')
    
    # GOVERD OECD
    data['goverd'] = pd.read_csv('data/GOVERD OECD.csv')
    
    # Broadband speed
    data['broadband_speed'] = pd.read_csv('data/broadband speed.csv')
    
    # BERID OECD
    data['berid'] = pd.read_csv('data/BERID OECD.csv')
    
    # HRED OECD
    data['hred'] = pd.read_csv('data/HRED OECD.csv')
    
    # VPN searches
    data['vpn_searches'] = pd.read_csv('data/multiTimeline.csv')
    
    return data

def prepare_oecd_data(df):
    """Prepare OECD data (transposed format)"""
    df_melted = df.melt(id_vars=['Time period'], var_name='Year', value_name='Value')
    df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
    df_melted = df_melted[df_melted['Year'] >= 2010]
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')
    df_melted = df_melted.dropna(subset=['Value', 'Year'])
    return df_melted

# Load data
data = load_data()

# Country name mappings
country_codes = {'DE': 'Germany', 'IE': 'Ireland', 'CH': 'Switzerland'}
country_colors = {'Germany': '#000000', 'Ireland': '#444444', 'Switzerland': '#888888'}
country_styles = {'Germany': '-', 'Ireland': '--', 'Switzerland': '-.'}
country_markers = {'Germany': 'o', 'Ireland': 's', 'Switzerland': '^'}

# ============================================================
# 1. Multi-Country Broadband Pricing Timeline with Regulation Overlays
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_prices = data['broadband_prices']
df_prices = df_prices[df_prices['geo'].isin(['DE', 'IE'])]
df_prices = df_prices[df_prices['TIME_PERIOD'] >= 2010]

# Filter for one specific offer type to avoid multiple lines
df_prices = df_prices[df_prices['offer'] == 'FI_MBPS100-200']

# Group by country and year, taking mean if there are duplicates
df_prices_clean = df_prices.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()

for geo_code in ['DE', 'IE']:
    country = country_codes.get(geo_code, geo_code)
    country_data = df_prices_clean[df_prices_clean['geo'] == geo_code].sort_values('TIME_PERIOD')
    if not country_data.empty:
        ax.plot(country_data['TIME_PERIOD'], country_data['OBS_VALUE'], 
               marker=country_markers.get(country, 'o'), linewidth=2, markersize=7, 
               linestyle=country_styles.get(country, '-'),
               label=country,
               color=country_colors.get(country, '#000000'))

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Price (PPP in EUR)', fontsize=14, fontweight='bold')
ax.set_title('Multi-Country Broadband Pricing Timeline (100-200 Mbps) with Regulation Overlays', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_01_broadband_pricing_timeline.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 2. Internet Usage Adoption Trajectory by Country
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_usage = data['internet_usage']
df_usage = df_usage[df_usage['Country or Area'].isin(['Germany', 'Ireland', 'Switzerland'])]
df_usage = df_usage[df_usage['Year'] >= 2010]

# Clean and sort
df_usage_clean = df_usage.groupby(['Country or Area', 'Year'])['Value'].mean().reset_index()

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_usage_clean[df_usage_clean['Country or Area'] == country].sort_values('Year')
    if not country_data.empty:
        ax.plot(country_data['Year'], country_data['Value'], 
               marker=country_markers[country], linewidth=2, markersize=7, 
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Internet Users (% of Population)', fontsize=14, fontweight='bold')
ax.set_title('Internet Usage Adoption Trajectory by Country with Regulatory Milestones', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_02_internet_usage_trajectory.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 3. Broadband Speed Coverage Evolution by Country
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_speed = data['broadband_speed']
df_speed = df_speed[df_speed['geo'].isin(['DE', 'IE', 'CH'])]
df_speed = df_speed[df_speed['TIME_PERIOD'] >= 2010]

# Filter for >1 Gbps coverage for cleaner comparison
df_speed_filtered = df_speed[df_speed['inet_spd'] == 'GBPS_GT1']
df_speed_clean = df_speed_filtered.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()

for geo_code in ['DE', 'IE', 'CH']:
    country = country_codes[geo_code]
    country_data = df_speed_clean[df_speed_clean['geo'] == geo_code].sort_values('TIME_PERIOD')
    if not country_data.empty:
        ax.plot(country_data['TIME_PERIOD'], country_data['OBS_VALUE'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, 
               color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Coverage (% of Households with >1 Gbps)', fontsize=14, fontweight='bold')
ax.set_title('Broadband Speed Coverage Evolution (>1 Gbps) by Country', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_03_broadband_speed_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 4. Individual Cloud Service Adoption Rates by Country
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_cloud = data['individual_cloud']
df_cloud = df_cloud[df_cloud['geo'].isin(['DE', 'IE', 'CH'])]
df_cloud = df_cloud[df_cloud['TIME_PERIOD'] >= 2010]

# Clean data - group by country and year
df_cloud_clean = df_cloud.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()

for geo_code in ['DE', 'IE', 'CH']:
    country = country_codes[geo_code]
    country_data = df_cloud_clean[df_cloud_clean['geo'] == geo_code].sort_values('TIME_PERIOD')
    if not country_data.empty:
        ax.plot(country_data['TIME_PERIOD'], country_data['OBS_VALUE'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Adoption Rate (% of Individuals)', fontsize=14, fontweight='bold')
ax.set_title('Individual Cloud Service Adoption Rates by Country', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_04_individual_cloud_adoption.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 5. Total R&D Expenditure Growth Rates (GERD)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_gerd = prepare_oecd_data(data['gerd'])

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_gerd[df_gerd['Time period'] == country].sort_values('Year')
    if not country_data.empty:
        ax.plot(country_data['Year'], country_data['Value'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('GERD Growth Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Total R&D Expenditure Growth Rates (GERD) with Regulatory Milestones', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_05_gerd_growth_rates.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 6. Business R&D Expenditure (BRED) Trends
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_bred = prepare_oecd_data(data['bred'])

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_bred[df_bred['Time period'] == country].sort_values('Year')
    if not country_data.empty:
        ax.plot(country_data['Year'], country_data['Value'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('BRED Growth Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Business R&D Expenditure (BRED) Trends with Regulatory Boundaries', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_06_bred_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 7. Government R&D Spending (GOVERD) Response
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_goverd = prepare_oecd_data(data['goverd'])

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_goverd[df_goverd['Time period'] == country].sort_values('Year')
    if not country_data.empty:
        ax.plot(country_data['Year'], country_data['Value'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('GOVERD Growth Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Government R&D Spending (GOVERD) Response to Regulations', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_07_goverd_response.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 8. Higher Education R&D Expenditure (HRED)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_hred = prepare_oecd_data(data['hred'])

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_hred[df_hred['Time period'] == country].sort_values('Year')
    if not country_data.empty:
        ax.plot(country_data['Year'], country_data['Value'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('HRED Growth Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Higher Education R&D Expenditure (HRED) Acceleration Post-Regulation', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_08_hred_acceleration.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 9-11. Comparative R&D Investment by Source (per country)
# ============================================================
for country in ['Germany', 'Ireland', 'Switzerland']:
    fig, ax = plt.subplots(figsize=(16, 9))
    
    for dataset_name, marker, label in [('gerd', 'o', 'GERD'), ('bred', 's', 'BRED'), 
                                         ('goverd', '^', 'GOVERD'), ('hred', 'D', 'HRED')]:
        df_temp = prepare_oecd_data(data[dataset_name])
        country_data = df_temp[df_temp['Time period'] == country].sort_values('Year')
        if not country_data.empty:
            ax.plot(country_data['Year'], country_data['Value'], 
                   marker=marker, linewidth=2, markersize=7, label=label)
    
    add_regulation_lines(ax)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Growth Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Comparative R&D Investment by Source - {country}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    graph_num = 9 if country == 'Germany' else (10 if country == 'Ireland' else 11)
    plt.savefig(f'graph_{graph_num:02d}_comparative_rd_{country.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# 12-14. Enterprise Cloud Computing Adoption by Company Size
# ============================================================
# Define distinct markers and line styles for each company size category
size_markers = {'10-249': 'o', '50-249': 's', 'GE10': '^', 'GE250': 'D'}
size_colors = {'10-249': '#000000', '50-249': '#333333', 'GE10': '#666666', 'GE250': '#999999'}
size_linestyles = {'10-249': '-', '50-249': '--', 'GE10': '-.', 'GE250': ':'}
# Map size codes to readable labels based on employee ranges
size_labels = {
    '10-249': '10-249 employees',
    '50-249': '50-249 employees',
    'GE10': '10+ employees',
    'GE250': '250+ employees'
}

for idx, geo_code in enumerate(['DE', 'IE', 'CH']):
    fig, ax = plt.subplots(figsize=(16, 9))
    df_enterprise = data['cloud_services']
    df_enterprise = df_enterprise[df_enterprise['geo'] == geo_code]
    df_enterprise = df_enterprise[df_enterprise['TIME_PERIOD'] >= 2010]
    
    # Clean and group by size and year
    df_enterprise_clean = df_enterprise.groupby(['size_emp', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()
    
    # Only plot categories that are in our size_labels mapping (exclude unexpected values)
    for size in df_enterprise_clean['size_emp'].unique():
        if size not in size_labels:
            continue  # Skip any size categories not in our mapping
        size_data = df_enterprise_clean[df_enterprise_clean['size_emp'] == size].sort_values('TIME_PERIOD')
        if not size_data.empty:
            marker = size_markers[size]
            color = size_colors[size]
            linestyle = size_linestyles[size]
            label = size_labels[size]
            ax.plot(size_data['TIME_PERIOD'], size_data['OBS_VALUE'], 
                   marker=marker, linewidth=2, markersize=7, 
                   linestyle=linestyle, color=color, label=label)
    
    add_regulation_lines(ax)
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cloud Adoption (% of Enterprises)', fontsize=14, fontweight='bold')
    ax.set_title(f'Enterprise Cloud Computing Adoption by Company Size - {country_codes[geo_code]}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'graph_{12+idx:02d}_enterprise_cloud_{country_codes[geo_code].lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# 15. Broadband Traffic Volume Evolution
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_traffic = data['broadband_traffic']
df_traffic = df_traffic[df_traffic['geo'].isin(['DE', 'IE', 'CH'])]
df_traffic = df_traffic[df_traffic['TIME_PERIOD'] >= 2010]

# Clean data
df_traffic_clean = df_traffic.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()

for geo_code in ['DE', 'IE', 'CH']:
    country = country_codes[geo_code]
    country_data = df_traffic_clean[df_traffic_clean['geo'] == geo_code].sort_values('TIME_PERIOD')
    if not country_data.empty:
        ax.plot(country_data['TIME_PERIOD'], country_data['OBS_VALUE'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Traffic Volume (Exabytes)', fontsize=14, fontweight='bold')
ax.set_title('Broadband Traffic Volume Evolution with Regulatory Overlays', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_15_broadband_traffic_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 16. Two-Country Comparison: Broadband Prices by Regulation Period
# ============================================================
df_prices = data['broadband_prices']
df_prices = df_prices[df_prices['geo'].isin(['DE', 'IE'])]
df_prices = df_prices[df_prices['TIME_PERIOD'] >= 2010]
df_prices = df_prices[df_prices['offer'] == 'FI_MBPS100-200']
df_prices_clean = df_prices.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()

for idx, geo_code in enumerate(['DE', 'IE']):
    fig, ax = plt.subplots(figsize=(16, 9))
    country = country_codes[geo_code]
    country_data = df_prices_clean[df_prices_clean['geo'] == geo_code].sort_values('TIME_PERIOD')
    if not country_data.empty:
        ax.plot(country_data['TIME_PERIOD'], country_data['OBS_VALUE'], 
                marker=country_markers[country], linewidth=2, markersize=7,
                linestyle=country_styles[country],
                color=country_colors[country],
                label=country)
        add_regulation_lines(ax)
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price (PPP EUR)', fontsize=14, fontweight='bold')
        ax.set_title(f'Broadband Prices (100-200 Mbps) - {country}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'graph_16{chr(97+idx)}_broadband_prices_{country.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================
# 17. Three-Country Comparison: Internet Usage Growth Trajectory
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_usage = data['internet_usage']
df_usage = df_usage[df_usage['Country or Area'].isin(['Germany', 'Ireland', 'Switzerland'])]
df_usage = df_usage[df_usage['Year'] >= 2010]
df_usage_clean = df_usage.groupby(['Country or Area', 'Year'])['Value'].mean().reset_index()

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_usage_clean[df_usage_clean['Country or Area'] == country].sort_values('Year')
    if not country_data.empty:
        ax.plot(country_data['Year'], country_data['Value'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Internet Users (% of Population)', fontsize=14, fontweight='bold')
ax.set_title('Three-Country Comparison: Internet Usage Growth Trajectory', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9, fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_17_three_country_internet_usage.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 18. Three-Country Comparison: Cloud Service Adoption Rates
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))

# Individual cloud
df_cloud = data['individual_cloud']
df_cloud = df_cloud[df_cloud['geo'].isin(['DE', 'IE', 'CH'])]
df_cloud = df_cloud[df_cloud['TIME_PERIOD'] >= 2010]
df_cloud_clean = df_cloud.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()

for geo_code in ['DE', 'IE', 'CH']:
    country = country_codes[geo_code]
    country_data = df_cloud_clean[df_cloud_clean['geo'] == geo_code].sort_values('TIME_PERIOD')
    if not country_data.empty:
        ax.plot(country_data['TIME_PERIOD'], country_data['OBS_VALUE'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=f'{country} (Individual)', 
               color=country_colors[country])

# Enterprise data (10-249 employees)
df_enterprise = data['cloud_services']
df_enterprise = df_enterprise[df_enterprise['geo'].isin(['DE', 'IE', 'CH'])]
df_enterprise = df_enterprise[df_enterprise['TIME_PERIOD'] >= 2010]
df_enterprise = df_enterprise[df_enterprise['size_emp'] == '10-249']
df_enterprise_clean = df_enterprise.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()

for geo_code in ['DE', 'IE', 'CH']:
    country = country_codes[geo_code]
    country_data = df_enterprise_clean[df_enterprise_clean['geo'] == geo_code].sort_values('TIME_PERIOD')
    if not country_data.empty:
        ax.plot(country_data['TIME_PERIOD'], country_data['OBS_VALUE'], 
               marker=country_markers[country], linewidth=1.5, markersize=5, linestyle=':',
               label=f'{country} (Enterprise 10-249)', 
               color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Cloud Adoption Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Three-Country Comparison: Cloud Service Adoption Rates', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_18_three_country_cloud_adoption.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 19. R&D Investment Acceleration Index
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))

df_gerd = prepare_oecd_data(data['gerd'])

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_gerd[df_gerd['Time period'] == country].sort_values('Year').copy()
    if not country_data.empty and len(country_data) > 0:
        country_data = country_data.reset_index(drop=True)
        country_data['Cumulative_Index'] = 100.0
        for i in range(1, len(country_data)):
            prev_idx = country_data.loc[i-1, 'Cumulative_Index']
            growth = country_data.loc[i, 'Value']
            if not np.isnan(growth):
                country_data.loc[i, 'Cumulative_Index'] = prev_idx * (1 + growth/100)
        
        ax.plot(country_data['Year'], country_data['Cumulative_Index'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Investment Index (2010 = 100)', fontsize=14, fontweight='bold')
ax.set_title('R&D Investment Acceleration Index (GERD) by Country', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_19_rd_acceleration_index.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 19a. BRED Investment Acceleration Index
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))

df_bred = prepare_oecd_data(data['bred'])

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_bred[df_bred['Time period'] == country].sort_values('Year').copy()
    if not country_data.empty and len(country_data) > 0:
        country_data = country_data.reset_index(drop=True)
        country_data['Cumulative_Index'] = 100.0
        for i in range(1, len(country_data)):
            prev_idx = country_data.loc[i-1, 'Cumulative_Index']
            growth = country_data.loc[i, 'Value']
            if not np.isnan(growth):
                country_data.loc[i, 'Cumulative_Index'] = prev_idx * (1 + growth/100)
        
        ax.plot(country_data['Year'], country_data['Cumulative_Index'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Investment Index (2010 = 100)', fontsize=14, fontweight='bold')
ax.set_title('Business R&D Investment Acceleration Index (BRED) by Country', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_19a_bred_acceleration_index.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 19b. GOVERD Investment Acceleration Index
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))

df_goverd = prepare_oecd_data(data['goverd'])

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_goverd[df_goverd['Time period'] == country].sort_values('Year').copy()
    if not country_data.empty and len(country_data) > 0:
        country_data = country_data.reset_index(drop=True)
        country_data['Cumulative_Index'] = 100.0
        for i in range(1, len(country_data)):
            prev_idx = country_data.loc[i-1, 'Cumulative_Index']
            growth = country_data.loc[i, 'Value']
            if not np.isnan(growth):
                country_data.loc[i, 'Cumulative_Index'] = prev_idx * (1 + growth/100)
        
        ax.plot(country_data['Year'], country_data['Cumulative_Index'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Investment Index (2010 = 100)', fontsize=14, fontweight='bold')
ax.set_title('Government R&D Investment Acceleration Index (GOVERD) by Country', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_19b_goverd_acceleration_index.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 19c. HRED Investment Acceleration Index
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))

df_hred = prepare_oecd_data(data['hred'])

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_hred[df_hred['Time period'] == country].sort_values('Year').copy()
    if not country_data.empty and len(country_data) > 0:
        country_data = country_data.reset_index(drop=True)
        country_data['Cumulative_Index'] = 100.0
        for i in range(1, len(country_data)):
            prev_idx = country_data.loc[i-1, 'Cumulative_Index']
            growth = country_data.loc[i, 'Value']
            if not np.isnan(growth):
                country_data.loc[i, 'Cumulative_Index'] = prev_idx * (1 + growth/100)
        
        ax.plot(country_data['Year'], country_data['Cumulative_Index'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Investment Index (2010 = 100)', fontsize=14, fontweight='bold')
ax.set_title('Higher Education R&D Investment Acceleration Index (HRED) by Country', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_19c_hred_acceleration_index.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 20. YoY Growth Rate Comparison Around Regulatory Milestones
# ============================================================
reg_dates = [2018, 2022, 2023, 2025]
reg_names = ['GDPR (2018)', 'DSA (2022)', 'DMA (2023)', 'AI Act (2025)']

for idx, (year, name) in enumerate(zip(reg_dates, reg_names)):
    fig, ax = plt.subplots(figsize=(16, 9))
    
    years_range = [year-2, year-1, year, year+1, year+2]
    
    df_gerd = prepare_oecd_data(data['gerd'])
    x_pos = np.arange(len(years_range))
    width = 0.25
    
    for cidx, country in enumerate(['Germany', 'Ireland', 'Switzerland']):
        country_data = df_gerd[df_gerd['Time period'] == country]
        values = []
        for y in years_range:
            val = country_data[country_data['Year'] == y]['Value'].values
            values.append(val[0] if len(val) > 0 else np.nan)
        
        positions = x_pos + cidx * width
        hatch_patterns = ['', '//', 'xx']
        ax.bar(positions, values, width, label=country,
               color=country_colors[country], 
               edgecolor='black', linewidth=0.5,
               hatch=hatch_patterns[cidx])
    
    ax.axvline(x=2 + width, color='#222222', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('GERD Growth Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'YoY GERD Growth Rate Comparison Around {name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(years_range)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'graph_20{chr(97+idx)}_yoy_growth_{name.split()[0].lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# 21. Digital Infrastructure Maturity Index
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))

df_usage = data['internet_usage']
df_usage = df_usage[df_usage['Country or Area'].isin(['Germany', 'Ireland', 'Switzerland'])]
df_usage = df_usage[df_usage['Year'] >= 2010]
df_usage_clean = df_usage.groupby(['Country or Area', 'Year'])['Value'].mean().reset_index()

for country in ['Germany', 'Ireland', 'Switzerland']:
    usage_data = df_usage_clean[df_usage_clean['Country or Area'] == country].sort_values('Year')
    
    if not usage_data.empty:
        years = usage_data['Year'].values
        composite = usage_data['Value'].values
        
        ax.plot(years, composite, marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Internet Penetration (%)', fontsize=14, fontweight='bold')
ax.set_title('Digital Infrastructure Maturity (Internet Penetration) by Country', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_21_digital_maturity_index.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 22. Broadband Traffic Growth Rate Before/After Each Regulation
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_traffic = data['broadband_traffic']
df_traffic = df_traffic[df_traffic['geo'].isin(['DE', 'IE', 'CH'])]
df_traffic = df_traffic[df_traffic['TIME_PERIOD'] >= 2010]
df_traffic_clean = df_traffic.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()

reg_years = [2018, 2022, 2023]
categories = []
for year in reg_years:
    categories.extend([f'Before {year}', f'After {year}'])

x_pos = np.arange(len(categories))
width = 0.25

for cidx, geo_code in enumerate(['DE', 'IE', 'CH']):
    country = country_codes[geo_code]
    country_data = df_traffic_clean[df_traffic_clean['geo'] == geo_code].sort_values('TIME_PERIOD').copy()
    if len(country_data) > 1:
        country_data['Growth_Rate'] = country_data['OBS_VALUE'].pct_change() * 100
        
        values = []
        for year in reg_years:
            before = country_data[country_data['TIME_PERIOD'] < year]['Growth_Rate'].mean()
            after = country_data[country_data['TIME_PERIOD'] >= year]['Growth_Rate'].mean()
            values.extend([before if not np.isnan(before) else 0, 
                          after if not np.isnan(after) else 0])
        
        offset = (cidx - 1) * width
        hatch_patterns = ['', '//', 'xx']
        ax.bar(x_pos + offset, values, width, 
              label=country, 
              color=country_colors[country],
              edgecolor='black', linewidth=0.5,
              hatch=hatch_patterns[cidx])

ax.set_xlabel('Period', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Traffic Growth Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Broadband Traffic Growth Rate Before/After Each Regulation', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('graph_22_traffic_growth_before_after.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 23. Cumulative Investment Change Index
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))

df_gerd = prepare_oecd_data(data['gerd'])

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_gerd[df_gerd['Time period'] == country].sort_values('Year').copy()
    if not country_data.empty and len(country_data) > 0:
        country_data = country_data.reset_index(drop=True)
        country_data['Cumulative_Index'] = 100.0
        for i in range(1, len(country_data)):
            prev_idx = country_data.loc[i-1, 'Cumulative_Index']
            growth = country_data.loc[i, 'Value']
            if not np.isnan(growth):
                country_data.loc[i, 'Cumulative_Index'] = prev_idx * (1 + growth/100)
        
        ax.plot(country_data['Year'], country_data['Cumulative_Index'], 
               marker=country_markers[country], linewidth=2, markersize=7,
               linestyle=country_styles[country],
               label=country, color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Investment Index (2010 = 100)', fontsize=14, fontweight='bold')
ax.set_title('Cumulative R&D Investment Change Index', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_23_cumulative_investment_index.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 24. Pre vs Post-Regulation Average Metrics Comparison (ALL GROWTH RATES)
# ============================================================
cutoff_year = 2018

# Graph 24a: Internet Usage GROWTH RATES
fig, ax = plt.subplots(figsize=(16, 9))
df_usage = data['internet_usage']
df_usage = df_usage[df_usage['Country or Area'].isin(['Germany', 'Ireland', 'Switzerland'])]
df_usage = df_usage[df_usage['Year'] >= 2010]
df_usage_clean = df_usage.groupby(['Country or Area', 'Year'])['Value'].mean().reset_index()

pre_vals = []
post_vals = []
labels = []

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_usage_clean[df_usage_clean['Country or Area'] == country].sort_values('Year').copy()
    if len(country_data) > 1:
        country_data['Growth_Rate'] = country_data['Value'].pct_change() * 100
        
        pre = country_data[country_data['Year'] < cutoff_year]['Growth_Rate'].mean()
        post = country_data[country_data['Year'] >= cutoff_year]['Growth_Rate'].mean()
        
        pre_vals.append(pre if not np.isnan(pre) else 0)
        post_vals.append(post if not np.isnan(post) else 0)
        labels.append(country)

x = np.arange(len(labels))
width = 0.35

ax.bar(x - width/2, pre_vals, width, label='Pre-GDPR (2010-2017)', 
       color='#CCCCCC', edgecolor='black', linewidth=0.5, hatch='')
ax.bar(x + width/2, post_vals, width, label='Post-GDPR (2018+)', 
       color='#666666', edgecolor='black', linewidth=0.5, hatch='//')

ax.set_xlabel('Country', fontsize=14, fontweight='bold')
ax.set_ylabel('Internet Usage Growth Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Pre vs Post-GDPR: Internet Usage Growth Rate Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('graph_24a_internet_usage_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Graph 24b: Cloud Adoption GROWTH RATES
fig, ax = plt.subplots(figsize=(16, 9))
df_cloud = data['individual_cloud']
df_cloud = df_cloud[df_cloud['geo'].isin(['DE', 'IE', 'CH'])]
df_cloud = df_cloud[df_cloud['TIME_PERIOD'] >= 2010]
df_cloud_clean = df_cloud.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()

pre_vals = []
post_vals = []
labels = []

country_map = {'DE': 'Germany', 'IE': 'Ireland', 'CH': 'Switzerland'}

for geo_code in ['DE', 'IE', 'CH']:
    country_data = df_cloud_clean[df_cloud_clean['geo'] == geo_code].sort_values('TIME_PERIOD').copy()
    if len(country_data) > 1:
        country_data['Growth_Rate'] = country_data['OBS_VALUE'].pct_change() * 100
        
        pre = country_data[country_data['TIME_PERIOD'] < cutoff_year]['Growth_Rate'].mean()
        post = country_data[country_data['TIME_PERIOD'] >= cutoff_year]['Growth_Rate'].mean()
        
        pre_vals.append(pre if not np.isnan(pre) else 0)
        post_vals.append(post if not np.isnan(post) else 0)
        labels.append(country_map[geo_code])

x = np.arange(len(labels))
width = 0.35

ax.bar(x - width/2, pre_vals, width, label='Pre-GDPR (2010-2017)', 
       color='#CCCCCC', edgecolor='black', linewidth=0.5, hatch='')
ax.bar(x + width/2, post_vals, width, label='Post-GDPR (2018+)', 
       color='#666666', edgecolor='black', linewidth=0.5, hatch='//')

ax.set_xlabel('Country', fontsize=14, fontweight='bold')
ax.set_ylabel('Cloud Adoption Growth Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Pre vs Post-GDPR: Cloud Adoption Growth Rate Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('graph_24b_cloud_adoption_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Graph 24c: GERD comparison
fig, ax = plt.subplots(figsize=(16, 9))
df_gerd = prepare_oecd_data(data['gerd'])

pre_vals = []
post_vals = []
labels = []

for country in ['Germany', 'Ireland', 'Switzerland']:
    country_data = df_gerd[df_gerd['Time period'] == country]
    
    pre = country_data[country_data['Year'] < cutoff_year]['Value'].mean()
    post = country_data[country_data['Year'] >= cutoff_year]['Value'].mean()
    
    pre_vals.append(pre if not np.isnan(pre) else 0)
    post_vals.append(post if not np.isnan(post) else 0)
    labels.append(country)

x = np.arange(len(labels))
width = 0.35

ax.bar(x - width/2, pre_vals, width, label='Pre-GDPR (2010-2017)', 
       color='#CCCCCC', edgecolor='black', linewidth=0.5, hatch='')
ax.bar(x + width/2, post_vals, width, label='Post-GDPR (2018+)', 
       color='#666666', edgecolor='black', linewidth=0.5, hatch='//')

ax.set_xlabel('Country', fontsize=14, fontweight='bold')
ax.set_ylabel('GERD Growth Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Pre vs Post-GDPR: GERD Growth Rate Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('graph_24c_gerd_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 25. Enterprise Cloud Adoption vs Regulatory Periods
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))

df_enterprise = data['cloud_services']
df_enterprise = df_enterprise[df_enterprise['geo'].isin(['DE', 'IE', 'CH'])]
df_enterprise = df_enterprise[df_enterprise['TIME_PERIOD'] >= 2010]

# Define specific size categories
size_map = {
    '10-249': '10-249',
    'GE250': '250+'
}

df_enterprise_clean = df_enterprise.groupby(['geo', 'size_emp', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()

for geo_code in ['DE', 'IE', 'CH']:
    country = country_codes[geo_code]
    for size_key, size_label in size_map.items():
        size_data = df_enterprise_clean[
            (df_enterprise_clean['geo'] == geo_code) & 
            (df_enterprise_clean['size_emp'] == size_key)
        ].sort_values('TIME_PERIOD')
        
        if not size_data.empty:
            linestyle = country_styles[country] if size_key == '10-249' else ':'
            linewidth = 2 if size_key == '10-249' else 1.5
            markersize = 6 if size_key == '10-249' else 5
            ax.plot(size_data['TIME_PERIOD'], size_data['OBS_VALUE'], 
                   marker=country_markers[country], linewidth=linewidth, 
                   markersize=markersize, linestyle=linestyle,
                   label=f'{country} ({size_label} employees)',
                   color=country_colors[country])

add_regulation_lines(ax)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Cloud Adoption (% of Enterprises)', fontsize=14, fontweight='bold')
ax.set_title('Enterprise Cloud Adoption vs Regulatory Periods', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_25_enterprise_cloud_vs_regulations.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 26-28. Choropleth Maps: Before/After GDPR Comparisons
# ============================================================

def create_choropleth_comparison(metric_name, before_data, after_data, filename_base, 
                                 unit_label, cmap_name='Greys'):
    """Create side-by-side before/after GDPR choropleth maps with horizontal layout"""
    
    # Load GeoJSON files for the three countries
    germany = gpd.read_file('geojson/germany.geojson')
    ireland = gpd.read_file('geojson/ireland.geojson')
    switzerland = gpd.read_file('geojson/switzerland.geojson')
    
    # Filter out Northern Ireland (keep only Republic of Ireland)
    ireland = ireland[ireland['GID_0'] == 'IRL'].copy()
    
    # Dissolve Ireland to create single outline (remove province boundaries)
    ireland = ireland.dissolve()
    
    # Normalize geometries to same scale
    def normalize_geometry(gdf):
        """Scale geometry to fit in unit square while preserving aspect ratio"""
        bounds = gdf.total_bounds  # minx, miny, maxx, maxy
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        scale = max(width, height)
        
        # Translate to origin and scale
        gdf_norm = gdf.copy()
        gdf_norm['geometry'] = gdf_norm.geometry.translate(-bounds[0], -bounds[1])
        gdf_norm['geometry'] = gdf_norm.geometry.scale(xfact=1/scale, yfact=1/scale, origin=(0, 0))
        return gdf_norm
    
    germany_norm = normalize_geometry(germany)
    ireland_norm = normalize_geometry(ireland)
    switzerland_norm = normalize_geometry(switzerland)
    
    # Position countries horizontally with spacing
    spacing = 0.3
    ireland_norm['geometry'] = ireland_norm.geometry.translate(xoff=0, yoff=0)
    germany_norm['geometry'] = germany_norm.geometry.translate(xoff=1 + spacing, yoff=0)
    switzerland_norm['geometry'] = switzerland_norm.geometry.translate(xoff=2 + 2*spacing, yoff=0)
    
    # Create figure with two subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 8))
    
    # Determine unified color scale across both maps
    all_values = list(before_data.values()) + list(after_data.values())
    vmin = min(all_values)
    vmax = max(all_values)
    
    # Create custom grayscale colormap
    cmap = plt.cm.get_cmap(cmap_name)
    
    # === BEFORE GDPR (left map) ===
    # Assign values
    ireland_before = ireland_norm.copy()
    germany_before = germany_norm.copy()
    switzerland_before = switzerland_norm.copy()
    
    ireland_before['value'] = before_data['Ireland']
    germany_before['value'] = before_data['Germany']
    switzerland_before['value'] = before_data['Switzerland']
    
    # Plot
    ireland_before.plot(column='value', ax=ax1, legend=False,
                        edgecolor='black', linewidth=2,
                        cmap=cmap, vmin=vmin, vmax=vmax)
    germany_before.plot(column='value', ax=ax1, legend=False, 
                        edgecolor='black', linewidth=2, 
                        cmap=cmap, vmin=vmin, vmax=vmax)
    switzerland_before.plot(column='value', ax=ax1, legend=False,
                            edgecolor='black', linewidth=2,
                            cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax1.set_title(f'Before GDPR (2010-2017)\n{metric_name}', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.axis('equal')
    ax1.axis('off')
    
    # Add country labels for before map
    for gdf, country_name in [(ireland_before, 'Ireland'), 
                               (germany_before, 'Germany'),
                               (switzerland_before, 'Switzerland')]:
        centroid = gdf.geometry.union_all().centroid
        value = before_data[country_name]
        # Position label below the country
        ax1.annotate(f"{country_name}\n{value:.1f}{unit_label}", 
                    xy=(centroid.x, -0.15),
                    ha='center', va='top',
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                             edgecolor='black', linewidth=1.5, alpha=0.95))
    
    # === AFTER GDPR (right map) ===
    # Assign values
    ireland_after = ireland_norm.copy()
    germany_after = germany_norm.copy()
    switzerland_after = switzerland_norm.copy()
    
    ireland_after['value'] = after_data['Ireland']
    germany_after['value'] = after_data['Germany']
    switzerland_after['value'] = after_data['Switzerland']
    
    # Plot
    ireland_after.plot(column='value', ax=ax2, legend=False,
                       edgecolor='black', linewidth=2,
                       cmap=cmap, vmin=vmin, vmax=vmax)
    germany_after.plot(column='value', ax=ax2, legend=False,
                       edgecolor='black', linewidth=2,
                       cmap=cmap, vmin=vmin, vmax=vmax)
    switzerland_after.plot(column='value', ax=ax2, legend=False,
                           edgecolor='black', linewidth=2,
                           cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax2.set_title(f'After GDPR (2018+)\n{metric_name}', 
                  fontsize=18, fontweight='bold', pad=20)
    ax2.axis('equal')
    ax2.axis('off')
    
    # Add country labels for after map
    for gdf, country_name in [(ireland_after, 'Ireland'), 
                               (germany_after, 'Germany'),
                               (switzerland_after, 'Switzerland')]:
        centroid = gdf.geometry.union_all().centroid
        value = after_data[country_name]
        # Position label below the country
        ax2.annotate(f"{country_name}\n{value:.1f}{unit_label}", 
                    xy=(centroid.x, -0.15),
                    ha='center', va='top',
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                             edgecolor='black', linewidth=1.5, alpha=0.95))
    
    # Add a single colorbar for both maps
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', 
                       fraction=0.02, pad=0.25, aspect=50)
    cbar.set_label(f'{metric_name} ({unit_label})', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.suptitle(f'Impact of GDPR on {metric_name}', 
                fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    plt.savefig(filename_base, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# Graph 26: Internet Usage Growth Rate
internet_before = {
    'Germany': 0.5,
    'Ireland': 2.7,
    'Switzerland': 1.0
}

internet_after = {
    'Germany': 1.5,
    'Ireland': 2.6,
    'Switzerland': 1.5
}

create_choropleth_comparison(
    'Internet Usage Growth Rate',
    internet_before,
    internet_after,
    'graph_26_choropleth_internet_growth.png',
    '%'
)

# Graph 27: Individual Cloud Adoption Growth Rate
cloud_before = {
    'Germany': 28.0,
    'Ireland': 36.0,
    'Switzerland': 61.0
}

cloud_after = {
    'Germany': 10.0,
    'Ireland': 4.0,
    'Switzerland': 19.0
}

create_choropleth_comparison(
    'Individual Cloud Adoption Growth Rate',
    cloud_before,
    cloud_after,
    'graph_27_choropleth_cloud_adoption_growth.png',
    '%'
)

# Graph 28: Total R&D (GERD) Growth Rate
gerd_before = {
    'Germany': 3.6,
    'Ireland': 2.4,
    'Switzerland': 2.9
}

gerd_after = {
    'Germany': 1.3,
    'Ireland': 15.9,
    'Switzerland': 2.8
}

create_choropleth_comparison(
    'Total R&D (GERD) Growth Rate',
    gerd_before,
    gerd_after,
    'graph_28_choropleth_gerd_growth.png',
    '%'
)

# ============================================================
# 29. EU Regulation Impact Survey Results
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))

# Data from the survey about regulation impacts
regulations_survey = [
    'Payment Services Directive 2 (PSD2)',
    'EU Cybersecurity Act',
    'Visa policy',
    'Digital Markets Act (DMA)',
    'Digital Services Act (DSA)',
    'EU AI Act (AIA)',
    'Anti-trust reviews',
    'Tax reforms',
    'Data privacy laws (e.g. GDPR)'
]

# Percentage values for each category (negative, no significant impact, positive)
negative = [24, 30, 44, 41, 40, 53, 40, 51, 60]
no_impact = [35, 37, 32, 38, 39, 27, 41, 33, 25]
positive = [40, 33, 23, 22, 21, 20, 19, 16, 15]

# Create the stacked bar chart
x = np.arange(len(regulations_survey))
width = 0.6

# Plot bars with grayscale colors and hatching patterns for print clarity
p1 = ax.barh(x, negative, width, label='Negative', 
            color='#333333', edgecolor='black', linewidth=0.8, hatch='///')
p2 = ax.barh(x, no_impact, width, left=negative, label='No significant impact',
            color='#666666', edgecolor='black', linewidth=0.8, hatch='xxx')
p3 = ax.barh(x, positive, width, left=np.array(negative)+np.array(no_impact), 
            label='Positive', color='#CCCCCC', edgecolor='black', linewidth=0.8, hatch='')

# Add percentage labels on bars
for i, (neg, no_imp, pos) in enumerate(zip(negative, no_impact, positive)):
    # Negative label
    if neg > 5:
        ax.text(neg/2, i, f'{neg}%', ha='center', va='center', 
               fontweight='bold', fontsize=10, color='white')
    # No impact label
    if no_imp > 5:
        ax.text(neg + no_imp/2, i, f'{no_imp}%', ha='center', va='center',
               fontweight='bold', fontsize=10, color='white')
    # Positive label
    if pos > 5:
        ax.text(neg + no_imp + pos/2, i, f'{pos}%', ha='center', va='center',
               fontweight='bold', fontsize=10, color='black')

ax.set_yticks(x)
ax.set_yticklabels(regulations_survey, fontsize=11)
ax.set_xlabel('Share of respondents (%)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 105)
ax.set_title('Impact of EU Regulations on Business: Survey Results', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, framealpha=0.9, fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('graph_29_regulation_impact_survey.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 30. Digital Rights and Principles Application Survey
# ============================================================
fig, ax = plt.subplots(figsize=(18, 12))

# Data from the survey about digital rights application
digital_rights = [
    'Freedom of assembly and association\nin digital environment',
    'Freedom of expression and information\nonline (platforms, social networks, search)',
    'Basic and advanced digital education,\ntraining and skills',
    'Access to safe and privacy-friendly\ndigital technologies',
    'Easy online access to all key\npublic services in the EU',
    'Affordable high-speed internet\nconnection for everyone in the EU',
    'Access to trustworthy, diverse and\nmultilingual digital environment',
    'Access to information on environmental\nimpact of digital technologies',
    'Fair and healthy working conditions\nin digital environment (work-life balance)',
    'Privacy online (respect for confidentiality\nof communications and information)',
    'Control of one\'s own data (how it is used\nonline and with whom it is shared)',
    'Effective freedom of choice online\n(including with AI, chatbots, assistants)',
    'Digital products and services that minimise\nenvironmental and social damage'
]

# Percentage values for each category
very_well = [15, 13, 13, 13, 13, 14, 12, 11, 12, 12, 13, 11, 11]
fairly_well = [45, 47, 43, 42, 41, 39, 40, 40, 39, 39, 36, 38, 37]
not_very_well = [18, 21, 24, 25, 25, 26, 25, 26, 26, 27, 28, 23, 26]
not_well_at_all = [5, 6, 6, 7, 7, 9, 7, 7, 7, 9, 11, 7, 8]
dont_know = [17, 13, 14, 13, 14, 12, 16, 16, 16, 13, 12, 21, 18]

# Create the stacked bar chart with increased spacing
x = np.arange(len(digital_rights)) * 1.2
width = 0.7

# Plot bars with grayscale colors and hatching patterns for print clarity
p1 = ax.barh(x, very_well, width, label='Very well', 
            color='#000000', edgecolor='black', linewidth=0.8, hatch='')
p2 = ax.barh(x, fairly_well, width, left=very_well, label='Fairly well',
            color='#444444', edgecolor='black', linewidth=0.8, hatch='///')
p3 = ax.barh(x, not_very_well, width, left=np.array(very_well)+np.array(fairly_well), 
            label='Not very well', color='#888888', edgecolor='black', linewidth=0.8, hatch='xxx')
p4 = ax.barh(x, not_well_at_all, width, 
            left=np.array(very_well)+np.array(fairly_well)+np.array(not_very_well),
            label='Not well at all', color='#CCCCCC', edgecolor='black', linewidth=0.8, hatch='\\\\\\')
p5 = ax.barh(x, dont_know, width,
            left=np.array(very_well)+np.array(fairly_well)+np.array(not_very_well)+np.array(not_well_at_all),
            label='Don\'t know', color='#E8E8E8', edgecolor='black', linewidth=0.8, hatch='...')

# Add percentage labels on bars with larger font and better visibility
for i, (vw, fw, nvw, nwaa, dk) in enumerate(zip(very_well, fairly_well, not_very_well, not_well_at_all, dont_know)):
    y_pos = x[i]
    # Very well
    if vw > 5:
        ax.text(vw/2, y_pos, f'{vw}%', ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    # Fairly well
    if fw > 5:
        ax.text(vw + fw/2, y_pos, f'{fw}%', ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    # Not very well
    if nvw > 5:
        ax.text(vw + fw + nvw/2, y_pos, f'{nvw}%', ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    # Not well at all
    if nwaa > 5:
        ax.text(vw + fw + nvw + nwaa/2, y_pos, f'{nwaa}%', ha='center', va='center', fontsize=13, fontweight='bold', color='black')
    # Don't know
    if dk > 5:
        ax.text(vw + fw + nvw + nwaa + dk/2, y_pos, f'{dk}%', ha='center', va='center', fontsize=13, fontweight='bold', color='black')

ax.set_yticks(x)
ax.set_yticklabels(digital_rights, fontsize=11, linespacing=1.4)
ax.set_xlabel('Share of respondents (%)', fontsize=15, fontweight='bold')
ax.set_xlim(0, 105)
ax.set_ylim(-0.5, max(x) + 0.5)
ax.set_title('Digital Rights Application in Europe: Survey Results (March 2023)', 
            fontsize=17, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5, framealpha=0.9, fontsize=12)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('graph_30_digital_rights_survey.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 31. DMA and DSA Cost Increases by Firm Size
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))

# Data from the survey
firm_sizes = ['0-9\nemployees', '10-19\nemployees', '20-49\nemployees', 
              '50-249\nemployees', '≥250\nemployees', 'Total']
cost_increase = [12495, 4688, 6724, 12114, 33410, 70999]
x_pos = np.arange(len(firm_sizes))

# Create bar chart with different styling for Total
colors = ['#333333'] * 5 + ['#CCCCCC']
hatches = ['///'] * 5 + ['']
edgecolors = ['black'] * 6

bars = ax.bar(x_pos, cost_increase, color=colors, edgecolor=edgecolors, 
              linewidth=1.2, width=0.7)

# Apply hatching
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Add value labels on bars
for i, (pos, val) in enumerate(zip(x_pos, cost_increase)):
    if i < 5:  # Increase values
        ax.text(pos, val/2, f'€ {val:,}', ha='center', va='center', 
                fontsize=13, fontweight='bold', color='white')
    else:  # Total
        ax.text(pos, val/2, f'€ {val:,}', ha='center', va='center', 
                fontsize=13, fontweight='bold', color='black')

ax.set_ylabel('Cost Increase (millions of euros)', fontsize=14, fontweight='bold')
ax.set_xlabel('Firm Size', fontsize=14, fontweight='bold')
ax.set_title('Potential Cost Increases from DMA and DSA on EU Businesses\nUsing U.S. Digital Service Providers (5% Technology Cost Increase)', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(firm_sizes, fontsize=12)
ax.set_ylim(0, 82000)
ax.grid(True, alpha=0.3, axis='y')

# Add legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor='#333333', edgecolor='black', 
                                linewidth=1.2, hatch='///', label='Increase'),
                  plt.Rectangle((0,0),1,1, facecolor='#CCCCCC', edgecolor='black', 
                                linewidth=1.2, label='Total')]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
         ncol=2, framealpha=0.9, fontsize=12)

plt.tight_layout()
plt.savefig('graph_31_dma_dsa_cost_increases.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 32. EU Companies' Actions in Response to Cost Increases
# ============================================================
fig, ax = plt.subplots(figsize=(16, 10))

# Data from the survey (approximate percentages from the chart)
actions = [
    'Have to change to other\npoorer quality technology',
    'Be less competitive in\nexport markets',
    'Pass the costs to customers\nand raise prices',
    'Have to sell more\nproducts/services',
    'Have to have less\ntechnology',
    'Probably have to change to\nChinese technology',
    'Not be able to raise\nour salaries',
    'Hire fewer people',
    'Probably sell less\nespecially online',
    'Slow down our digital\ntransformation plans',
    'Get rid of some\nemployees',
    'No impact',
    'Invest less in R&D'
]

percentages = [29, 28, 27, 24, 23, 18, 17, 17, 14, 12, 10, 10, 10]

# Create horizontal bar chart
y_pos = np.arange(len(actions))
bars = ax.barh(y_pos, percentages, color='#333333', edgecolor='black', 
               linewidth=1.0, height=0.7, hatch='///')

# Add percentage labels
for i, (y, pct) in enumerate(zip(y_pos, percentages)):
    ax.text(pct + 0.5, y, f'{pct}%', va='center', ha='left', 
            fontsize=12, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(actions, fontsize=11)
ax.set_xlabel('Percentage of Companies (%)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 32)
ax.set_title('Likely Actions Taken by EU Companies in Response to\n5-10% Increase in U.S. Digital Service Provider Costs', 
            fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('graph_32_company_actions_cost_response.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 33. European Firms' Response to Tech Cost Increase vs Other Challenges
# ============================================================
fig, ax = plt.subplots(figsize=(18, 10))

# Data from the survey
challenges = [
    'Slowing demand',
    'No issues,\nbusiness is good',
    'Cost of employees',
    'Inflation - higher supply\nand transport costs',
    'Supply chain backlog',
    'Managing new\nregulations',
    'Finding talent'
]

# Percentage values for each category
much_worse = [18, 15, 15, 14, 14, 13, 11]
worse = [30, 32, 32, 28, 28, 27, 26]
less_bad = [16, 19, 18, 21, 20, 22, 24]
irrelevant = [14, 15, 15, 17, 18, 18, 19]

# Create stacked horizontal bar chart
y_pos = np.arange(len(challenges)) * 1.2
width = 0.7

p1 = ax.barh(y_pos, much_worse, width, label='5% tech cost increase would be much worse than this',
            color='#000000', edgecolor='black', linewidth=0.8, hatch='')
p2 = ax.barh(y_pos, worse, width, left=much_worse, 
            label='5% tech cost increase would be worse than this',
            color='#444444', edgecolor='black', linewidth=0.8, hatch='///')
p3 = ax.barh(y_pos, less_bad, width, left=np.array(much_worse)+np.array(worse),
            label='5% tech cost increase would be less bad than this',
            color='#888888', edgecolor='black', linewidth=0.8, hatch='xxx')
p4 = ax.barh(y_pos, irrelevant, width, 
            left=np.array(much_worse)+np.array(worse)+np.array(less_bad),
            label='5% tech cost increase would be pretty irrelevant compared to this',
            color='#CCCCCC', edgecolor='black', linewidth=0.8, hatch='\\\\\\')

# Add percentage labels
for i, (mw, w, lb, ir) in enumerate(zip(much_worse, worse, less_bad, irrelevant)):
    y = y_pos[i]
    if mw > 8:
        ax.text(mw/2, y, f'{mw}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    if w > 8:
        ax.text(mw + w/2, y, f'{w}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    if lb > 8:
        ax.text(mw + w + lb/2, y, f'{lb}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    if ir > 8:
        ax.text(mw + w + lb + ir/2, y, f'{ir}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(challenges, fontsize=12, linespacing=1.3)
ax.set_xlabel('Share of respondents (%)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 105)
ax.set_ylim(-0.5, max(y_pos) + 0.5)
ax.set_title('European Firms\' Response: "How would a 5% increase in tech costs rate\nvis-à-vis other challenges for your company this year?"', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, 
         framealpha=0.9, fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('graph_33_tech_cost_vs_challenges.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 34. Cost-Savings Benefits from Integrated Digital Services by Firm Size
# ============================================================
fig, ax = plt.subplots(figsize=(18, 10))

# Data from the survey
firm_sizes_survey = [
    'Micro\n(<10 employees)',
    'Very Small\n(10-19 employees)',
    'Small\n(20-49 employees)',
    'Medium\n(50-249 employees)',
    'Large\n(≥250 employees)'
]

# Percentage values for each category
extremely_important = [28, 33, 28, 38, 35]
important = [46, 43, 45, 38, 35]
somewhat_important = [11, 11, 14, 12, 12]
unimportant = [4, 3, 4, 3, 3]
inconvenient = [3, 2, 2, 2, 5]

# Create stacked horizontal bar chart
y_pos = np.arange(len(firm_sizes_survey)) * 1.2
width = 0.7

p1 = ax.barh(y_pos, extremely_important, width, label='Extremely important',
            color='#000000', edgecolor='black', linewidth=0.8, hatch='')
p2 = ax.barh(y_pos, important, width, left=extremely_important,
            label='Important',
            color='#444444', edgecolor='black', linewidth=0.8, hatch='///')
p3 = ax.barh(y_pos, somewhat_important, width, 
            left=np.array(extremely_important)+np.array(important),
            label='Somewhat important',
            color='#888888', edgecolor='black', linewidth=0.8, hatch='xxx')
p4 = ax.barh(y_pos, unimportant, width,
            left=np.array(extremely_important)+np.array(important)+np.array(somewhat_important),
            label='Unimportant',
            color='#CCCCCC', edgecolor='black', linewidth=0.8, hatch='\\\\\\')
p5 = ax.barh(y_pos, inconvenient, width,
            left=np.array(extremely_important)+np.array(important)+np.array(somewhat_important)+np.array(unimportant),
            label='Inconvenient, would prefer to not be tied to a single IT provider',
            color='#DDDDDD', edgecolor='black', linewidth=0.8, hatch='...')

# Add percentage labels
for i, (ei, imp, sw, un, inc) in enumerate(zip(extremely_important, important, somewhat_important, unimportant, inconvenient)):
    y = y_pos[i]
    # Extremely important
    if ei > 5:
        ax.text(ei/2, y, f'{ei}%', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    # Important
    if imp > 5:
        ax.text(ei + imp/2, y, f'{imp}%', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    # Somewhat important
    if sw > 5:
        ax.text(ei + imp + sw/2, y, f'{sw}%', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    # Unimportant
    if un > 5:
        ax.text(ei + imp + sw + un/2, y, f'{un}%', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='black')
    # Inconvenient
    if inc > 5:
        ax.text(ei + imp + sw + un + inc/2, y, f'{inc}%', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(firm_sizes_survey, fontsize=13, linespacing=1.3)
ax.set_xlabel('Share of respondents (%)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 105)
ax.set_ylim(-0.5, max(y_pos) + 0.5)
ax.set_title('Percent of European Firms Stating Cost-Savings Benefits from\nIntegrated Digital Services, by Firm Size', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, 
         framealpha=0.9, fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('graph_34_cost_savings_by_firm_size.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 35. Technology Use by European Firms
# ============================================================
fig, ax = plt.subplots(figsize=(16, 10))

# Data from the survey
technologies = [
    'Had internet access',
    'Used a fixed broadband\ninternet connection',
    'Had a website',
    'Connected to internet via\nmobile broadband connection',
    'Used social media',
    'Purchased online',
    'Used cloud computing',
    'Used enterprise resource\nplanning (ERP) software',
    'Sent or received e-invoices',
    'Used customer relationship\nmanagement (CRM) software',
    'Used internet of things (IoT)',
    'Paid to advertise on the internet',
    'Had e-commerce sales',
    'Used radio frequency\nidentification (RFID) technologies',
    'Used AI technologies'
]

percentages = [99, 96, 85, 74, 63, 53, 45, 42, 40, 37, 31, 28, 25, 12, 8]

# Create horizontal bar chart
y_pos = np.arange(len(technologies))
bars = ax.barh(y_pos, percentages, color='#333333', edgecolor='black', 
               linewidth=1.0, height=0.65, hatch='///')

# Add percentage labels
for i, (y, pct) in enumerate(zip(y_pos, percentages)):
    ax.text(pct + 1.5, y, f'{pct}%', va='center', ha='left', 
            fontsize=12, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(technologies, fontsize=11, linespacing=1.3)
ax.set_xlabel('Percentage of Firms (%)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 105)
ax.set_title('Technology Use by European Firms\n(Enterprises with at least 10 employees and self-employed people)', 
            fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('graph_35_technology_use_european_firms.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 36. Priority Digital Services for European Firms (Next 5 Years)
# ============================================================
fig, ax = plt.subplots(figsize=(18, 10))

# Data from the survey
firm_sizes_priority = [
    'Micro\n(<10 employees)',
    'Very Small\n(10-19 employees)',
    'Small\n(20-49 employees)',
    'Medium\n(50-249 employees)',
    'Large\n(≥250 employees)'
]

# Approximate percentages for each category
software_service = [18, 30, 26, 27, 30]
ai_ml = [22, 27, 28, 20, 25]
cloud_computing = [19, 20, 30, 30, 23]
iot = [20, 12, 10, 13, 13]
blockchain = [8, 5, 5, 7, 7]
edge_computing = [5, 3, 2, 2, 1]
other = [8, 3, 1, 1, 1]

# Create stacked horizontal bar chart
y_pos = np.arange(len(firm_sizes_priority)) * 1.2
width = 0.7

p1 = ax.barh(y_pos, software_service, width, label='Software as service',
            color='#000000', edgecolor='black', linewidth=0.8, hatch='')
p2 = ax.barh(y_pos, ai_ml, width, left=software_service,
            label='Artificial intelligence, machine learning',
            color='#CCCCCC', edgecolor='black', linewidth=0.8, hatch='')
p3 = ax.barh(y_pos, cloud_computing, width, 
            left=np.array(software_service)+np.array(ai_ml),
            label='Cloud computing in general',
            color='#888888', edgecolor='black', linewidth=0.8, hatch='///')
p4 = ax.barh(y_pos, iot, width,
            left=np.array(software_service)+np.array(ai_ml)+np.array(cloud_computing),
            label='Internet of things',
            color='#CCCCCC', edgecolor='black', linewidth=0.8, hatch='xxx')
p5 = ax.barh(y_pos, blockchain, width,
            left=np.array(software_service)+np.array(ai_ml)+np.array(cloud_computing)+np.array(iot),
            label='Blockchain',
            color='#CCCCCC', edgecolor='black', linewidth=0.8, hatch='\\\\\\')
p6 = ax.barh(y_pos, edge_computing, width,
            left=np.array(software_service)+np.array(ai_ml)+np.array(cloud_computing)+np.array(iot)+np.array(blockchain),
            label='Edge computing',
            color='#AAAAAA', edgecolor='black', linewidth=0.8, hatch='...')
p7 = ax.barh(y_pos, other, width,
            left=np.array(software_service)+np.array(ai_ml)+np.array(cloud_computing)+np.array(iot)+np.array(blockchain)+np.array(edge_computing),
            label='Other',
            color='#999999', edgecolor='black', linewidth=0.8, hatch='|||')

# Add percentage labels for larger segments
for i, (sws, ai, cc, iot_val, bc, ec, ot) in enumerate(zip(software_service, ai_ml, cloud_computing, iot, blockchain, edge_computing, other)):
    y = y_pos[i]
    if sws > 5:
        ax.text(sws/2, y, f'{sws}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    if ai > 5:
        ax.text(sws + ai/2, y, f'{ai}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='black')
    if cc > 5:
        ax.text(sws + ai + cc/2, y, f'{cc}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    if iot_val > 5:
        ax.text(sws + ai + cc + iot_val/2, y, f'{iot_val}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='black')
    if bc > 5:
        ax.text(sws + ai + cc + iot_val + bc/2, y, f'{bc}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(firm_sizes_priority, fontsize=13, linespacing=1.3)
ax.set_xlabel('Share of respondents (%)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 105)
ax.set_ylim(-0.5, max(y_pos) + 0.5)
ax.set_title('Priority Digital Services for European Firms in the Next Five Years, by Firm Size', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, 
         framealpha=0.9, fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('graph_36_priority_digital_services.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 37. Percent of European Firms Using U.S. Digital Services by Firm Size
# ============================================================
fig, ax = plt.subplots(figsize=(18, 12))

# Data from the survey - services listed in reverse order for bottom-to-top display
services = [
    'Other U.S. technology',
    'Other',
    'Siemens Global Business Services',
    'SAP',
    'Oracle',
    'Salesforce',
    'Slack',
    'Skype for Business',
    'Amazon Cloud - AWS',
    'None of the above',
    'Google Meet',
    'Microsoft 365',
    'OneDrive/Microsoft',
    'Microsoft Cloud',
    'Google Workspace',
    'TikTok',
    'Microsoft Teams',
    'LinkedIn',
    'Microsoft Windows',
    'Zoom',
    'Google Cloud',
    'YouTube',
    'Facebook',
    'Instagram'
]

# Approximate data for each firm size (reading from the chart)
micro = [2, 2, 2, 2, 2, 2, 3, 4, 7, 10, 13, 15, 15, 16, 17, 18, 22, 26, 30, 32, 37, 38, 50, 55]
very_small = [2, 2, 2, 3, 5, 7, 3, 8, 10, 8, 15, 20, 15, 18, 18, 10, 28, 20, 35, 30, 30, 35, 48, 52]
small = [2, 2, 2, 5, 8, 10, 3, 8, 12, 8, 18, 25, 18, 20, 25, 8, 35, 18, 40, 28, 32, 38, 45, 50]
medium = [2, 2, 3, 8, 12, 12, 4, 10, 15, 10, 20, 30, 22, 25, 30, 10, 42, 18, 42, 30, 48, 38, 48, 52]
large = [3, 2, 3, 25, 15, 15, 5, 25, 25, 8, 25, 40, 30, 35, 20, 12, 45, 32, 42, 45, 35, 40, 48, 52]

# Create grouped horizontal bar chart
y_pos = np.arange(len(services)) * 1.3
width = 0.22

b1 = ax.barh(y_pos - 2*width, micro, width, label='Micro (<10 employees)',
            color='#000000', edgecolor='black', linewidth=0.6, hatch='')
b2 = ax.barh(y_pos - width, very_small, width, label='Very small (10-19 employees)',
            color='#333333', edgecolor='black', linewidth=0.6, hatch='///')
b3 = ax.barh(y_pos, small, width, label='Small (20-49 employees)',
            color='#666666', edgecolor='black', linewidth=0.6, hatch='xxx')
b4 = ax.barh(y_pos + width, medium, width, label='Medium (50-249 employees)',
            color='#999999', edgecolor='black', linewidth=0.6, hatch='...')
b5 = ax.barh(y_pos + 2*width, large, width, label='Large (≥ 250 employees)',
            color='#CCCCCC', edgecolor='black', linewidth=0.6, hatch='\\\\')

ax.set_yticks(y_pos)
ax.set_yticklabels(services, fontsize=10)
ax.set_xlabel('Percentage of Firms (%)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 62)
ax.set_title('Percent of European Firms Using U.S. Digital Services, by Firm Size', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5, 
         framealpha=0.9, fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('graph_37_us_digital_services_by_firm_size.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 38. Benefits from U.S. Digital Services by Firm Age
# ============================================================
fig, ax = plt.subplots(figsize=(18, 10))

# Data from the survey
benefit_categories = [
    'Security/cybersecurity',
    'Performance',
    'Price',
    'Ability to manage\ndata privacy',
    'Integration with work\nand solutions',
    'Efficiencies to reduce\ncarbon footprint',
    'Ability to innovate faster',
    'Ease of use'
]

# Approximate percentages for each firm age category
less_1year = [36, 36, 32, 30, 26, 22, 22, 18]
one_3years = [34, 36, 38, 34, 34, 32, 28, 38]
three_5years = [32, 34, 32, 34, 32, 34, 32, 36]
five_10years = [32, 34, 30, 30, 32, 28, 32, 32]
more_10years = [34, 36, 32, 34, 32, 30, 32, 34]

# Create grouped horizontal bar chart
y_pos = np.arange(len(benefit_categories)) * 1.5
width = 0.28

b1 = ax.barh(y_pos - 2*width, less_1year, width, label='<1 year',
            color='#000000', edgecolor='black', linewidth=0.7, hatch='')
b2 = ax.barh(y_pos - width, one_3years, width, label='1-3 years',
            color='#444444', edgecolor='black', linewidth=0.7, hatch='///')
b3 = ax.barh(y_pos, three_5years, width, label='3-5 years',
            color='#888888', edgecolor='black', linewidth=0.7, hatch='xxx')
b4 = ax.barh(y_pos + width, five_10years, width, label='5-10 years',
            color='#AAAAAA', edgecolor='black', linewidth=0.7, hatch='...')
b5 = ax.barh(y_pos + 2*width, more_10years, width, label='>10 years',
            color='#CCCCCC', edgecolor='black', linewidth=0.7, hatch='\\\\')

ax.set_yticks(y_pos)
ax.set_yticklabels(benefit_categories, fontsize=13, linespacing=1.3)
ax.set_xlabel('Percentage of Firms Classifying as "Extremely Great" Advantage (%)', 
             fontsize=14, fontweight='bold')
ax.set_xlim(0, 42)
ax.set_ylim(-0.8, max(y_pos) + 0.8)
ax.set_title('Percent of European Firms Classifying Benefits from U.S. Digital Services as\n"Extremely Great" Advantage for Their Business, by Firm Age', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, 
         framealpha=0.9, fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('graph_38_benefits_by_firm_age.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 39. EU Regulations Cost Increases on U.S. Businesses by Sector
# ============================================================
fig, ax = plt.subplots(figsize=(16, 12))

# Data from the survey - sectors listed in reverse order for bottom-to-top display
sectors = [
    'Total',
    'Other services, except government',
    'Arts, entertainment, recreation,\naccommodation, and food services',
    'Educational services, healthcare,\nand social assistance',
    'Professional and business services',
    'Finance, insurance, real estate,\nrental, and leasing',
    'Information',
    'Transportation and warehousing',
    'Retail trade',
    'Wholesale trade',
    'Manufacturing',
    'Construction',
    'Utilities',
    'Mining',
    'Agriculture, forestry, fishing,\nand hunting'
]

# Cost values in millions
cost_increase = [275, 284, 331, 955, 6187, 6477, 7381, 3811, 17337, 27742, 16024, 6200, 2421, 1138, 96561]
total_cost = [96561] + [0] * 14  # Only total has a total value shown

# Create horizontal bar chart
y_pos = np.arange(len(sectors)) * 1.1
width = 0.65

# Different styling for Total vs sector bars
colors = ['#CCCCCC'] + ['#333333'] * 14
hatches = [''] + ['///'] * 14

bars = ax.barh(y_pos, cost_increase, width, color=colors, edgecolor='black', 
               linewidth=1.0)

# Apply hatching
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Add value labels on bars
for i, (y, val) in enumerate(zip(y_pos, cost_increase)):
    if i == 0:  # Total
        ax.text(val + 2000, y, f'${val:,}', va='center', ha='left', 
                fontsize=12, fontweight='bold', color='black')
    else:  # Sectors
        if val > 5000:
            ax.text(val/2, y, f'${val:,}', va='center', ha='center', 
                    fontsize=11, fontweight='bold', color='white')
        else:
            ax.text(val + 400, y, f'${val:,}', va='center', ha='left', 
                    fontsize=11, fontweight='bold', color='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(sectors, fontsize=11, linespacing=1.3)
ax.set_xlabel('Cost Increase (millions of dollars)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 110000)
ax.set_ylim(-0.6, max(y_pos) + 0.6)
ax.set_title('Potential Cost Increases Implied by EU Regulations on U.S. Businesses\nThat Use U.S. Digital Service Providers, by Sector', 
            fontsize=16, fontweight='bold', pad=20)

# Add legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor='#333333', edgecolor='black', 
                                linewidth=1.0, hatch='///', label='Increase'),
                  plt.Rectangle((0,0),1,1, facecolor='#CCCCCC', edgecolor='black', 
                                linewidth=1.0, label='Total')]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), 
         ncol=2, framealpha=0.9, fontsize=12)

ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('graph_39_eu_regulations_cost_by_sector.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 40. Notifications Assessed Within 24 Hours - Trend Over Time
# ============================================================
fig, ax = plt.subplots(figsize=(18, 10))

# Data from the chart - monitoring periods
periods = ['DEC 2016', 'MAY 2017', 'JAN 2018', 
           'FEB 2019', 'JUN 2020', 'OCT 2021', 
           'NOV 2022']

# Approximate percentage values for each platform
facebook_vals = [50, 55, 60, 82, 95, 85, 82]
youtube_vals = [63, 50, 90, 82, 90, 82, 96]
twitter_vals = [25, 40, 82, 82, 82, 78, 65]
instagram_vals = [0, 0, 0, 78, 90, 63, 52]
tiktok_vals = [0, 0, 0, 0, 0, 0, 0]  # No data shown for TikTok in early periods
average_vals = [42, 47, 70, 85, 92, 82, 72]

# Create line chart with different styles for each platform
ax.plot(periods, facebook_vals, color='#000000', linewidth=3, 
        linestyle='-', marker='o', markersize=8, label='Facebook')
ax.plot(periods, youtube_vals, color='#333333', linewidth=3, 
        linestyle='--', marker='s', markersize=8, label='YouTube')
ax.plot(periods, twitter_vals, color='#555555', linewidth=3, 
        linestyle='-.', marker='^', markersize=8, label='Twitter')
ax.plot(periods, instagram_vals, color='#777777', linewidth=3, 
        linestyle=':', marker='D', markersize=8, label='Instagram')
ax.plot(periods, average_vals, color='#AAAAAA', linewidth=4, 
        linestyle='-', marker='*', markersize=10, label='Average of companies')

ax.set_xlabel('Monitoring Period', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage of notifications assessed within 24 hours (%)', fontsize=14, fontweight='bold')
ax.set_title('Percentage of Notifications Assessed Within 24 Hours - Trend Over Time', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 105)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5, 
         framealpha=0.9, fontsize=12)
ax.grid(True, alpha=0.3)

# Add GDPR implementation marker
gdpr_position = 2.5  # Between 3rd and 4th monitoring
ax.axvline(x=gdpr_position, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.text(gdpr_position, 102, 'GDPR\nImplemented', ha='center', va='top', 
        fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', 
        facecolor='white', edgecolor='black', linewidth=1))

plt.tight_layout()
plt.savefig('graph_40_notifications_24h_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 41. Social Media Platform Compliance Monitoring Over Time (Top 3 Platforms)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))

# Data from monitoring reports (percentage values)
monitoring_dates = ['Dec 2016', 'May 2017', 'Jan 2018', 'Feb 2019', 'Jun 2020', 'Oct 2021', 'Nov 2022']
monitoring_periods = [1, 2, 3, 4, 5, 6, 7]

# Only include Twitter, YouTube, and Facebook
facebook_data = [28.3, 66.5, 79.8, 82.4, 87.6, 70.2, 69.1]
youtube_data = [48.5, 66.0, 75.0, 85.4, 79.7, 58.8, 90.4]
twitter_data = [19.1, 37.4, 45.7, 43.5, 35.9, 49.8, 45.4]

platforms = {
    'Facebook': facebook_data,
    'YouTube': youtube_data,
    'Twitter': twitter_data
}

platform_colors = {
    'Facebook': '#000000',
    'YouTube': '#444444',
    'Twitter': '#888888'
}

platform_markers = {
    'Facebook': 'o',
    'YouTube': 's',
    'Twitter': '^'
}

platform_linestyles = {
    'Facebook': '-',
    'YouTube': '--',
    'Twitter': '-.'
}

for platform, platform_data in platforms.items():
    ax.plot(monitoring_periods, platform_data,
            color=platform_colors[platform],
            linestyle=platform_linestyles[platform],
            marker=platform_markers[platform],
            markersize=10,
            linewidth=2.5,
            label=platform,
            markeredgecolor='black',
            markeredgewidth=0.8)

# Add GDPR regulation line
gdpr_period = 2.5  # Between monitoring 2 and 3 (May 2017 and Jan 2018)
ax.axvline(x=gdpr_period, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax.text(gdpr_period, ax.get_ylim()[1] * 0.95, 'GDPR\n(May 2018)', 
        rotation=0, verticalalignment='top', horizontalalignment='center',
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', 
        facecolor='white', edgecolor='black', linewidth=1))

ax.set_xlabel('Monitoring Period', fontsize=14, fontweight='bold')
ax.set_ylabel('Compliance Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Social Media Platform Compliance Monitoring Over Time\n(Facebook, YouTube, Twitter)', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(monitoring_periods)
ax.set_xticklabels(monitoring_dates, rotation=45, ha='right')
ax.set_ylim(0, 105)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.9, fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_41_social_media_compliance_monitoring.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 42. VPN Searches in Texas with House Bill 1181
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
df_vpn = data['vpn_searches']

# Convert Week column to datetime
df_vpn['Week'] = pd.to_datetime(df_vpn['Week'])

# Plot VPN searches
ax.plot(df_vpn['Week'], df_vpn['VPN Searches'],
        color='#000000',
        linestyle='-',
        marker='o',
        markersize=8,
        linewidth=2.5,
        label='VPN Searches',
        markeredgecolor='black',
        markeredgewidth=0.8)

# Calculate and plot mean line
mean_value = df_vpn['VPN Searches'].mean()
ax.axhline(y=mean_value, color='#666666', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean ({mean_value:.1f})')

# Add House Bill 1181 marker on September 1, 2023
hb1181_date = datetime(2023, 9, 1)
ax.axvline(x=hb1181_date, color='#444444', linestyle='--', linewidth=2.5, alpha=0.8)
ax.text(hb1181_date, ax.get_ylim()[1] * 0.95, 'House Bill 1181\n(Sept 1, 2023)', 
        rotation=0, verticalalignment='top', horizontalalignment='center',
        fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor='black', linewidth=1.5))

ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylabel('Search Interest (Normalized)', fontsize=14, fontweight='bold')
ax.set_title('VPN-Related Search Interest in Texas with House Bill 1181', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('graph_42_vpn_searches_texas.png', dpi=300, bbox_inches='tight')
plt.close()


print("All graphs completed successfully!")