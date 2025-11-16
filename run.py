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

print("All graphs completed successfully!")
