"""Download the raw data from the original sources and create the cleaned data sets in this folder.

Run from the repository root:  python Data/prepare_data.py

Created files
- zurich_bike_counts_2024_2025.csv   daily bike counts at 16 counting stations in Zurich + weather + calendar
- avalanche_accidents_switzerland.csv avalanche accidents in Switzerland since 1970/71 (SLF)
- shark_attacks_2000_2025.csv        shark incidents worldwide 2000-2025 (Global Shark Attack File)
- titanic.csv                        Titanic passengers (Kaggle)

apartments_data_enriched_cleaned.csv (ZHAW course data) is not downloaded; it is part of the repository.
"""
import io
import json
import os
import re
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'ml_basics_raw_data')
os.makedirs(CACHE_DIR, exist_ok=True)


def download(url, filename):
    """Download a file into the cache directory (only once) and return its path."""
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print('Downloading', url)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, 'wb') as f:
            f.write(r.read())
    return path


# --------------------------------------------------------------------------------------------------
# 1) Zurich bike counts 2024-2025 (Open Data Stadt Zurich, CC0)
# --------------------------------------------------------------------------------------------------
BIKE_URL = ('https://data.stadt-zuerich.ch/dataset/ted_taz_verkehrszaehlungen_werte_fussgaenger_velo/'
            'download/{year}_verkehrszaehlungen_werte_fussgaenger_velo.csv')
METEO_URL = ('https://data.stadt-zuerich.ch/dataset/ugz_meteodaten_tagesmittelwerte/download/'
             'ugz_ogd_meteo_d1_{year}.csv')
STATIONS_URL = ('https://www.ogd.stadt-zuerich.ch/wfs/geoportal/Standorte_der_automatischen_Fuss__und_Velozaehlungen'
                '?service=WFS&version=1.1.0&request=GetFeature&outputFormat=GeoJSON&typename=view_eco_standorte')

# Counting stations with at least 350 complete days in 2024 and in 2025 (FK_STANDORT -> column name)
BIKE_STATIONS = {
    2989: 'langstrasse_underpass_north', 4249: 'langstrasse_underpass_south', 1037: 'hardbruecke_south',
    4267: 'limmatquai_central', 4266: 'bertastrasse', 4270: 'muehlebachstrasse', 5002: 'mythenquai',
    3012: 'bucheggplatz', 732: 'hardbruecke_north', 2997: 'lux_guyer_weg', 4242: 'langstrasse_road_north',
    4258: 'hofwiesenstrasse', 4241: 'langstrasse_road_south', 4262: 'militaerbruecke',
    4269: 'tannenrauchstrasse', 4255: 'toedistrasse',
}

# Public holidays in the city of Zurich
HOLIDAYS = ['2024-01-01', '2024-01-02', '2024-03-29', '2024-04-01', '2024-05-01', '2024-05-09', '2024-05-20',
            '2024-08-01', '2024-12-25', '2024-12-26',
            '2025-01-01', '2025-01-02', '2025-04-18', '2025-04-21', '2025-05-01', '2025-05-29', '2025-06-09',
            '2025-08-01', '2025-12-25', '2025-12-26']

# School holidays of the Volksschule Stadt Zurich (first and last day, incl. weekends in between)
SCHOOL_HOLIDAYS = [('2023-12-23', '2024-01-07'), ('2024-02-10', '2024-02-25'), ('2024-04-20', '2024-05-05'),
                   ('2024-07-13', '2024-08-18'), ('2024-10-05', '2024-10-20'), ('2024-12-21', '2025-01-05'),
                   ('2025-02-08', '2025-02-23'), ('2025-04-17', '2025-05-04'), ('2025-07-12', '2025-08-17'),
                   ('2025-10-04', '2025-10-19'), ('2025-12-20', '2026-01-04')]


def prepare_bike_counts():
    # Daily counts per station (only days with >= 92 of 96 quarter hours)
    daily = []
    for year in [2024, 2025]:
        v = pd.read_csv(download(BIKE_URL.format(year=year), f'bike_{year}.csv'),
                        usecols=['FK_STANDORT', 'DATUM', 'VELO_IN', 'VELO_OUT'])
        v = v[v['FK_STANDORT'].isin(BIKE_STATIONS)]
        v['bikes'] = v[['VELO_IN', 'VELO_OUT']].sum(axis=1, min_count=1)
        v = v[v['bikes'].notna()]
        v['date'] = v['DATUM'].str[:10]
        daily.append(v.groupby(['date', 'FK_STANDORT']).agg(bikes=('bikes', 'sum'), n=('bikes', 'size')).reset_index())
    daily = pd.concat(daily)
    daily = daily[(daily['n'] >= 92) & (daily['bikes'] > 0)]      # a daily count of 0 means a counter failure
    counts = (daily.pivot(index='date', columns='FK_STANDORT', values='bikes')
              .rename(columns=BIKE_STATIONS)[list(BIKE_STATIONS.values())]
              .dropna().astype(int))

    # Weather (station Stampfenbachstrasse, missing values filled with station Schimmelstrasse)
    meteo = pd.concat([pd.read_csv(download(METEO_URL.format(year=y), f'meteo_{y}.csv')) for y in [2024, 2025]])
    meteo['date'] = meteo['Datum'].str[:10]
    weather = {}
    for station in ['Zch_Stampfenbachstrasse', 'Zch_Schimmelstrasse']:
        weather[station] = meteo[meteo['Standort'] == station].pivot_table(index='date', columns='Parameter',
                                                                           values='Wert')
    w = weather['Zch_Stampfenbachstrasse'].combine_first(weather['Zch_Schimmelstrasse'])
    w = w.rename(columns={'T': 'temp_mean', 'T_max_h1': 'temp_max', 'RainDur': 'rain_duration_min',
                          'StrGlo': 'solar_radiation', 'p': 'pressure'})
    w = w[['temp_mean', 'temp_max', 'rain_duration_min', 'solar_radiation', 'pressure']].round(1)

    # Calendar
    df = pd.DataFrame(index=counts.index)
    dates = pd.to_datetime(df.index)
    df['year'] = dates.year
    df['month'] = dates.month
    df['weekday'] = dates.day_name()
    df['weekend'] = (dates.dayofweek >= 5).astype(int)
    df['holiday'] = df.index.isin(HOLIDAYS).astype(int)
    df['school_holiday'] = 0
    for start, end in SCHOOL_HOLIDAYS:
        df.loc[(df.index >= start) & (df.index <= end), 'school_holiday'] = 1
    df = df.join(w, how='left')
    df['bikes_total'] = counts.sum(axis=1)
    df = df.join(counts)
    df = df.dropna()
    df.index.name = 'date'
    df.to_csv(os.path.join(DATA_DIR, 'zurich_bike_counts_2024_2025.csv'))
    print('zurich_bike_counts_2024_2025.csv:', df.shape)

    # Station list (name, direction) for documentation
    stations = pd.DataFrame([f['properties'] for f in
                             json.load(open(download(STATIONS_URL, 'bike_stations.json')))['features']])
    stations = stations.drop_duplicates('id1').set_index('id1').reindex(list(BIKE_STATIONS))
    stations = pd.DataFrame({'column': list(BIKE_STATIONS.values()), 'station_id': list(BIKE_STATIONS),
                             'name': stations['bezeichnung'].values,
                             'direction_in': stations['richtung_in'].values,
                             'direction_out': stations['richtung_out'].values})
    stations.to_csv(os.path.join(DATA_DIR, 'zurich_bike_stations.csv'), index=False)
    print('zurich_bike_stations.csv:', stations.shape)


# --------------------------------------------------------------------------------------------------
# 2) Avalanche accidents in Switzerland since 1970/71 (WSL Institute for Snow and Avalanche Research SLF)
# --------------------------------------------------------------------------------------------------
AVALANCHE_URL = ('https://www.envidat.ch/dataset/aa035efb-630a-4b7f-a406-f7a579a74de9/resource/'
                 '944beac1-11d1-4c84-9bd9-683d12a3c581/download/version2_avalanche_accidents_all_switzerland_since_1970.csv')


def prepare_avalanches():
    path = download(AVALANCHE_URL, 'avalanche_accidents.csv')
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    header_row = [i for i, line in enumerate(lines) if 'avalanche.id' in line][0]
    names = lines[header_row].strip('"').replace('""', '').split(',')
    a = pd.read_csv(path, skiprows=header_row + 1, header=None, names=names)
    a = a[a['date'].notna()].copy()
    a.columns = [c.replace('start.zone.', '').replace('forecasted.', '').replace('.', '_') for c in a.columns]
    a = a.rename(columns={'coordinates_latitude': 'latitude', 'coordinates_longitude': 'longitude',
                          'dangerlevel_rating1': 'danger_level', 'dangerlevel_rating1_subdivision': 'danger_level_sub',
                          'dangerlevel_rating2': 'danger_level_2', 'dangerlevel_rating2_subdivision': 'danger_level_2_sub',
                          'most_dangerous_aspect_and_elevation': 'in_core_zone', 'slope_aspect': 'aspect',
                          'number_dead': 'dead', 'number_caught': 'caught', 'number_fully_buried': 'fully_buried'})
    a = a.drop(columns=['avalanche_id'])
    a['month'] = pd.to_datetime(a['date']).dt.month
    a['fatal'] = (a['dead'] > 0).astype(int)
    a.to_csv(os.path.join(DATA_DIR, 'avalanche_accidents_switzerland.csv'), index=False)
    print('avalanche_accidents_switzerland.csv:', a.shape)


# --------------------------------------------------------------------------------------------------
# 3) Shark incidents 2000-2025 (Global Shark Attack File, www.sharkattackfile.net)
# --------------------------------------------------------------------------------------------------
GSAF_URL = 'https://www.sharkattackfile.net/spreadsheets/GSAF5.xls'

ACTIVITY_GROUPS = [('spearfishing', r'spear'),
                   ('surfing', r'surf|board|kite|foil|paddl|sup\b|skim'),
                   ('diving / snorkeling', r'div|snorkel|scuba'),
                   ('fishing', r'fish|net|line|hook|lobster|crab|oyster|shell'),
                   ('swimming', r'swim|bath|float|treading'),
                   ('wading / standing', r'wad|stand|walk|play|jump|splash'),
                   ('kayaking / boating', r'kayak|canoe|boat|row|sail|raft')]
SPECIES_GROUPS = [('white shark', r'white'), ('tiger shark', r'tiger'), ('bull shark', r'bull'),
                  ('blacktip / spinner shark', r'blacktip|black tip|spinner'), ('nurse shark', r'nurse'),
                  ('other identified species', r'hammerhead|mako|lemon|reef|wobbegong|bronze|sand|blue|dusky|'
                                               r'whaler|cookiecutter|galapagos|shortfin|copper|seven')]


def group_text(text, groups, default):
    text = str(text).lower()
    for name, pattern in groups:
        if re.search(pattern, text):
            return name
    return default


def parse_hour(text):
    m = re.search(r'(\d{1,2})\s*h', str(text).lower())
    if m and int(m.group(1)) < 24:
        return int(m.group(1))
    return np.nan


def prepare_sharks():
    g = pd.read_excel(download(GSAF_URL, 'GSAF5.xls'))
    g.columns = g.columns.str.strip()
    g['Year'] = pd.to_numeric(g['Year'], errors='coerce')
    g = g[(g['Year'] >= 2000) & (g['Year'] <= 2025)].copy()
    g['fatal'] = g['Fatal Y/N'].astype(str).str.strip().str.upper().map({'Y': 1, 'N': 0})
    g['type'] = g['Type'].astype(str).str.strip()
    g = g[g['fatal'].notna() & g['type'].isin(['Unprovoked', 'Provoked', 'Watercraft', 'Sea Disaster'])]
    date = pd.to_datetime(g['Date'].astype(str), errors='coerce', format='mixed', dayfirst=True)
    out = pd.DataFrame({
        'year': g['Year'].astype(int),
        'month': date.dt.month,
        'country': g['Country'].astype(str).str.strip().str.title(),
        'area': g['State'].astype(str).str.strip().replace('nan', np.nan),
        'type': g['type'],
        'activity': g['Activity'].astype(str).str.strip().replace('nan', np.nan),
        'sex': g['Sex'].astype(str).str.strip().str.upper().where(lambda s: s.isin(['M', 'F'])),
        'age': pd.to_numeric(g['Age'], errors='coerce'),
        'hour': g['Time'].map(parse_hour),
        'species': g['Species'].astype(str).str.strip().replace('nan', np.nan),
        'injury': g['Injury'].astype(str).str.strip(),
        'fatal': g['fatal'].astype(int),
    })
    out['activity_group'] = out['activity'].map(lambda t: group_text(t, ACTIVITY_GROUPS, 'other'))
    out['species_group'] = out['species'].map(lambda t: group_text(t, SPECIES_GROUPS, 'unknown'))
    out = out.sort_values(['year', 'month'], na_position='first').reset_index(drop=True)
    out.to_csv(os.path.join(DATA_DIR, 'shark_attacks_2000_2025.csv'), index=False)
    print('shark_attacks_2000_2025.csv:', out.shape)


# --------------------------------------------------------------------------------------------------
# 4) Titanic (Kaggle)
# --------------------------------------------------------------------------------------------------
TITANIC_URL = 'https://www.kaggle.com/api/v1/datasets/download/yasserh/titanic-dataset'


def prepare_titanic():
    with zipfile.ZipFile(download(TITANIC_URL, 'titanic.zip')) as z:
        t = pd.read_csv(io.BytesIO(z.read('Titanic-Dataset.csv')))
    t.to_csv(os.path.join(DATA_DIR, 'titanic.csv'), index=False)
    print('titanic.csv:', t.shape)


if __name__ == '__main__':
    prepare_bike_counts()
    prepare_avalanches()
    prepare_sharks()
    prepare_titanic()
