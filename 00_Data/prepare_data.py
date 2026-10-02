"""Download the raw data from the original sources and create the cleaned data sets in this folder.

Run from the repository root:  python 00_Data/prepare_data.py

Created files
- car_fuel_consumption_2025.csv      fuel consumption of new car models, model year 2025 (Natural Resources Canada)
- avalanche_accidents_switzerland.csv avalanche accidents in Switzerland since 1970/71 (SLF)
- titanic.csv                        Titanic passengers (Kaggle)

rental_apartments_canton_zh.csv (ZHAW course data) is not downloaded; it is part of the repository.

The CSV files in this folder are the reference versions used in the notebooks. The original sources may change or
become unavailable; in that case the script reports it and keeps the existing file.
"""
import io
import json
import os
import tempfile
import urllib.request
import zipfile

import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'ml_basics_raw_data')
os.makedirs(CACHE_DIR, exist_ok=True)


def download(url, filename):
    """Download a file into the cache directory (only once) and return its path."""
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print('Downloading', url)
        req = urllib.request.Request(url, headers={'User-Agent': 'ml-basics-course-data-script'})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, 'wb') as f:
            f.write(r.read())
    return path


# --------------------------------------------------------------------------------------------------
# 1) Fuel consumption ratings of new cars, model year 2025 (Natural Resources Canada,
#    Open Government Licence - Canada, https://open.canada.ca/en/open-government-licence-canada)
# --------------------------------------------------------------------------------------------------
FCR_API = ('https://open.canada.ca/data/api/action/package_show?id=98f1a129-f628-4ce4-b24d-6f16bf24dd64')

FUEL_TYPES = {'X': 'regular petrol', 'Z': 'premium petrol', 'D': 'diesel', 'E': 'E85'}
TRANSMISSION_TYPES = {'A': 'automatic', 'AM': 'automated manual', 'AS': 'automatic with select shift',
                      'AV': 'continuously variable', 'M': 'manual'}


def prepare_cars():
    # Look up the current download link of the English CSV file "2025 Fuel Consumption Ratings"
    package = json.load(open(download(FCR_API, 'fcr_package.json')))['result']
    url = [r['url'] for r in package['resources']
           if r['name'].startswith('2025 Fuel Consumption Ratings') and 'en' in r.get('language', [])][0]
    c = pd.read_csv(download(url, 'fcr_2025.csv'), encoding='latin-1')
    c.columns = ['model_year', 'make', 'model', 'vehicle_class', 'engine_size', 'cylinders', 'transmission',
                 'fuel_type', 'city', 'highway', 'combined', 'combined_mpg', 'co2', 'co2_rating', 'smog_rating']
    c['fuel_type'] = c['fuel_type'].map(FUEL_TYPES)
    c['transmission_type'] = c['transmission'].str.extract(r'^([A-Z]+)')[0].map(TRANSMISSION_TYPES)
    c['km_per_l'] = (100 / c['combined']).round(2)
    c = c[['make', 'model', 'vehicle_class', 'engine_size', 'cylinders', 'transmission_type',
           'fuel_type', 'city', 'highway', 'combined', 'km_per_l', 'co2', 'co2_rating', 'smog_rating']]
    c.to_csv(os.path.join(DATA_DIR, 'car_fuel_consumption_2025.csv'), index=False)
    print('car_fuel_consumption_2025.csv:', c.shape)


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
# 3) Titanic (Kaggle)
# --------------------------------------------------------------------------------------------------
TITANIC_URL = 'https://www.kaggle.com/api/v1/datasets/download/yasserh/titanic-dataset'


def prepare_titanic():
    with zipfile.ZipFile(download(TITANIC_URL, 'titanic.zip')) as z:
        t = pd.read_csv(io.BytesIO(z.read('Titanic-Dataset.csv')))
    t.to_csv(os.path.join(DATA_DIR, 'titanic.csv'), index=False)
    print('titanic.csv:', t.shape)


if __name__ == '__main__':
    # Each source is prepared separately: if one source is not reachable, the others are still prepared and the
    # existing CSV file of the unreachable source in this folder stays unchanged.
    for prepare in [prepare_cars, prepare_avalanches, prepare_titanic]:
        try:
            prepare()
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
            print(f'{prepare.__name__}: source not reachable or changed ({e}); the existing CSV file is kept.')
