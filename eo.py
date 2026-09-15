import os
import rasterio
from rasterio.env import Env
import numpy as np
from rasterio.warp import transform
from rasterio.windows import Window
from pystac_client import Client
import matplotlib.pyplot as plt

# Fast GDAL network streaming settings
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_DATA_FOR_ALL_DAILY_FILES": "YES",
    "VSI_CACHE": "TRUE",
}


def get_satellite_image(latitude, longitude):
    catalog = Client.open("https://earth-search.aws.element84.com/v1")

    # Fetch the single best/most recent image matching criteria
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [longitude, latitude]},
        datetime="2025-01-01/2026-12-31",
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=1
    )

    items = list(search.items())
    if not items:
        raise Exception("No suitable low-cloud satellite data found for these coordinates.")

    item = items[0]
    start_time = item.properties.get("datetime")
    assets = item.assets

    red_url = assets["red"].href
    green_url = assets["green"].href
    blue_url = assets["blue"].href

    os.makedirs("static/outputs", exist_ok=True)

    crop_size = 500
    half = crop_size // 2

    # Stream pixels directly over HTTPS
    with Env(**GDAL_ENV):
        with rasterio.open(red_url) as src:
            x, y = transform("EPSG:4326", src.crs, [longitude], [latitude])
            row, col = src.index(x[0], y[0])

            row_start = max(0, row - half)
            row_end = min(src.height, row + half)
            col_start = max(0, col - half)
            col_end = min(src.width, col + half)

            window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
            red = src.read(1, window=window)

        with rasterio.open(green_url) as src:
            green = src.read(1, window=window)

        with rasterio.open(blue_url) as src:
            blue = src.read(1, window=window)

    # RGB composite assembly and contrast enhancement
    rgb_crop = np.dstack((red, green, blue)).astype(float)
    rgb_display = rgb_crop.copy()
    rgb_display[rgb_display < 0] = 0

    valid = rgb_display[rgb_display > 0]
    if len(valid) == 0:
        raise Exception("No valid imagery pixels found in this crop window.")

    low, high = np.percentile(valid, (2, 98))
    if high <= low:
        raise Exception("Unable to enhance image: insufficient pixel contrast.")

    rgb_display = np.clip((rgb_display - low) / (high - low), 0, 1)
    rgb_display = np.power(rgb_display, 0.7)

    output_file = "static/outputs/sentinel_image.png"

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_display)
    plt.axis("off")
    plt.savefig(output_file, dpi=120, bbox_inches="tight")
    plt.close()

    return {
        "image": "/static/outputs/sentinel_image.png",
        "observation": start_time,
        "granule": item.id,
        "latitude": latitude,
        "longitude": longitude
    }