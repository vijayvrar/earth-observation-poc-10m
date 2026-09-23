import os

import rasterio
from rasterio.env import Env
import numpy as np
from rasterio.warp import transform
from rasterio.windows import Window
from pystac_client import Client

from PIL import Image, ImageFilter


# ============================================================
# GDAL network streaming settings
# ============================================================

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_DATA_FOR_ALL_DAILY_FILES": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
    "GDAL_CACHEMAX": 32,
}


# ============================================================
# CREATE RGB IMAGE FROM A SENTINEL-2 ITEM
# ============================================================

def create_rgb_image(
    item,
    latitude,
    longitude,
    output_file
):

    assets = item.assets

    red_url = assets["red"].href
    green_url = assets["green"].href
    blue_url = assets["blue"].href

    os.makedirs(
        "static/outputs",
        exist_ok=True
    )

    # --------------------------------------------------------
    # 800 x 800 Sentinel-2 pixels
    #
    # Sentinel-2 RGB resolution = 10 m
    #
    # 800 x 10 m = approximately 8 km
    # --------------------------------------------------------

    crop_size = 500
    half = crop_size // 2


    # ========================================================
    # Read Red, Green and Blue bands
    # ========================================================

    with Env(**GDAL_ENV):

        # ----------------------------------------------------
        # RED
        # ----------------------------------------------------

        with rasterio.open(red_url) as src:

            # Convert latitude/longitude
            # from WGS84 into the raster CRS

            x, y = transform(
                "EPSG:4326",
                src.crs,
                [longitude],
                [latitude]
            )

            # Convert projected coordinates
            # into row/column pixel coordinates

            row, col = src.index(
                x[0],
                y[0]
            )

            # ------------------------------------------------
            # Create crop window
            # ------------------------------------------------

            row_start = max(
                0,
                row - half
            )

            row_end = min(
                src.height,
                row + half
            )

            col_start = max(
                0,
                col - half
            )

            col_end = min(
                src.width,
                col + half
            )

            window = Window(
                col_start,
                row_start,
                col_end - col_start,
                row_end - row_start
            )

            red = src.read(
                1,
                window=window
            )


        # ----------------------------------------------------
        # GREEN
        # ----------------------------------------------------

        with rasterio.open(green_url) as src:

            green = src.read(
                1,
                window=window
            )


        # ----------------------------------------------------
        # BLUE
        # ----------------------------------------------------

        with rasterio.open(blue_url) as src:

            blue = src.read(
                1,
                window=window
            )


    # ========================================================
    # Combine RGB bands
    # ========================================================

    rgb_crop = np.dstack(
        (
            red,
            green,
            blue
        )
    ).astype(float)


    rgb_display = rgb_crop.copy()


    # Remove invalid negative values

    rgb_display[
        rgb_display < 0
    ] = 0


    # ========================================================
    # Find valid pixels
    # ========================================================

    valid = rgb_display[
        rgb_display > 0
    ]


    if len(valid) == 0:

        raise Exception(
            "No valid imagery pixels found "
            "in this crop window."
        )


    # ========================================================
    # Contrast enhancement
    # ========================================================

    low, high = np.percentile(
        valid,
        (2, 98)
    )


    if high <= low:

        raise Exception(
            "Unable to enhance image: "
            "insufficient pixel contrast."
        )


    rgb_display = np.clip(
        (rgb_display - low)
        / (high - low),
        0,
        1
    )


    # ========================================================
    # Gamma adjustment
    # ========================================================

    rgb_display = np.power(
        rgb_display,
        0.7
    )


    # ========================================================
    # Convert to 8-bit RGB
    # ========================================================

    rgb_uint8 = (
        rgb_display * 255
    ).astype(
        np.uint8
    )


    # ========================================================
    # Create PIL image
    # ========================================================

    image = Image.fromarray(
        rgb_uint8
    )


    # ========================================================
    # Upscale 2x
    # ========================================================

    image = image.resize(
        (
            image.width * 2,
            image.height * 2
        ),
        Image.Resampling.LANCZOS
    )


    # ========================================================
    # Mild sharpening
    # ========================================================

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=1.2,
            percent=120,
            threshold=3
        )
    )


    # ========================================================
    # Save image
    # ========================================================

    image.save(
        output_file
    )

    print(
        "Saved:",
        output_file
    )


# ============================================================
# CURRENT SATELLITE IMAGE
# ============================================================

def get_satellite_image(
    latitude,
    longitude
):

    catalog = Client.open(
        "https://earth-search.aws.element84.com/v1"
    )


    # --------------------------------------------------------
    # Search for a low-cloud Sentinel-2 image
    # --------------------------------------------------------

    search = catalog.search(

        collections=[
            "sentinel-2-l2a"
        ],

        intersects={
            "type": "Point",
            "coordinates": [
                longitude,
                latitude
            ]
        },

        datetime=(
            "2025-01-01/"
            "2026-12-31"
        ),

        query={
            "eo:cloud_cover": {
                "lt": 20
            }
        },

        max_items=1
    )


    items = list(
        search.items()
    )


    if not items:

        raise Exception(
            "No suitable low-cloud satellite "
            "data found for these coordinates."
        )


    item = items[0]


    start_time = item.properties.get(
        "datetime"
    )


    # ========================================================
    # Generate current image
    # ========================================================

    output_file = (
        "static/outputs/"
        "sentinel_image.png"
    )


    create_rgb_image(
        item,
        latitude,
        longitude,
        output_file
    )


    # ========================================================
    # Return current image information
    # ========================================================

    return {

        "image":
            "/static/outputs/"
            "sentinel_image.png",

        "observation":
            start_time,

        "granule":
            item.id,

        "cloud_cover":
            item.properties.get(
                "eo:cloud_cover"
            ),

        "latitude":
            latitude,

        "longitude":
            longitude
    }


# ============================================================
# HISTORICAL IMAGE SEARCH
# ============================================================

def find_historical_images(
    latitude,
    longitude
):

    catalog = Client.open(
        "https://earth-search.aws.element84.com/v1"
    )


    # --------------------------------------------------------
    # Find lowest-cloud image for a particular year
    # --------------------------------------------------------

    def find_best_image(year):

        search = catalog.search(

            collections=[
                "sentinel-2-l2a"
            ],

            intersects={
                "type": "Point",
                "coordinates": [
                    longitude,
                    latitude
                ]
            },

            # Search the entire year

            datetime=(
                f"{year}-01-01/"
                f"{year}-12-31"
            ),

            # Temporarily allow up to 50%
            # while searching

            query={
                "eo:cloud_cover": {
                    "lt": 50
                }
            },

            max_items=50
        )


        items = list(
            search.items()
        )


        print(
            f"{year}: Found "
            f"{len(items)} images"
        )


        if not items:

            raise Exception(
                f"No Sentinel-2 image found "
                f"for {year}."
            )


        # ----------------------------------------------------
        # Select the image with lowest cloud cover
        # ----------------------------------------------------

        best_item = min(

            items,

            key=lambda item:
                item.properties.get(
                    "eo:cloud_cover",
                    100
                )
        )


        return best_item


    # ========================================================
    # Find 2018 image
    # ========================================================

    before = find_best_image(
        2018
    )


    # ========================================================
    # Find 2026 image
    # ========================================================

    after = find_best_image(
        2026
    )


    # ========================================================
    # Print selected scenes
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "       SELECTED HISTORICAL IMAGES"
    )

    print(
        "=========================================="
    )


    print()
    print(
        "2018 IMAGE"
    )

    print(
        "ID:",
        before.id
    )

    print(
        "Date:",
        before.properties.get(
            "datetime"
        )
    )

    print(
        "Cloud cover:",
        before.properties.get(
            "eo:cloud_cover"
        )
    )


    print()
    print(
        "2026 IMAGE"
    )

    print(
        "ID:",
        after.id
    )

    print(
        "Date:",
        after.properties.get(
            "datetime"
        )
    )

    print(
        "Cloud cover:",
        after.properties.get(
            "eo:cloud_cover"
        )
    )


    # ========================================================
    # Generate 2018 image
    # ========================================================

    create_rgb_image(

        before,

        latitude,

        longitude,

        "static/outputs/"
        "sentinel_2018.png"
    )


    # ========================================================
    # Generate 2026 image
    # ========================================================

    create_rgb_image(

        after,

        latitude,

        longitude,

        "static/outputs/"
        "sentinel_2026.png"
    )


    print()
    print(
        "Historical images generated successfully."
    )


    # ========================================================
    # Return information to Flask
    # ========================================================

    return {

        "before": {

            "year": 2018,

            "image":
                "/static/outputs/"
                "sentinel_2018.png",

            "observation":
                before.properties.get(
                    "datetime"
                ),

            "granule":
                before.id,

            "cloud_cover":
                before.properties.get(
                    "eo:cloud_cover"
                )
        },


        "after": {

            "year": 2026,

            "image":
                "/static/outputs/"
                "sentinel_2026.png",

            "observation":
                after.properties.get(
                    "datetime"
                ),

            "granule":
                after.id,

            "cloud_cover":
                after.properties.get(
                    "eo:cloud_cover"
                )
        }
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    find_historical_images(
        13.0827,
        80.2707
    )
