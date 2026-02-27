import rasterio
import os
import geopandas as gpd
import logging

def is_multiband(raster_path):
        """Check if raster has more than one band."""
        with rasterio.open(raster_path) as src:
            return src.count > 1
        
def infer_band_name(filename):
    base = os.path.splitext(filename)[0].upper()
    tokens = base.split("_")

    # 1. Explicit spectral bands (B1, B2, ...)
    for t in tokens:
        if t.startswith("B") and t[1:].isdigit():
            return t
        elif t.startswith('B8A'):
            return t

    # 2. Known index-style tokens (generic, pipeline-safe)
    for t in tokens:
        if t.isalpha() and len(t) <= 6:  # NDVI, BSI, MNDWI, etc.
            return t

    raise ValueError(f"Cannot infer band name from {filename}")

def mask(src_file, plots_path, output_path):
    with rasterio.open(src_file) as src:
        crs = src.crs
        gdf = gpd.read_file(plots_path).to_crs('EPSG:32737')
        geoms = [feature["geometry"] for feature in gdf.__geo_interface__["features"]]
        out_image, out_transform = rasterio.mask.mask(src, geoms, crop=True)
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform})
    with rasterio.open(output_path, 'w', **out_meta) as dest:
        dest.write(out_image)
    logging.info('Clipping complete.')
