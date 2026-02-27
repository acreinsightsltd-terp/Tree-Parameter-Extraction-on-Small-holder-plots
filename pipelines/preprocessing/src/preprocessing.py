import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.mask import mask
import geopandas as gpd
from shared.utils.utils import is_multiband, infer_band_name
from rasterio.io import MemoryFile
import logging
import numpy as np
import json
import os

class Preprocessor:
    '''
    Main class defining methods for preprocessing including but not limited to:
    - stacking
    - masking
    '''
    def __init__(self, params) -> None:
        self.params = params
        self.logger = logging.getLogger('Preprocessor')
        self.excluded_bands = params['preprocessing']['excluded_bands']
        self.imagery_folder = params['indices']['imagery_folder']
        
    def find_bands(self, year: int):
        '''
        Find bands to stack for a specific year
        '''
        files = []
        for file in os.listdir(self.imagery_folder):
            if not file.lower().endswith('.tif') or str(year) not in file:
                continue
            path = os.path.join(self.imagery_folder, file)
            if is_multiband(path):
                continue
            base = os.path.splitext(file)[0].upper()
            tokens = base.split("_")
            band_name = infer_band_name(file)

            # Exclude unwanted Sentinel bands
            if band_name in self.excluded_bands:
                continue

            files.append({
                "path": path,
                "band": band_name
            })
            self.logger.info(f'Appended band paths.')
        return files
        
    def stack_rasters(self, files: list, year: int):
        '''
        Create a stack of bands and indices to be used in classification
        
        :param files: File paths and band names for rasters to be stacked
        :param year: The year of the image being processed
        '''
        arrays = []
        band_names = []
        output_path = os.path.join(self.imagery_folder, f'stacked_{str(year)}.tif') 
        if os.path.exists(output_path):
            self.logger.warning(f'Stacked rasters for {year} already exists')
            return
        with rasterio.open(files[0]["path"]) as ref:
            meta = ref.meta.copy()
            crs = ref.crs
            transform = ref.transform
            width = ref.width
            height = ref.height

        for f in files:
            with rasterio.open(f["path"]) as src:
                if (
                    src.crs != crs or
                    src.transform != transform or
                    src.width != width or
                    src.height != height
                ):
                    raise ValueError(f"Raster mismatch: {f['path']}")

                arrays.append(src.read(1))
                band_names.append(f["band"])
                self.logger.info(f'Stacking bands: {band_names}')

        stack = np.stack(arrays, axis=0)

        meta.update({
            "count": len(stack),
            "dtype": stack.dtype
        })

        with rasterio.open(output_path, "w", **meta) as dst:
            for i, band in enumerate(stack, start=1):
                dst.write(band, i)
                dst.set_band_description(i, band_names[i - 1])
        self.logger.info(f'Stacked rasters and saved to {output_path}')

        return band_names
            
            
    def coregister_raster(self, src_path: str, dest_path: str, ref_path: str) -> None:
        """Reproject and align raster to match reference raster."""
        #open the raster to be aligned
        with rasterio.open(src_path) as src:
            src_transform = src.transform
            #open the reference raster/ raster with target dimensions and resolution
            with rasterio.open(ref_path) as ref:
                dst_crs = ref.crs
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs,    
                    dst_crs,    
                    ref.width,   
                    ref.height,  
                    *ref.bounds,
                )
                dst_kwargs = src.meta.copy()
                dst_kwargs.update({
                                "crs": dst_crs,
                                "transform": dst_transform,
                                "width": dst_width,
                                "height": dst_height,
                                "nodata": 0,
                                "dtype": src.meta["dtype"],})
                #write the aligned raster
                with rasterio.open(dest_path, 'w', **dst_kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest#nearest resampling avoids overstretching rasters, esp given we are moving from 10m to around 3m
                        )

        self.logger.info(f"Coregistered: {os.path.basename(src_path)} to {dest_path}")
            