import numpy as np
from rasterio.warp import reproject, Resampling
from shared.utils.utils import mask
from rasterio.features import rasterize
from shapely.geometry import box
import geopandas as gpd
import rasterio
import logging

class CHM:
    '''
    Main entry point that defines behaviour of Canopy Height Model file functions
    '''
    def __init__(self, dsm_file, dtm_file, chm_output_path, clipped_chm_path, plots_shapefile) -> None:
        self.output_path = chm_output_path
        self.clipped_chm_path = clipped_chm_path
        self.dsm_file = dsm_file
        self.dtm_file = dtm_file
        self.plots = gpd.read_file(plots_shapefile)
        self.logger = logging.getLogger('CHM Generator')
        
        
    def generate_chm(self):
        '''
        Obtains the canopy height model which is a representation of tree heights. This is the difference between surface height(dsm) and terrain level(dtm)
        '''
        with rasterio.open(self.dsm_file) as dsm_src, rasterio.open(self.dtm_file) as dtm_src:

            meta = dsm_src.meta.copy()
            meta.update(
                {
                    "dtype": "float32",
                    "count": 1,
                    "nodata": -9999,
                    "compress": "lzw",
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                    "BIGTIFF": "YES",
                }
            )
            self.logger.info('Opened DSM and DTM.')
            with rasterio.open(self.output_path, "w", **meta) as dst:

                for _, window in dsm_src.block_windows(1):
                    dsm = dsm_src.read(1, window=window).astype("float32")
                    dtm = dtm_src.read(1, window=window).astype("float32")

                    chm = dsm - dtm
                    chm[chm < 0] = 0
                    chm[np.isnan(dsm) | np.isnan(dtm)] = -9999
                    self.logger.info('Writing CHM to disk.')
                    dst.write(chm, 1, window=window )

        return self.output_path
    
    def mask_chm(self):
        """
        Clip CHM to retain data inside AWG plots using windowed processing.
        Memory safe for very large rasters.
        """

        self.logger.info("Clipping CHM (windowed, memory-safe)")

        with rasterio.open(self.output_path) as chm_src:

            meta = chm_src.meta.copy()
            meta.update(
                {
                    "dtype": "float32",
                    "count": 1,
                    "nodata": -9999,
                    "compress": "lzw",
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                    "BIGTIFF": "YES",
                }
            )

            with rasterio.open(self.clipped_chm_path, "w", **meta) as dst:

                for _, window in chm_src.block_windows(1):

                    # Read CHM window
                    chm = chm_src.read(1, window=window).astype("float32")

                    # Spatial extent of this window
                    window_transform = chm_src.window_transform(window)
                    window_bounds = rasterio.windows.bounds(window, chm_src.transform)
                    window_geom = box(*window_bounds)

                    # Keep only plots intersecting this window
                    shapes = [
                        geom
                        for geom in self.plots.geometry
                        if geom.intersects(window_geom)
                    ]

                    if not shapes:
                        self.logger.warning('Could not find any shapes, writing no data')
                        # No plots here → write nodata
                        chm[:] = -9999
                    else:
                        # Rasterize plots mask to window
                        mask_arr = rasterize(
                            shapes,
                            out_shape=chm.shape,
                            transform=window_transform,
                            fill=0,
                            default_value=1,
                            dtype="uint8",
                        )

                        # Apply mask
                        chm = np.where(mask_arr == 1, chm, -9999)
                        chm[chm < 0] = 0

                    dst.write(chm, 1, window=window)
                    self.logger.debug(f'Writing CHM clipped window {window}')

        self.logger.info("Clipping complete.")       