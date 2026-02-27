import logging
import rasterio
import rioxarray as rio
import numpy as np
import logging
import os

class Indices:
    '''
    Main class where indices methods and behaviour is defined
    '''
    def __init__(self, params) -> None:
        self.params = params
        self.imagery_dir = params['indices']['imagery_folder']
        self.logger = logging.getLogger('IndicesCalculator')


        
    def explode_geotiff(self):
        '''
        Explode multiband tiff images into individual bands in input directory

        '''
        for file in os.listdir(self.imagery_dir):
            if not file.lower().endswith('.tif') or 'b' in file.lower() or file.startswith('20') or file.startswith('stack'):
                continue
            input_path = os.path.join(self.imagery_dir, file)
            base_name = os.path.splitext(file)[0]

            with rasterio.open(input_path) as src:
                profile = src.profile

                for band_index in range(1, src.count + 1):
                    band_data = src.read(band_index)

                    # 🔑 Get existing band name (authoritative)
                    band_name = src.descriptions[band_index - 1]

                    if band_name is None:
                        raise ValueError(
                            f"Band {band_index} in {file} has no name — "
                            "metadata is inconsistent with expectations."
                        )

                    safe_band_name = band_name.replace(" ", "_")

                    output_name = f"{base_name}_{safe_band_name}.tif"
                    output_path = os.path.join(self.imagery_dir, output_name)

                    profile.update(count=1)
                    if not os.path.exists(output_path):
                        with rasterio.open(output_path, "w", **profile) as dst:
                            dst.write(band_data, 1)

                            # 🔑 Preserve band description
                            dst.set_band_description(1, band_name)
                    else:
                        self.logger.warning(f'Skipped writing, file already exists at {output_path}')
                        
            self.logger.warning(f"Exploded with preserved band names: {file}")
       
     
    def _find_band(self, input_dir, target_band: str, year: int):
        '''
        Finds a specific band in a directory containing satellite imagery bands
        :param target_band: this is a string containing the band desc to target
        :returns: path of the band matching the regex
        >>> find_band('b2')
        data/processed/sentinel2_b2.tif
        '''
        for file in os.listdir(input_dir):
            if str(year) not in file.lower():
                continue
            if target_band in file and file.lower().endswith('.tif'):
                band_path = os.path.join(input_dir, file)
                self.logger.info(f'Found band {band_path}')
                return band_path
        raise FileNotFoundError(f'Could not find any specific band matching the {target_band}.')
    
    def normalized_difference(self, first: str, next: str, year, index: str) -> None:
        '''
        This is a function to calculate the normalized difference between bands
        :param first: The first band to be used in equation- key in a desc('b2')
        :param next: The second band to be used in equation- key in a desc('b8')
        :returns: Normalized difference tif
        '''
        first_path = self._find_band(self.imagery_dir, first, year)
        next_path = self._find_band(self.imagery_dir, next, year)
        out_path = os.path.join(self.imagery_dir, f'{year}_{index}.tif')
        if not os.path.exists(out_path): 
            with rasterio.open(first_path) as band, rasterio.open(next_path) as band0:
                if band.shape != band0.shape or band.crs !=band0.crs:
                    raise ValueError('Ensure crs and shape align for rasters')
                red = band.read(1).astype('float32')
                nir = band0.read(1).astype('float32')
                # Avoid division by zero
                np.seterr(divide='ignore', invalid='ignore')
                ndvi = (red - nir) / (red + nir)
                ndvi = np.clip(ndvi, -1, 1)
                meta = band.meta.copy()
                meta.update({
                "count": 1,
                "dtype": "float32",
                "driver": "GTiff",
                "nodata": -9999
                })
                self.logger.info(f'Writing the index to disk: {out_path}')
                with rasterio.open(out_path, 'w', **meta) as dest:
                    dest.write(ndvi, 1)
        else:
            self.logger.warning(f'File already exists at {out_path}.')
    
    def bsi(self, year):
        """
        Bare Soil Index (BSI)
        BSI = ((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))
        """
        out_path = os.path.join(self.imagery_dir, f"{year}_BSI.tif")
        if os.path.exists(out_path):
            self.logger.warning(f"{out_path} exists, skipping.")
            return

        sensor = self._detect_sensor(year)
        bands = self.params['missions'][sensor]['bands']

        swir, swir_src = self._load_band(bands["swir1"], year)
        red,  red_src  = self._load_band(bands["red"], year)
        nir,  nir_src  = self._load_band(bands["nir"], year)
        blue, blue_src = self._load_band(bands["blue"], year)

        self._validate_alignment([swir_src, red_src, nir_src, blue_src])

        np.seterr(divide="ignore", invalid="ignore")
        bsi = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue))
        bsi = np.clip(bsi, -1, 1)

        meta = swir_src.meta.copy()
        meta.update(dtype="float32", count=1, nodata=-9999)

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(bsi, 1)

        self.logger.info(f"BSI written to {out_path}")
        
    def msavi(self, year):
        '''
        Modified Soil Adjusted Vegetation Index that allows the reduction of soil reflectance esp in our context of semi arid region
        
        :param year: Year to calculate
        '''
        out_path = os.path.join(self.imagery_dir, f"{year}_MSAVI.tif")
        if os.path.exists(out_path):
            self.logger.warning(f"{out_path} exists, skipping.")
            return
        sensor = self._detect_sensor(year)
        bands = self.params['missions'][sensor]['bands']

        red,  red_src  = self._load_band(bands["red"], year)
        nir,  nir_src  = self._load_band(bands["nir"], year)

        self._validate_alignment([red_src, nir_src])

        np.seterr(divide="ignore", invalid="ignore")
        msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
        msavi = np.clip(msavi, -1, 1)

        meta = nir_src.meta.copy()
        meta.update(dtype="float32", count=1, nodata=-9999)

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(msavi, 1)

        self.logger.info(f"MSAVI written to {out_path}")
        
            
    def _detect_sensor(self, year):
        year_files = [
            f.lower()
            for f in os.listdir(self.imagery_dir)
            if str(year) in f
        ]

        sensors = set()

        for f in year_files:
            if "sentinel" in f:
                sensors.add("sentinel2")
            elif "landsat8" in f:
                sensors.add("landsat8")
            elif "landsat5" in f:
                sensors.add("landsat5")

        if len(sensors) == 1:
            return sensors.pop()

        if len(sensors) > 1:
            raise ValueError(
                f"Multiple sensors detected for {year}: {sensors}"
            )

        return None


    def _load_band(self, band_code, year):
        path = self._find_band(self.imagery_dir, band_code, year)
        with rasterio.open(path) as src:
            return src.read(1).astype("float32"), src

    def _validate_alignment(self, sources):
        shapes = {src.shape for src in sources}
        crs = {src.crs for src in sources}
        if len(shapes) > 1 or len(crs) > 1:
            raise ValueError("Raster alignment mismatch")
        

    