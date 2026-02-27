import yaml
import geopandas as gpd
import zipfile
import tempfile
import os
from shapely.geometry import GeometryCollection, Point, MultiPoint
import pandas as pd
import logging

class Samples_Merger:
    '''
    Reusable factory that merges google earth labels into one file and saves them as a shapefile
    '''
    def __init__(self, kmz_dir, crs) -> None:
        self.logger = logging.getLogger('TrainingSamplesMerger')
        self.crs = crs
        self.kmz_dir = kmz_dir
    
    @staticmethod
    def _normalize_geometries(gdf):
        '''
        Explode multipoint/geometry collection from kmz to usable geometry by geopandas
        
        :param gdf: The geodataframe containing the geometries
        '''
        rows = []

        for _, row in gdf.iterrows():
            geom = row.geometry

            if isinstance(geom, Point):
                rows.append(row)

            elif isinstance(geom, MultiPoint):
                for pt in geom.geoms:
                    new_row = row.copy()
                    new_row.geometry = pt
                    rows.append(new_row)

            elif isinstance(geom, GeometryCollection):
                for g in geom.geoms:
                    if isinstance(g, Point):
                        new_row = row.copy()
                        new_row.geometry = g
                        rows.append(new_row)

            else:
                # silently skip unsupported geometries
                pass

        return gpd.GeoDataFrame(rows, crs=gdf.crs)
    
    def merge_kmls(self, year, output_path):
        '''
        Iterates through directory to find all kmz files for a certain year 
        :param kmz_dir: Directory housing the labels 
        :param year: Year of labels to find 
        :param output_path: The path to save the merged shapefile
        '''

        # Check once per year
        if os.path.exists(output_path):
            self.logger.warning(
                f"Training labels for {year} already exist — skipping."
            )
            return

        gdfs = []

        for file in os.listdir(self.kmz_dir):
            if file.lower().endswith('.kmz') and str(year) in file:

                self.logger.info(f"Processing {file}")

                name, _ = os.path.splitext(file)
                class_id = name.split("_")[-1]
                kmz_path = os.path.join(self.kmz_dir, file)

                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(kmz_path, 'r') as kmz:
                        kmz.extractall(tmpdir)

                    kml_files = [f for f in os.listdir(tmpdir) if f.endswith(".kml")]
                    if not kml_files:
                        continue

                    gdf = gpd.read_file(
                        os.path.join(tmpdir, kml_files[0]),
                        driver="KML"
                    )

                    gdf = self._normalize_geometries(gdf)
                    gdf["class_id"] = int(class_id)

                    gdfs.append(gdf)

        if not gdfs:
            self.logger.warning(f"No KMZs found for {year}")
            return

        self.logger.info("Merging gdfs...")
        merged_gdf = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True),
            crs=gdfs[0].crs
        ).to_crs(self.crs)
        self.logger.info(f'Class_id counts for {year}: {merged_gdf['class_id'].value_counts()}')

        merged_gdf.to_file(output_path, driver="ESRI Shapefile")

        self.logger.info(f"Merged training samples for {year}")