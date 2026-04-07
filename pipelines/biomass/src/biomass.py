from shared.utils.allometry import estimate_crown_area, estimate_dbh_from_height, chave_formula, kuyah_formula
import geopandas as gpd
import pandas as pd
import numpy as np
import logging

class Biomass:
    '''
    Main entry point that defines the methods for processing biomass using both Chave and Kuyah allometry methodologies
    '''
    def __init__(self, tree_height_metrics_gdf, wood_density, biomass_output_path) -> None:
        self.gdf = gpd.read_file(tree_height_metrics_gdf)
        self.wood_density = wood_density
        self.output_path = biomass_output_path
        self.logger = logging.getLogger('Biomass')
        

    def calculate_plot_biomass(self) -> None:
        '''
        Calculate the biomass per plot using Kuyah and Chave methods
        '''
        self._feature_engineering()
        self._calculate_chave_biomass()
        self._calculate_kuyah_biomass()
        self.gdf.to_file(f'{self.output_path}')
        self.logger.info(f'Biomass calculation done. File saved at {self.output_path}')
        
    def _feature_engineering(self):
        self.logger.info('Feature engineering...')
        self.gdf['area'] = self.gdf.geometry.area 
        self.logger.info('Canopy area from canopy percent')
        self.gdf['canopy_area'] = (self.gdf['canopy_pct'] * self.gdf['area']) / 100
        self.logger.info('Crown area as total canopy area divided by no of trees')
        self.gdf['crown_area'] = self.gdf.apply(lambda x: estimate_crown_area(x['canopy_area'], x['trees_aliv']), axis=1)
        self.logger.info('DBH estimated from mean height(Chave)')
        
    def _calculate_chave_biomass(self):
        self.gdf['chave_dbh'] = self.gdf['hmean'].apply(estimate_dbh_from_height)
        
        #biomass per tree and total biomass
        self.logger.info('Calculating biomass from Chave')
        self.gdf['chave_biomass_per_tree'] = self.gdf.apply(
            lambda x: chave_formula(self.wood_density, x['chave_dbh'], x['hmean']), 
            axis=1
        )
        self.gdf['chave_total_biomass'] = self.gdf['chave_biomass_per_tree'] * self.gdf['trees_aliv']
        
    def _calculate_kuyah_biomass(self):
        self.logger.info('Calculating Kuyah biomass')
        self.gdf['kuyah_biomass_per_tree'] = self.gdf.apply(
            lambda x: kuyah_formula(x['crown_area'], x['hmean']),
            axis=1
        )
        self.gdf['kuyah_total_biomass'] = self.gdf['kuyah_biomass_per_tree'] * self.gdf['trees_aliv']
        