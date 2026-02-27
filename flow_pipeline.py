from pipelines.training_samples_merger.src.merger import Samples_Merger
from pipelines.indices.src.indices import Indices
from pipelines.preprocessing.src.preprocessing import Preprocessor
from pipelines.classification.src.classification import RFClassifier
from pipelines.tree_height.src.chm import CHM
from pipelines.tree_height.src.perplot import PerPlotMetrics
import yaml
import logging


class Flow_State:
    '''
    Entry point for orchestrating the whole pipeline
    '''
    def __init__(self, params) -> None:
        self.params = params
        self.kmz_dir = params['training_samples']['kmz_dir']
        self.crs = params['project']['crs']
      
    
    def training_samples_pipeline(self) -> None:
        for year in [2010, 2015, 2020, 2025]:
            output_path = f'{self.kmz_dir}/training_samples_{year}.shp'
            merger = Samples_Merger(self.kmz_dir, self.crs)
            merger.merge_kmls(year, output_path)
            
    def indices_pipeline(self) -> None:
        indices_calc = Indices(self.params)
        indices_calc.explode_geotiff()

        years = [2010, 2015, 2020, 2025]
        indices = ["NDVI", "MNDWI", "BSI", "MSAVI"]

        for year in years:
            mission = self.params["year_mission_map"][year]
            bands = self.params["missions"][mission]["bands"]

            # NDVI
            indices_calc.normalized_difference(
                bands["nir"],
                bands["red"],
                year,
                "NDVI"
            )

            # MNDWI
            indices_calc.normalized_difference(
                bands["green"],
                bands["swir1"],
                year,
                "MNDWI"
            )

            # BSI
            indices_calc.bsi(year)
            
            #MSAVI
            indices_calc.msavi(year)
        
    def preprocessing_pipeline(self) -> None:
        preprocessor = Preprocessor(self.params)
        for year in [2010, 2015, 2020, 2025]:
            files = preprocessor.find_bands(year)
            preprocessor.stack_rasters(files, year)
            
    def classification_pipeline(self) -> None:
        classifier = RFClassifier(self.params['classification']['imagery_folder'], 
                                  self.params['classification']['labels_folder'], 
                                  self.params['classification']['processed_folder'],
                                  self.params['project']['crs'])
        for year in [2010, 2015, 2020, 2025]:
            classifier.load_data(year)
            classifier.sample_training_data()
            classifier.split_data()
            classifier.train_rf(tuned=True)   # or True
            classifier.classify_rf(tuned=True)
        
    def tree_height_pipeline(self) ->  None:
        chm = CHM(self.params['tree_height']['dsm_path'],
                  self.params['tree_height']['dtm_path'],
                  self.params['tree_height']['chm_output_path'],
                  self.params['tree_height']['clipped_chm_output_path'],
                  self.params['tree_height']['plot_shapefile_path']
                  )
        # chm.generate_chm()
        # chm.mask_chm()
        perplot = PerPlotMetrics(self.params['tree_height']['clipped_chm_output_path'],
                                 self.params['tree_height']['plot_shapefile_path'],
                                 self.params['tree_height']['metrics_output_path'],
                                 self.params
                                 )
        gdf = perplot.compute_all_metrics()