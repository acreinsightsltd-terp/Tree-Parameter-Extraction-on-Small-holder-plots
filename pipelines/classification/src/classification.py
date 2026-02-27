import os
import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from tqdm import tqdm

class RFClassifier:
    """
    Random Forest classification pipeline driven by year-based inputs.
    Designed for looping across years from a flow pipeline.
    """

    def __init__(self, imagery_dir, labels_dir, processed_dir, crs):
        self.logger = logging.getLogger("RFClassification")
        self.imagery_dir = imagery_dir
        self.labels_dir = labels_dir
        self.processed_dir = processed_dir
        self.crs = crs

        os.makedirs(f"{processed_dir}/models", exist_ok=True)
        os.makedirs(f"{processed_dir}/classified", exist_ok=True)
        os.makedirs(f"{processed_dir}/metrics", exist_ok=True)
        
    def load_data(self, year):
        self.year = year

        self.raster_path = os.path.join(
            self.imagery_dir, f"stacked_{year}.tif"
        )
        if not os.path.exists(self.raster_path):
            raise FileExistsError(f'No stacked raster found for {year}')
            
        self.training_path = os.path.join(
            self.labels_dir, f"training_samples_{year}.shp"
        )
        if not os.path.exists(self.training_path):
            raise FileExistsError(f'No training samples found for {year}')
        
        self.logger.info(f"Loading data for year {year}")

        self.raster = rasterio.open(self.raster_path)
        self.training = gpd.read_file(self.training_path).to_crs(self.raster.crs)

        self.bands = self.raster.count
        self.feature_names = list(self.raster.descriptions)

        self.logger.info(f"Raster bands: {self.feature_names}")
  
        
    def sample_training_data(self):
        """
        Samples raster values at training points.
        Preserves original class_id values.
        """
        self.logger.info("Sampling training data...")

        X, y = [], []

        for _, row in tqdm(self.training.iterrows(), total=len(self.training)):
            geom = [row.geometry.__geo_interface__]

            try:
                out_image, _ = rasterio.mask.mask(
                    self.raster, geom, crop=True
                )
                data = out_image.reshape(self.bands, -1).T
                data = data[~np.any(np.isnan(data), axis=1)]

                labels = np.full(data.shape[0], row["class_id"])

                X.append(data)
                y.append(labels)

            except Exception as e:
                self.logger.warning(f"Skipped geometry: {e}")

        self.X = np.vstack(X)
        self.y = np.hstack(y)

        # Class distribution logging
        class_counts = pd.Series(self.y).value_counts()
        self.logger.info("Training sample counts:")
        self.logger.info(class_counts)
    
    def split_data(self):
        self.logger.info("Splitting train/test...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )
        
    def train_rf(self, tuned=False):
        self.logger.info(f"Training RF ({'tuned' if tuned else 'base'})")

        rf = RandomForestClassifier(random_state=42)

        if tuned:
            params = {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
            }
            grid = GridSearchCV(rf, params, cv=3, n_jobs=-1)
            grid.fit(self.X_train, self.y_train)
            rf = grid.best_estimator_
        else:
            rf.fit(self.X_train, self.y_train)

        self.rf_model = rf
        self._evaluate_model(tuned)
        self._save_model(tuned)

    def _evaluate_model(self, tuned):
        '''
        Evaluation of the model confusion matrix to know how the model is performing for various classes
        
        :param tuned: A boolean arg to set tuned to true if rf_tuned model is used
        '''
        preds = self.rf_model.predict(self.X_test)

        report = classification_report(self.y_test, preds)
        cm = confusion_matrix(self.y_test, preds)

        self.logger.info(f"\n{report}")
        self.logger.info(f"\nConfusion Matrix:\n{cm}")

        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f"RF Confusion Matrix {self.year}")
        plt.savefig(
            f"{self.processed_dir}/metrics/confusion_{self.year}_{'tuned' if tuned else 'base'}.png"
        )
        plt.close()
        
    
    def _save_model(self, tuned):
        model_path = (
            f"{self.processed_dir}/models/"
            f"rf_{self.year}_{'tuned' if tuned else 'base'}.pkl"
        )
        joblib.dump(self.rf_model, model_path)


    def classify_rf(self, tuned=False):
        model_name = f"rf_{self.year}_{'tuned' if tuned else 'base'}"
        model_path = f"{self.processed_dir}/models/{model_name}.pkl"

        self.logger.info(f"Classifying raster using {model_name}")

        model = joblib.load(model_path)

        img = self.raster.read().reshape(self.bands, -1).T
        img = np.nan_to_num(img)

        preds = model.predict(img)
        classified = preds.reshape(self.raster.height, self.raster.width)

        meta = self.raster.meta.copy()
        meta.update(count=1, dtype="int16")

        out_path = f"{self.processed_dir}/classified/{model_name}.tif"
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(classified, 1)

        self.logger.info(f"Saved classified raster: {out_path}")

