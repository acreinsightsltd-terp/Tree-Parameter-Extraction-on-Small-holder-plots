import rasterio
import numpy as np
import geopandas as gpd
import logging
from rasterio.features import rasterize
import os


class PerPlotMetrics:
    """
    Compute per-plot canopy height metrics from a CHM raster.
    """

    def __init__(self, chm_path, plots_gdf, metrics_output_path, params):
        self.chm_path = chm_path
        self.plots = gpd.read_file(plots_gdf)
        self.plots['id_col'] = self.plots.index + 1
        self.output_path = metrics_output_path
        self.params = params
        self.logger = logging.getLogger('PerPlotMetrics')

        self.height_threshold = params["tree_height"]["height_threshold_m"]
        self.height_bins = params["tree_height"]["height_bins"]
        self.nodata = params["tree_height"]["nodata_value"]

        self.logger.info("Initialized PerPlotMetrics")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def compute_all_metrics(self):
        """
        Compute all canopy metrics for each plot.

        Returns
        -------
        geopandas.GeoDataFrame
            Plot geometries with appended structural metrics
        """
        records = []

        self.logger.info("Starting per-plot metric extraction")

        with rasterio.open(self.chm_path) as src:
            self.logger.info(f'Opened clipped chm')
            assert self.plots.crs == src.crs, 'Plots and chm crs incompatible!'
            for idx, plot in self.plots.iterrows():
                self.logger.info(f"Processing plot ID {plot['id_col']}")
                values = self._extract_plot_pixels(src, plot.geometry)

                if values.size == 0:
                    self.logger.warning(f"No CHM data for plot {plot['id']}")
                    metrics = self._empty_metrics()
                else:
                    metrics = self._compute_metrics(values)

                record = {**plot.drop("geometry"), **metrics}
                records.append(record)
                self.logger.info(f'Completed processing for plot {plot['id_col']}')
        gdf = gpd.GeoDataFrame(records, geometry=self.plots.geometry, crs=self.plots.crs)
        gdf.to_file(self.output_path)
        if os.path.exists(self.output_path):
            self.logger.info(f'Completed processing and saved file to {self.output_path}')
            self.logger.info(f'{gdf.sample(10)}')
        else:
            self.logger.warning('Could not save file.')

    # --------------------------------------------------
    # Internal methods
    # --------------------------------------------------

    def _extract_plot_pixels(self, src, geometry):
        """
        Extract CHM pixel values inside a plot geometry.
        Memory-safe via windowed read.
        """
        window = rasterio.features.geometry_window(src, [geometry], pad_x=1, pad_y=1)
        transform = src.window_transform(window)

        chm = src.read(1, window=window)
        mask = rasterize(
            [(geometry, 1)],
            out_shape=chm.shape,
            transform=transform,
            fill=0,
            dtype="uint8"
        )

        values = chm[(mask == 1) & (chm != self.nodata)]
        return values

    def _binary_tree_mask(self, heights):
        """Binary tree / non-tree mask."""
        return heights >= self.height_threshold

    def _compute_metrics(self, heights):
        """Compute all height-based metrics."""
        tree_mask = self._binary_tree_mask(heights)
        tree_heights = heights[tree_mask]
        tree_heights = np.asarray(tree_heights)

        canopy_area_pct = self._compute_canopy_area(tree_mask, heights.size)

        distribution = np.histogram(tree_heights, bins=self.height_bins)
        if tree_heights.size == 0:
            self.logger.warning(f"No heights above thres for plot.")
            return {
            "hmax": np.nan,
            "h95": np.nan,
            "hmedian": np.nan,
            "hmean": np.nan,
            "hmin": np.nan,
            "hstd": np.nan,
            "canopy_pct": 0.0,
            "height_distribution": None
            }
        else:
            return {
                "hmax": float(np.max(tree_heights)),
                "h95": float(np.percentile(tree_heights, 95)),
                "hmedian": float(np.median(tree_heights)),
                "hmean": float(np.mean(tree_heights)),
                "hmin": float(np.min(tree_heights)),
                "hstd": float(np.std(tree_heights)),
                "canopy_pct": canopy_area_pct,
                "height_distribution": distribution  # stored in-memory
            }

    def _compute_canopy_area(self, tree_mask, total_pixels):
        """Canopy area as percentage of plot."""
        return (np.sum(tree_mask) / total_pixels) * 100.0

    def _empty_metrics(self):
        """Fallback metrics for empty plots."""
        return {
            "hmax": np.nan,
            "h95": np.nan,
            "hmedian": np.nan,
            "hmean": np.nan,
            "hmin": np.nan,
            "hstd": np.nan,
            "canopy_pct": 0.0,
            "height_distribution": None
        }
