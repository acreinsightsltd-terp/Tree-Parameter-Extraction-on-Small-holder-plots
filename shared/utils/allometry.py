import numpy as np
import geopandas as gpd

def estimate_crown_area(canopy_area, trees_alive):
    """Calculates average crown area per living tree."""
    return canopy_area / trees_alive if trees_alive > 0 else 0

def estimate_dbh_from_height(hmean):
    """Derives DBH from mean height using the Chave height-diameter relationship."""
    return 0.6 * (hmean ** 1.3)

def chave_formula(wood_density, dbh, hmean):
    """The core Chave allometric equation for a single tree's biomass."""
    return 0.0673 * (wood_density * (dbh**2) * hmean) ** 0.976

def kuyah_formula(crown_area, hmean):
    '''
    The core Kuyah allometric equation for a single tree  biomass
    '''
    return 0.25 * (crown_area * hmean ** 0.9)