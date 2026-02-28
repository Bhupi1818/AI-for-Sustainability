import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

# --- 1. Load Data ---
# Load Delhi-NCR shapefile
ncr_gdf = gpd.read_file('path/to/Delhi_NCR_Shapefile.shp') # Update path

# Load satellite images metadata (assuming a CSV or similar with lat/lon and filenames)
images_df = pd.read_csv('path/to/images_metadata.csv') 

# Convert image points to a GeoDataFrame (EPSG:4326)
images_gdf = gpd.GeoDataFrame(
    images_df, 
    geometry=gpd.points_from_xy(images_df.longitude, images_df.latitude),
    crs="EPSG:4326"
)

# --- 2. Filter Satellite Images ---
print(f"Total images before filtering: {len(images_gdf)}")

# Keep only images whose center coordinates fall inside the Delhi-NCR shapefile
filtered_images_gdf = gpd.sjoin(images_gdf, ncr_gdf, predicate='within')

print(f"Total images after filtering: {len(filtered_images_gdf)}")

# --- 3. Overlay 60x60 km Uniform Grid ---
# Convert CRS to EPSG:32644 (projected CRS in meters) for accurate gridding
ncr_projected = ncr_gdf.to_crs(epsg=32644)

# Get bounding box of the shapefile
xmin, ymin, xmax, ymax = ncr_projected.total_bounds
grid_size = 60000 # 60 km = 60,000 meters

# Generate grid polygons
grid_cells =[]
for x0 in np.arange(xmin, xmax, grid_size):
    for y0 in np.arange(ymin, ymax, grid_size):
        x1 = x0 + grid_size
        y1 = y0 + grid_size
        grid_cells.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
        
grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:32644")

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))
ncr_projected.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2, label='Delhi-NCR')
grid_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linestyle='--', alpha=0.5)
plt.title('Delhi-NCR Region with 60x60 km Uniform Grid')
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.show()