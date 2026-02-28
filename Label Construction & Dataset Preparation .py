import rasterio
from scipy.stats import mode
import seaborn as sns
from tqdm import tqdm

# --- 1. Define ESA Class Mappings ---
# Mapping ESA WorldCover 2021 codes to Simplified Categories
esa_to_simplified = {
    50: 'Built-up',
    10: 'Vegetation', 20: 'Vegetation', 30: 'Vegetation', 95: 'Vegetation', 100: 'Vegetation',
    80: 'Water', 90: 'Water',
    40: 'Cropland',
    60: 'Others', 70: 'Others' # Bare/sparse vegetation, Snow/ice
}

# Mapping Simplified categories to integer IDs for PyTorch
category_to_id = {'Built-up': 0, 'Vegetation': 1, 'Water': 2, 'Cropland': 3, 'Others': 4}
id_to_category = {v: k for k, v in category_to_id.items()}

# --- 2. Extract Labels from TIF ---
def get_dominant_label(lon, lat, tif_src, window_size=128):
    # Convert lat/lon to raster row/col indices
    row, col = tif_src.index(lon, lat)
    
    # Calculate offset to center the 128x128 window
    half_w = window_size // 2
    window = rasterio.windows.Window(col - half_w, row - half_w, window_size, window_size)
    
    # Read the patch
    patch = tif_src.read(1, window=window)
    
    # Handle edge cases where patch is cut off at the borders
    if patch.shape != (window_size, window_size):
        return None
        
    # Calculate the dominant class (mode)
    dominant_class = mode(patch.flatten(), keepdims=True).mode[0]
    
    # Map to simplified label and return integer ID
    simplified_label = esa_to_simplified.get(dominant_class, 'Others')
    return category_to_id[simplified_label]

# Open raster and assign labels
labels =[]
with rasterio.open('path/to/land_cover.tif') as src:
    for idx, row in tqdm(filtered_images_gdf.iterrows(), total=len(filtered_images_gdf), desc="Extracting Labels"):
        label_id = get_dominant_label(row.geometry.x, row.geometry.y, src)
        labels.append(label_id)

filtered_images_gdf['target_id'] = labels

# Drop images near edges where 128x128 patches couldn't be extracted
filtered_images_gdf = filtered_images_gdf.dropna(subset=['target_id'])
filtered_images_gdf['target_id'] = filtered_images_gdf['target_id'].astype(int)

# --- (Your Train/Test split code goes here) ---
# train_df, test_df = train_test_split(...)

# --- 3. Visualize Class Distribution ---
plt.figure(figsize=(8, 5))
sns.countplot(x=train_df['target_id'].map(id_to_category), palette='viridis')
plt.title('Class Distribution in Training Set')
plt.xlabel('Land Cover Category')
plt.ylabel('Count')
plt.show()