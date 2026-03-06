Land Cover Classification — Delhi-NCR

Classifies satellite imagery into 5 land cover categories using ESA WorldCover 2021 labels and
ResNet18.

Categories

ID Label ID Label

0 Built-up 3 Cropland
1 Vegetation 4 Others
2 Water

Pipeline

1. Spatial Filtering — Clips images to Delhi-NCR boundary, overlays 60×60 km grid

2. Label Extraction — Reads dominant ESA class from a 128×128 px GeoTIFF window per
image

3. Train & Evaluate — ResNet18 with Accuracy, Weighted F1, and Confusion Matrix

Setup

pip install rasterio geopandas pandas numpy matplotlib seaborn \
scipy shapely tqdm scikit -learn torch torchvision

Run

python "Spatial_Reasoning_&_Data_Filtering.py"
python "Label_Construction_&_Dataset_Preparation.py"
python "Model_Training_&_Supervised_Evaluation.py"

Update file paths for the Delhi-NCR shapefile, image metadata CSV, and ESA WorldCover Geo-
TIFF before running.

1
