# Google Earth Engine

Google Earth Engine (GEE) is a cloud based platform for working with large collections of geospatial data. It provides access to a vast catalog of satellite imagery and environmental datasets along with server side tools for processing them at global scale. Researchers, students, and analysts use it for tasks such as land cover mapping, vegetation monitoring, environmental change analysis, disaster assessment, and many other forms of earth observation. Because the computation happens on Google's servers, users can run complex spatial operations on huge datasets without needing powerful local hardware.

## Basic tutorial
Search for "Google Earth Engine". Go to the site and create a project under academic use.

```bash
pip install earthengine-api
pip install geemap
# pip install notebook ipywidgets
# pip install jupyterlab
```

```python
import ee
import geemap

ee.Authenticate()  # run once
ee.Initialize(project="my-project-project_id")  # use your project ID from the GEE site

Map = geemap.Map(center=[52.37, 4.89], zoom=10)

image = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(ee.Geometry.Point([4.895168, 52.370216]))
    .filterDate("2020-06-01", "2020-06-10")
    .sort("CLOUDY_PIXEL_PERCENTAGE")
    .first()
)

ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

# Add layers
vis = {"min": 0, "max": 1, "palette": ["blue", "white", "green"]}
Map.addLayer(ndvi, vis, "NDVI")
Map.addLayerControl()

# In a Jupyter notebook
Map

# Export as HTML for static viewing
# Map.to_html("ndvi_map.html")
```