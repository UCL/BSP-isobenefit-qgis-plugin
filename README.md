# bsp-future-urban-growth

Future Urban Growth (Isobenefit Cities)

## Installation

- `brew install qt5 pyqt pdm qgis`
- `pdm install`
- add `.env` file with path appendages:

```bash
PYTHONPATH="${env:PYTHONPATH}:/Applications/Qgis.app/Contents/Resources/python/plugins:/Applications/Qgis.app/Contents/Resources/python"
```

## Website

A website providing a broad overview of this plugin can be found at: [BSP-futurb](https://github.com/UCL/BSP-futurb)

## Plugin setup

Plugin requires additional pip packages:

> See: [https://enmap-box.readthedocs.io/en/latest/usr_section/usr_installation.html#package-installer]()

```bash
/Applications/QGIS.app/Contents/MacOS/bin/pip install shapely rasterio numba
```

> On windows, use the OSGeo4W shell app to run pip

## References

- [http://g-sherman.github.io/plugin_build_tool/](PB Tool)
- [https://qgis.org/pyqgis/3.28/](PyQgis)
- [https://gis-ops.com/qgis-3-plugin-tutorial-plugin-development-reference-guide/](Plugin Guide)
- [https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/index.html](Cookbook)
- [https://webgeodatavore.github.io/pyqgis-samples/index.html](Samples)
- [https://www.geodose.com/2021/03/netcdf-temporal-visualization-qgis.html](temporal range raster)
- [https://anitagraser.com](open source GIS ramblings)
- [https://docs.qgis.org/3.4/en/docs/user_manual/working_with_mesh/mesh_properties.html#what-s-a-mesh](mesh layers)

## PB Tool

- Using `pb_tool` deploy doesn't seem to copy the files across, so use a softlink instead.
- `pb_tool` uses a `pb_tool.cfg` file.
- Build the resources: `pb_tool compile`.

## Linking plugin file

- On Windows: copy and paste the dev plugin folder to the user's plugin directory, e.g. `<user>/AppData/Roaming/QGIS/QGIS3/prfiles/default/plugins`
- On Mac: Create a softlink from the dev plugin folder to the QGIS plugins directory, this depends on the system, e.g.
  - `ln -s /Users/gareth/dev/other/BSP-future-urban-growth/futurb /Users/gareth/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins`.
  - `ln -s /Users/gareth/dev/BSP-future-urban-growth/futurb /Users/gareth/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins`
