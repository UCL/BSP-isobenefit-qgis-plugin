# BSP Isobenefit Urbanism QGIS Plugin

QGIS plugin for Isobenefit Urbanism

## Installation

- `brew install qt5 pyqt pdm qgis`
- `pdm install`
- add `.env` file with path appendages:

```bash
PYTHONPATH="${env:PYTHONPATH}:/Applications/Qgis.app/Contents/Resources/python/plugins:/Applications/Qgis.app/Contents/Resources/python"
```

## Original version

This work is an outgrowth of the [original work](https://github.com/mitochevole/isobenefit-cities) (forked to [BSP-isobenefit-original](https://github.com/UCL/BSP-isobenefit-original)) developed by Michele Voto and Luca D'Acci. This has subsequently been developed as part of the Future Urban Growth project at the Bartlett School of Planning.

## Website

A website providing a broad overview of this plugin can be found at [BSP-isobenefit-urbanism](https://github-pages.ucl.ac.uk/BSP-isobenefit-urbanism) with the associated repo located at [UCL/BSP-isobenefit-urbanism](https://github.com/UCL/BSP-isobenefit-urbanism).

## Plugin setup

QGIS does not yet have an integrated dependency management system. This means that certain of the dependencies required by this plugin have to be installed manually. Restart QGIS after the Python dependencies have been installed.

> If the below steps don't work, see [https://enmap-box.readthedocs.io/en/latest/usr_section/usr_installation.html#package-installer]() for additional context / ideas.

### Windows

1. From the Start menu, open OSGeo4W Shell.
2. Run `pip install shapely==2.0.1 rasterio==1.3.6 numba==0.55.2`

### Mac

1. Open the Terminal application
2. Run `/Applications/QGIS.app/Contents/MacOS/bin/pip install shapely==2.0.1 rasterio==1.3.6 numba==0.55.2`

### Linux

1. Open the terminal
2. Run `python3 -m pip install shapely==2.0.1 rasterio==1.3.6 numba==0.55.2`

## PB Tool

> Using `pb_tool` deploy doesn't seem to copy the files across, so using a softlink instead per next section.

- `pb_tool` uses a `pb_tool.cfg` file.
- Build the resources: `pb_tool compile`.

## Linking plugin file

- On Windows: copy and paste the dev plugin folder to the user's plugin directory, e.g. `<user>/AppData/Roaming/QGIS/QGIS3/prfiles/default/plugins`
- On Mac: Create a softlink from the development folder (`isobenefit`) to the QGIS plugins directory, this depends on the system, e.g.
  - `ln -s /Users/gareth/dev/other/BSP-isobenefit-qgis-plugin/isobenefit/ '/Users/gareth/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins'`
- Restart QGIS.

## References

- [http://g-sherman.github.io/plugin_build_tool/](PB Tool)
- [https://qgis.org/pyqgis/3.28/](PyQgis)
- [https://gis-ops.com/qgis-3-plugin-tutorial-plugin-development-reference-guide/](Plugin Guide)
- [https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/index.html](Cookbook)
- [https://webgeodatavore.github.io/pyqgis-samples/index.html](Samples)
- [https://www.geodose.com/2021/03/netcdf-temporal-visualization-qgis.html](temporal range raster)
- [https://anitagraser.com](open source GIS ramblings)
- [https://docs.qgis.org/3.4/en/docs/user_manual/working_with_mesh/mesh_properties.html#what-s-a-mesh](mesh layers)
