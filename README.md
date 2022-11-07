# bsp-future-urban-growth

Future Urban Growth (Isobenefit Cities)

## Installation

- `brew install qt5 pyqt pdm qgis`
- `pdm install`
- add `.env` file with path appendages:

```bash
PYTHONPATH="${env:PYTHONPATH}:/Applications/Qgis.app/Contents/Resources/python/plugins:/Applications/Qgis.app/Contents/Resources/python"
```

## References

- [http://g-sherman.github.io/plugin_build_tool/](PB Tool)
- [https://qgis.org/pyqgis/3.28/](PyQgis)
- [https://gis-ops.com/qgis-3-plugin-tutorial-plugin-development-reference-guide/](Plugin Guide)
- [https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/index.html](Cookbook)

## Dev

> Using `pb_tool` deploy doesn't seem to copy the files across, so use a softlink instead.

- `pb_tool` uses a `pb_tool.cfg` file.
- Build the resources: `pb_tool compile`.
- Create a softlink from the dev plugin folder to the QGIS plugins directory: `ln -s /Users/gareth/dev/other/BSP-future-urban-growth/futurb /Users/gareth/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins`.
