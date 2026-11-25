#!/usr/bin/env python3
"""
Figure 6: high-NO2 persistence frequency maps, land pixels only, city-wide 75th-percentile
threshold. Reads the four *_NO2_persistence_frequency_landonly.tif exports from Earth Engine.

Input : <data-dir>/<City>_NO2_persistence_frequency_landonly.tif (four files)
Output: figure6_persistence_landonly.png and .pdf in --out-dir

Usage (Colab):
  !pip install -q rasterio geopandas
  !python plot_figure6_landonly.py \
      --data-dir "/content/drive/MyDrive/NO2_persistence_exports/scripts" \
      --outline-dir "/content/drive/MyDrive/NO2_persistence_exports/scripts/shapes" \
      --out-dir "/content/drive/MyDrive/NO2_persistence_exports/scripts/figures"
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Ellipse

# (file key used by Earth Engine, panel label, centre lat, centre lon)
PANELS = [
    ("Lagos", "Lagos", 6.5244, 3.3792),
    ("Kinshasa", "Kinshasa", -4.4419, 15.2663),
    ("Dar_es_Salaam", "Dar es Salaam", -6.7924, 39.2083),
    ("Johannesburg", "Johannesburg", -26.2041, 28.0473),
]
BUFFER_KM = 30
PANEL_HALF_KM = 36          # half-width of each square panel; the 30 km buffer sits inside it
CMAP = "YlOrBr"
BOUNDS = np.arange(0, 101, 10)
NORM = BoundaryNorm(BOUNDS, plt.get_cmap(CMAP).N)
SHAPE_EXTS = {".shp", ".geojson", ".gpkg", ".json"}

# Boundary simplification tolerance in degrees (0.001° ≈ 100 m), applied to all four outlines
SIMPLIFY_TOL_DEG = 0.001

# Boundary line width (pt) per city. Set each value after checking the geometry printout.
OUTLINE_LW = {"Lagos": 0.9, "Kinshasa": 0.9, "Dar es Salaam": 0.9, "Johannesburg": 0.9}

# Panel grid geometry (figure fractions), shared with the colorbar placement
GRID = dict(left=0.07, right=0.86, top=0.96, bottom=0.12, wspace=0.08, hspace=0.20)


def read_percent(path: Path):
    """Persistence GeoTIFF (0-1 fraction) -> values in % and extent (left, right, bottom, top)."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype(float)
        b = src.bounds
        extent = (b.left, b.right, b.bottom, b.top)
    return arr * 100.0, extent


def find_outline(outline_dir: Path, key: str, label: str):
    """Return the first shapefile/GeoJSON/GeoPackage whose name contains the city key or label."""
    if outline_dir is None or not outline_dir.exists():
        return None
    keys = {key.lower(), label.lower().replace(" ", "_"), label.lower().replace(" ", "")}
    for p in sorted(outline_dir.rglob("*")):
        if p.suffix.lower() in SHAPE_EXTS and any(k in p.stem.lower() for k in keys):
            return p
    return None


def describe_geometry(label: str, gdf, stage: str = ""):
    """Print the geometry of a boundary so line width can be chosen from real data."""
    n_parts = 0
    n_vertices = 0
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        polys = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        n_parts += len(polys)
        for poly in polys:
            n_vertices += len(poly.exterior.coords) + sum(len(r.coords) for r in poly.interiors)
    minx, miny, maxx, maxy = gdf.total_bounds
    print(f"[geometry] {label} ({stage}): features={len(gdf)} parts={n_parts} vertices={n_vertices} "
          f"bounds lon [{minx:.3f}, {maxx:.3f}] lat [{miny:.3f}, {maxy:.3f}] "
          f"line width={OUTLINE_LW.get(label, 0.9)} pt")


def draw_outline(ax, path: Path, label: str):
    import geopandas as gpd
    gdf = gpd.read_file(path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    describe_geometry(label, gdf, "original")
    # Simplify to about 100 m so dense coastlines do not render heavier than smooth boundaries
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)
    describe_geometry(label, gdf, f"simplified {SIMPLIFY_TOL_DEG}°")
    gdf.boundary.plot(ax=ax, color="black", linewidth=OUTLINE_LW.get(label, 0.9), zorder=4)


def draw_buffer(ax, clat, clon):
    dlat = BUFFER_KM / 111.0
    dlon = BUFFER_KM / (111.0 * np.cos(np.radians(clat)))
    ax.add_patch(Ellipse((clon, clat), width=2 * dlon, height=2 * dlat,
                         fill=False, ls="--", lw=0.8, ec="black", zorder=5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--outline-dir", type=Path, default=None,
                    help="Folder containing the city boundary shapefiles / GeoJSON / GeoPackage")
    ap.add_argument("--no-buffer", action="store_true", help="Do not draw the 30 km buffer")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2))
    im = None
    missing_outline = []

    for i, (ax, (key, label, clat, clon)) in enumerate(zip(axes.flat, PANELS)):
        panel = ["(a)", "(b)", "(c)", "(d)"][i]
        tif = args.data_dir / f"{key}_NO2_persistence_frequency_landonly.tif"
        if not tif.exists():
            raise SystemExit(f"Missing file: {tif}")
        data, raster_extent = read_percent(tif)
        # Same physical size for every panel: a square of 2 * PANEL_HALF_KM around the city
        half_lat = PANEL_HALF_KM / 111.0
        half_lon = PANEL_HALF_KM / (111.0 * np.cos(np.radians(clat)))
        extent = (clon - half_lon, clon + half_lon, clat - half_lat, clat + half_lat)

        im = ax.imshow(np.ma.masked_invalid(data), extent=raster_extent, origin="upper",
                       cmap=CMAP, norm=NORM, interpolation="nearest")
        ax.set_facecolor("#f4f7fb")  # water / no-data pixels
        ax.set_title(f"{panel} {label}", loc="left", fontsize=9, fontweight="bold")

        outline = find_outline(args.outline_dir, key, label)
        if outline is not None:
            draw_outline(ax, outline, label)
        else:
            missing_outline.append(label)

        if not args.no_buffer:
            draw_buffer(ax, clat, clon)

        ax.scatter(clon, clat, s=14, facecolor="white", edgecolor="black", lw=0.6, zorder=6)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        # 1 degree latitude and 1 degree longitude span equal distances in km at this latitude
        ax.set_aspect(1.0 / np.cos(np.radians(clat)))
        ax.tick_params(labelsize=8.5)
        # Set labels explicitly after drawing: geopandas writes "Geodetic ... [degree]" labels when it
        # plots a geographic boundary, so every axis label is overwritten here.
        # x-label on the bottom row only, y-label on the left column only.
        ax.set_xlabel("Longitude (°)" if i >= 2 else "", fontsize=9)
        ax.set_ylabel("Latitude (°)" if i % 2 == 0 else "", fontsize=9)

    if missing_outline:
        print("No outline found for:", ", ".join(missing_outline),
              "- check file names in --outline-dir")

    # Colorbar spans the full height of the panel grid
    cax = fig.add_axes([0.89, GRID["bottom"], 0.018, GRID["top"] - GRID["bottom"]])
    cbar = fig.colorbar(im, cax=cax, ticks=BOUNDS)
    cbar.set_label("High-NO$_2$ persistence frequency (%)", fontsize=10.5)
    cbar.ax.tick_params(labelsize=8.5)

    outline_text = "Solid outline: administrative boundary. " if not missing_outline else ""
    buffer_text = "" if args.no_buffer else "Dashed outline: 30 km analysis buffer. "
    caption = (
        "Figure 6. High-NO$_2$ persistence frequency, 2019–2024: percentage of months in which each land "
        "pixel exceeded the city-wide 75th percentile of monthly land-only city-mean NO$_2$. Water pixels "
        "(JRC water occurrence ≥ 50%) are masked and not interpolated; white pixels inside Johannesburg are "
        "water-classified. " + outline_text + buffer_text + "Panels are "
        f"{2 * PANEL_HALF_KM} km squares centred on each city."
    )
    fig.text(0.02, 0.01, caption, fontsize=6.6, wrap=True, color="#333333")
    fig.subplots_adjust(**GRID)

    for ax in axes.flat:  # guard: no "Geodetic" wording may reach the figure
        assert "geodetic" not in (ax.get_xlabel() + ax.get_ylabel()).lower(), "Geodetic label found"

    png = args.out_dir / "figure6_persistence_landonly.png"
    pdf = args.out_dir / "figure6_persistence_landonly.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved {png}\nSaved {pdf}")


if __name__ == "__main__":
    main()
