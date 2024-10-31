"""Vector to raster conversion for 2-D soil profile data.

This module provides functionality for converting 2-D soil profile vector data
(shapefiles) to raster format with proper coordinate transformation and
horizon mapping.

Example:
    Basic usage example:

    >>> horizon_mapping, raster_data = rasterize_soil_profile(
    ...     "profile.shp",
    ...     "output.tif",
    ...     220,
    ...     100
    ... )
"""

# Standard library imports
from typing import Tuple, Dict, Any

# Third-party imports
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import Affine
import matplotlib.pyplot as plt
from shapely.affinity import scale, translate


def transform_geometry(
    geom: Any,
    minx: float,
    miny: float,
    x_scale: float,
    y_scale: float,
    depth_cm: int
) -> Any:
    """Transform a geometry from original coordinates to pixel space.

    Args:
        geom: Input geometry
        minx: Minimum x coordinate
        miny: Minimum y coordinate
        x_scale: X scaling factor
        y_scale: Y scaling factor
        depth_cm: Target depth in centimeters

    Returns:
        Transformed geometry
    """
    geom = translate(geom, xoff=-minx, yoff=-miny)
    geom = scale(geom, xfact=x_scale, yfact=-y_scale, origin=(0, 0))
    return translate(geom, xoff=0, yoff=depth_cm)


def create_verification_plots(
    gdf: gpd.GeoDataFrame,
    burned: np.ndarray,
    width_cm: int,
    depth_cm: int
) -> None:
    """Create verification plots for the rasterization process.

    Args:
        gdf: GeoDataFrame with transformed geometries
        burned: Rasterized data
        width_cm: Width in centimeters
        depth_cm: Depth in centimeters
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

    # Plot transformed geometries
    gdf.plot(column='horizon_value', ax=ax1, legend=True)
    ax1.set_title('Transformed Geometries')
    ax1.set_xlim(0, width_cm)
    ax1.set_ylim(0, depth_cm)

    # Plot rasterized result
    im = ax2.imshow(burned, cmap='tab20', aspect='equal')
    ax2.set_title('Rasterized Result')
    plt.colorbar(im, ax=ax2, label='Horizon Values')

    plt.tight_layout()
    plt.show()


def rasterize_soil_profile(
    shp_path: str,
    output_raster_path: str,
    width_cm: int,
    depth_cm: int,
    horizon_field: str = 'hziid'
) -> Tuple[Dict, np.ndarray]:
    """Rasterize a soil profile shapefile with coordinate transformation.

    Args:
        shp_path: Path to input shapefile
        output_raster_path: Path for output GeoTIFF
        width_cm: Target width in centimeters
        depth_cm: Target depth in centimeters
        horizon_field: Field name containing horizon identifiers

    Returns:
        Tuple containing:
            - horizon_mapping: Dictionary mapping horizons to numeric values
            - burned: Numpy array containing rasterized data

    Raises:
        FileNotFoundError: If input shapefile does not exist
        ValueError: If input parameters are invalid
    """
    # Read the shapefile
    gdf = gpd.read_file(shp_path)

    # Create horizon mapping
    unique_horizons = sorted(gdf[horizon_field].unique())
    horizon_mapping = {
        horizon: idx for idx, horizon in enumerate(unique_horizons, start=1)
    }
    horizon_reverse_mapping = {
        str(idx): horizon for horizon, idx in horizon_mapping.items()
    }

    gdf['horizon_value'] = gdf[horizon_field].map(horizon_mapping)

    # Get bounds and calculate scaling
    minx, miny, maxx, maxy = gdf.total_bounds
    x_scale = width_cm / (maxx - minx)
    y_scale = depth_cm / (maxy - miny)

    # Transform geometries
    gdf.geometry = gdf.geometry.apply(
        lambda geom: transform_geometry(
            geom,
            minx,
            miny,
            x_scale,
            y_scale,
            depth_cm
        )
    )

    # Set up rasterization
    transform = Affine.translation(0, 0) * Affine.scale(1, 1)
    shapes = (
        (geom, int(value))
        for geom, value in zip(gdf.geometry, gdf.horizon_value)
    )

    # Create raster
    burned = features.rasterize(
        shapes=shapes,
        out_shape=(depth_cm, width_cm),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
        merge_alg=rasterio.enums.MergeAlg.replace
    )

    # Save output
    raster_profile = {
        'driver': 'GTiff',
        'height': depth_cm,
        'width': width_cm,
        'count': 1,
        'dtype': np.uint8,
        'crs': None,
        'transform': transform
    }

    with rasterio.open(output_raster_path, 'w', **raster_profile) as dst:
        dst.write(burned, 1)
        for value, horizon in horizon_reverse_mapping.items():
            dst.update_tags(**{f'horizon_{value}': horizon})

    # Create verification plots
    create_verification_plots(gdf, burned, width_cm, depth_cm)

    # Print verification info
    print("\nRaster statistics:")
    print(f"Unique values in result: {np.unique(burned)}")
    print(f"Horizon mapping: {horizon_mapping}")

    return horizon_mapping, burned


if __name__ == "__main__":
    # Example usage
    EXAMPLE_SHP_PATH = "HS 2-2-combined.shp"
    EXAMPLE_OUTPUT_PATH = "HS 2-2-combined_raster.tif"
    EXAMPLE_WIDTH = 220
    EXAMPLE_DEPTH = 100

    horizon_mapping, raster_data = rasterize_soil_profile(
        EXAMPLE_SHP_PATH,
        EXAMPLE_OUTPUT_PATH,
        EXAMPLE_WIDTH,
        EXAMPLE_DEPTH
    )
