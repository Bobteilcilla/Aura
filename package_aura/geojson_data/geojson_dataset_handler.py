"""
Ealing GeoJSON → CSV Conversion Script
-------------------------------------
This script converts a specific local GeoJSON file—
`United Kingdom_England_Ealing.points.geojson`—into a CSV file containing
only its non-geometric attributes.

Execution:
    python geojson_to_csv.py

The script expects the GeoJSON file to be located in the same directory as
this Python file. The resulting CSV will be written alongside it, using the
same basename.
"""

import sys
import os
from pathlib import Path
import geopandas as gpd   ### TO-DO Add geopandas to requirements.txt


def main():
    """Locate the Ealing GeoJSON file and export it as CSV."""

    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / "United Kingdom_England_Ealing.points.geojson"

    if not input_path.exists():
        print(f"Error: Expected GeoJSON not found: {input_path}")
        sys.exit(1)

    print(f"Loading GeoJSON: {input_path}")
    gdf = gpd.read_file(input_path)

    # Remove geometry for a pure attribute table
    df = gdf.drop(columns="geometry") if "geometry" in gdf.columns else gdf.copy()

    output_path = input_path.with_suffix(".csv")
    df.to_csv(output_path, index=False)

    print(f"CSV saved at: {output_path}")


if __name__ == "__main__":
    main()
