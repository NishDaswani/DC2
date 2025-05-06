# Data Integration Tasks Summary (scripts/02_integrate_external_data.py)

This file summarizes the key data loading, processing, and merging steps performed in the `02_integrate_external_data.py` script.

## Initial Loading
1.  **Load Burglary Features:** Loaded the pre-engineered time-based features from `data/features_engineered.csv`. This includes LSOA code, month, burglary count, and various lag/rolling window features.
2.  **Load IMD Data:** Loaded the English Indices of Multiple Deprivation (IMD) 2019 shapefile (`data/English IMD 2019/IMD_2019.shp`) using `geopandas`.

## IMD Integration
3.  **Select & Rename IMD Columns:** Selected the LSOA code (`lsoa11cd`) and IMD score (`IMDScore`) from the IMD GeoDataFrame and renamed `lsoa11cd` to `LSOA code`.
4.  **Merge IMD Score:** Performed a left merge to add the `IMDScore` to the main burglary DataFrame based on `LSOA code`. Noted ~6.4% missing values, likely due to LSOAs in burglary data not present in the IMD file.

## Census Data Setup
5.  **Load OA-LSOA Lookup:** Loaded the postcode/OA/LSOA lookup table (`data/census_data/postcode_oa_lsoa_msoa_lad_2011_ew.csv`). Required adjusting expected column names to uppercase (`OA11CD`, `LSOA11CD`). Selected only OA and LSOA code columns and dropped duplicates.
6.  **Load Census Age Data:** Loaded the 2011 Census age data (`data/census_data/census_age.csv`), which is at the Output Area (OA) level.

## Census Age & Population Density Integration
7.  **Calculate OA Mean Age:** Calculated the mean age for each OA using the counts for individual years of age provided in the census file.
8.  **Merge Age Data with Lookup:** Merged the OA-level mean age data with the lookup table based on the OA code (`geography code` in census, `OA11CD` in lookup).
9.  **Aggregate Mean Age to LSOA:** Grouped the merged data by LSOA code (`LSOA11CD`) and calculated the mean of the OA mean ages within each LSOA to get an LSOA-level mean age (`lsoa_mean_age`).
10. **Merge LSOA Mean Age:** Merged the aggregated `lsoa_mean_age` into the main burglary DataFrame. Noted ~7.4% missing values.
11. **Calculate LSOA Area:** Calculated the area (in sq km) for each LSOA using the geometry column from the loaded IMD GeoDataFrame.
12. **Aggregate Total Population to LSOA:** Used the total population column from the OA-level census age data, merged it with the lookup table, and summed the population for each LSOA.
13. **Calculate LSOA Population Density:** Divided the aggregated LSOA total population by the calculated LSOA area.
14. **Merge Population Density:** Merged the calculated `population_density` into the main burglary DataFrame. Noted ~7.4% missing values.

## Census Accommodation Data (In Progress)
15. **Load Census Accommodation Data:** Loaded the 2011 Census accommodation type data (`data/census_data/census_accommodation.csv`) at the OA level for inspection. Identified relevant columns for total accommodations and specific types (Detached, Semi-detached, Terraced, Flat, Caravan/Mobile).

## Next Steps (as per script TODOs)
*   Calculate accommodation type percentages at OA level.
*   Aggregate accommodation percentages to LSOA level using the lookup table.
*   Merge LSOA-level accommodation percentages into the main DataFrame.
*   Repeat loading, aggregation, and merging for other census files (dwellings, tenure).
*   Load, process, and merge PTAL data.
*   Handle missing values accumulated from merges (imputation).
*   Save the final merged DataFrame. 