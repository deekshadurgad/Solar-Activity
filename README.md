# Color-Magnitude Diagram of Star Clusters

A interactive Streamlit application for visualizing and analyzing Color-Magnitude Diagrams (CMD) of star clusters using Gaia DR3 data. This tool enables astronomers and astrophysics enthusiasts to explore stellar populations, identify main-sequence turn-off points, and compare different clusters.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- 🌟 **Real-time Gaia DR3 Data**: Query the Gaia archive directly for up-to-date stellar data
- 🔍 **Cluster Membership Cleaning**: Uses DBSCAN clustering algorithm to filter cluster members based on proper motion and parallax
- 🌈 **Extinction Correction**: Apply E(B-V) reddening correction to account for interstellar extinction
- 📊 **Interactive Visualization**: Plotly-based interactive CMD plots with spectral class regions (O, B, A, F, G, K, M)
- 🔄 **Cluster Comparison**: Compare two clusters side-by-side on the same diagram
- 📍 **Turn-off Point Detection**: Automatically identifies and marks the main-sequence turn-off point
- 💾 **Data Export**: Download cleaned cluster catalogs in CSV format

## Supported Star Clusters

- **Pleiades** (M45) - Young open cluster
- **Hyades** - Nearest open cluster
- **M67** - Old open cluster
- **NGC 188** - Ancient open cluster
- **NGC 2420** - Intermediate age cluster
- **NGC 6791** - Metal-rich old cluster

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HR_Diagram.git
cd HR_Diagram
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run HR_Diagram.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Controls

**Sidebar Options:**
- **Primary cluster**: Select the main cluster to analyze
- **Enable cluster comparison**: Toggle to compare two clusters
- **Secondary cluster**: Choose a second cluster for comparison (when enabled)
- **Apparent G magnitude limit**: Filter stars by brightness (10-20 mag)
- **E(B-V) reddening**: Apply extinction correction (0.0-0.5)
- **Refresh Gaia data**: Force a new query to Gaia archive

### Workflow

1. Select a cluster from the dropdown menu
2. Adjust the magnitude limit to control the number of stars
3. Apply extinction correction using the E(B-V) slider
4. Enable comparison mode to overlay another cluster
5. Analyze the CMD plot with spectral class annotations
6. Identify the main-sequence turn-off point (marked with X)
7. Download the cleaned catalog for further analysis

## Data Processing Pipeline

1. **Query Gaia DR3**: Retrieves stars within defined RA/Dec boundaries with minimum parallax thresholds
2. **Distance Calculation**: Converts parallax to distance in parsecs
3. **Absolute Magnitude**: Calculates absolute G magnitude from apparent magnitude and distance
4. **Membership Cleaning**: DBSCAN clustering on proper motion (pmra, pmdec) and parallax to isolate cluster members
5. **Extinction Correction**: Applies standard extinction coefficients:
   - A_G = 2.742 × E(B-V)
   - E(BP-RP) = 1.289 × E(B-V)
6. **Turn-off Detection**: Identifies the main-sequence turn-off using gradient analysis

## Technical Details

### Dependencies

- **streamlit**: Web application framework
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **plotly**: Interactive plotting
- **astroquery**: Gaia archive queries
- **astropy**: Astronomical data handling
- **scikit-learn**: DBSCAN clustering algorithm

### Caching

The application uses Streamlit's `@st.cache_data` decorator with a 1-hour TTL to minimize redundant Gaia queries and improve performance.

## Color-Magnitude Diagram Interpretation

- **Y-axis (reversed)**: Absolute G magnitude (brighter stars at top)
- **X-axis**: BP-RP color index (bluer stars on left, redder on right)
- **Main Sequence**: Diagonal band from upper-left to lower-right
- **Turn-off Point**: Where stars leave the main sequence (indicates cluster age)
- **Giant Branch**: Stars moving upward and rightward from turn-off
- **Spectral Classes**: Vertical bands labeled O, B, A, F, G, K, M

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Known Limitations

- Query times may vary depending on Gaia archive server load
- Large magnitude limits may result in slower processing
- DBSCAN parameters (eps=1.5, min_samples=10) are fixed and may not be optimal for all clusters

## Future Enhancements

- [ ] Add more star clusters
- [ ] Customizable DBSCAN parameters
- [ ] Isochrone overlays for age determination
- [ ] HR diagram (luminosity vs. temperature) option
- [ ] Proper motion vector visualization
- [ ] 3D position visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Gaia Mission**: ESA mission providing high-precision astrometric data
- **Gaia DR3**: Third data release of the Gaia mission
- **Astroquery**: Accessing astronomical data archives
- **Streamlit**: Making data apps accessible

## References

- Gaia Collaboration et al. (2023), "Gaia Data Release 3", A&A
- Extinction coefficients from Gaia DR3 documentation
- Cluster parameters from WEBDA database

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for the astronomy community**
