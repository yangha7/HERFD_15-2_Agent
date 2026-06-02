"""
HERFD XAS Data Utilities
========================
Parsing, averaging, normalization, and export functions for hard X-ray
HERFD (High Energy Resolution Fluorescence Detected) XAS data collected
at SSRL in SPEC format.

Key capabilities:
  - Parse SPEC-format .dat files (header metadata + columnar data)
  - Identify XAS scans vs. non-XAS scans (alignment, emission, etc.)
  - Average multiple XAS scans with interpolation onto a common grid
  - Athena-style pre-edge subtraction / post-edge normalization
  - Savitzky-Golay smoothed 1st & 2nd derivatives
  - Export averaged and normalized spectra

Data conventions:
  - Energy column: 'energy' (monochromator setpoint) or 'absev' (calibrated)
  - HERFD signal: 'vortDT' (deadtime-corrected vortex detector counts)
  - Normalization: vortDT / I0 (ion chamber)
  - Background-subtracted I0: bgI0 = I0 - dark current

All file I/O is on-demand — nothing is loaded at import time.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("HERFD_DATA_DIR", "2026-05_Yano")
EXPORT_DIR = os.environ.get("HERFD_EXPORT_DIR", "exported_data")

# XAS edge database for hard X-ray edges (SSRL BL 15-2, ~4.5-37 keV)
XAS_EDGE_DB = {
    # 3d transition metals — K edges
    "Ti": {"K": 4966.0}, "V": {"K": 5470.0}, "Cr": {"K": 5989.0},
    "Mn": {"K": 6539.0}, "Fe": {"K": 7112.0}, "Co": {"K": 7709.0},
    "Ni": {"K": 8333.0}, "Cu": {"K": 8979.0}, "Zn": {"K": 9659.0},
    # 4d transition metals — K edges
    "Zr": {"K": 17998.0}, "Nb": {"K": 18986.0}, "Mo": {"K": 19999.0},
    "Tc": {"K": 21044.0}, "Ru": {"K": 22117.0}, "Rh": {"K": 23220.0},
    "Pd": {"K": 24350.0}, "Ag": {"K": 25514.0}, "Cd": {"K": 26711.0},
    # 5d transition metals — L edges
    "Hf": {"L3": 9561.0, "L2": 10739.0}, "Ta": {"L3": 9881.0, "L2": 11136.0},
    "W": {"L3": 10207.0, "L2": 11544.0}, "Re": {"L3": 10535.0, "L2": 11959.0},
    "Os": {"L3": 10871.0, "L2": 12385.0}, "Ir": {"L3": 11215.0, "L2": 12824.0},
    "Pt": {"L3": 11564.0, "L2": 13273.0}, "Au": {"L3": 11919.0, "L2": 13734.0},
    # Lanthanides — L edges
    "La": {"L3": 5483.0, "L2": 5891.0}, "Ce": {"L3": 5723.0, "L2": 6164.0},
    "Pr": {"L3": 5964.0, "L2": 6440.0}, "Nd": {"L3": 6208.0, "L2": 6722.0},
    "Sm": {"L3": 6716.0, "L2": 7312.0}, "Eu": {"L3": 6977.0, "L2": 7617.0},
    "Gd": {"L3": 7243.0, "L2": 7930.0}, "Tb": {"L3": 7514.0, "L2": 8252.0},
    "Dy": {"L3": 7790.0, "L2": 8581.0}, "Ho": {"L3": 8071.0, "L2": 8918.0},
    "Er": {"L3": 8358.0, "L2": 9264.0}, "Yb": {"L3": 8944.0, "L2": 9978.0},
    # Other elements
    "Ga": {"K": 10367.0}, "Ge": {"K": 11103.0}, "As": {"K": 11867.0},
    "Se": {"K": 12658.0}, "Br": {"K": 13474.0}, "Sr": {"K": 16105.0},
    "Y": {"K": 17038.0},
    # Actinides — L edges
    "Th": {"L3": 16300.0}, "U": {"L3": 17166.0},
    "Np": {"L3": 17610.0}, "Pu": {"L3": 18057.0},
}


def identify_element_from_energy(energy_range: tuple[float, float],
                                  metadata_text: str = "") -> list[dict]:
    """Identify candidate element/edge from the energy range of a scan.

    Parameters
    ----------
    energy_range : (float, float)
        (min_energy, max_energy) in eV from the scan data.
    metadata_text : str
        Any metadata text (sample name, directory name) to search for
        element symbols.

    Returns
    -------
    list of dict: [{"element": str, "edge": str, "ref_energy": float,
                     "source": str}, ...]
        Candidates sorted by likelihood. "source" is "metadata" or "energy".
    """
    candidates = []
    e_start, e_end = energy_range
    e_mid = (e_start + e_end) / 2

    # 1. Check metadata for element symbols
    if metadata_text:
        # Look for element symbols in the text (2-letter or 1-letter)
        for elem, edges in XAS_EDGE_DB.items():
            # Case-sensitive match for element symbols
            if re.search(rf'\b{elem}\b|{elem}[^a-z]|[^a-zA-Z]{elem}', metadata_text):
                for edge_name, ref_e in edges.items():
                    # Check if the edge energy is within the scan range
                    if e_start - 100 <= ref_e <= e_end + 100:
                        candidates.append({
                            "element": elem,
                            "edge": edge_name,
                            "ref_energy": ref_e,
                            "source": "metadata",
                            "delta": abs(e_mid - ref_e),
                        })

    # 2. Find edges whose energy falls within the scan range
    for elem, edges in XAS_EDGE_DB.items():
        for edge_name, ref_e in edges.items():
            if e_start <= ref_e <= e_end:
                # Check if already found via metadata
                already = any(c["element"] == elem and c["edge"] == edge_name
                              for c in candidates)
                if not already:
                    candidates.append({
                        "element": elem,
                        "edge": edge_name,
                        "ref_energy": ref_e,
                        "source": "energy_range",
                        "delta": abs(e_mid - ref_e),
                    })

    # 3. If no exact match, find nearest edges
    if not candidates:
        for elem, edges in XAS_EDGE_DB.items():
            for edge_name, ref_e in edges.items():
                delta = min(abs(e_start - ref_e), abs(e_end - ref_e))
                if delta < 200:  # within 200 eV
                    candidates.append({
                        "element": elem,
                        "edge": edge_name,
                        "ref_energy": ref_e,
                        "source": "nearest",
                        "delta": delta,
                    })

    # Sort: metadata matches first, then by distance to mid-energy
    def sort_key(c):
        source_priority = {"metadata": 0, "energy_range": 1, "nearest": 2}
        return (source_priority.get(c["source"], 3), c["delta"])

    candidates.sort(key=sort_key)
    return candidates


# ---------------------------------------------------------------------------
# SPEC File Parsing
# ---------------------------------------------------------------------------

def parse_spec_header(filepath: str) -> dict:
    """Parse the SPEC-format header from a .dat file.

    Extracts metadata from #S, #D, #T, #P, #N, #L lines.

    Returns
    -------
    dict with keys:
        scan_number : int
        scan_command : str (full #S line content)
        scan_type : str ('gscan', 'ascan', 'a2scan', etc.)
        scan_motor : str (first motor name, e.g. 'energy', 'Sz', 'emiss')
        date : str
        count_time : float (seconds)
        n_columns : int
        column_names : list[str]
        motor_positions : dict[str, float]
        is_xas : bool (True if scan_motor == 'energy')
        is_aborted : bool
        filepath : str
    """
    meta = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "motor_positions": {},
        "column_names": [],
        "is_aborted": False,
    }

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip("\n")

        if line.startswith("#S "):
            # #S 6  gscan energy 22070 22100 2 ...
            parts = line[3:].strip().split(None, 1)
            meta["scan_number"] = int(parts[0])
            meta["scan_command"] = parts[1] if len(parts) > 1 else ""
            # Parse scan type and motor
            cmd_parts = meta["scan_command"].split()
            if cmd_parts:
                meta["scan_type"] = cmd_parts[0]
                meta["scan_motor"] = cmd_parts[1] if len(cmd_parts) > 1 else ""
            else:
                meta["scan_type"] = ""
                meta["scan_motor"] = ""

        elif line.startswith("#D "):
            meta["date"] = line[3:].strip()

        elif line.startswith("#T "):
            # #T 6  (sec)
            m = re.match(r"#T\s+([\d.]+)", line)
            if m:
                meta["count_time"] = float(m.group(1))

        elif line.startswith("#N "):
            meta["n_columns"] = int(line[3:].strip())

        elif line.startswith("#L "):
            meta["column_names"] = line[3:].strip().split("  ")
            # Clean up: split on double-space, then strip each
            meta["column_names"] = [c.strip() for c in meta["column_names"] if c.strip()]

        elif line.startswith("#P"):
            # Motor positions: #P0 dummy0=3 energy=22069.99 ...
            pairs = re.findall(r"(\w+)=([\d.eE+-]+)", line)
            for name, val in pairs:
                try:
                    meta["motor_positions"][name] = float(val)
                except ValueError:
                    meta["motor_positions"][name] = val

        elif line.startswith("#C ") and "aborted" in line.lower():
            meta["is_aborted"] = True

    # Determine if this is an XAS scan
    meta["is_xas"] = (
        meta.get("scan_type", "") in ("gscan",)
        and meta.get("scan_motor", "") == "energy"
        and not meta.get("is_aborted", False)
    )

    return meta


def load_spec_scan(filepath: str) -> pd.DataFrame:
    """Load a SPEC-format .dat scan file into a DataFrame.

    Skips all header lines (starting with #) and reads the columnar data.
    Column names are taken from the #L header line.

    Returns
    -------
    pd.DataFrame with named columns
    """
    column_names = []
    data_lines = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#L "):
                # Parse column names from #L line
                column_names = [c.strip() for c in line[3:].strip().split("  ") if c.strip()]
            elif line.startswith("#") or not line.strip():
                continue
            else:
                data_lines.append(line)

    if not data_lines:
        return pd.DataFrame()

    # Parse data
    data = []
    for line in data_lines:
        vals = line.split()
        try:
            row = [float(v) for v in vals]
            data.append(row)
        except ValueError:
            continue

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Assign column names if they match
    if column_names and len(column_names) == df.shape[1]:
        df.columns = column_names
    elif column_names:
        # Try to match as many as possible
        n = min(len(column_names), df.shape[1])
        df.columns = list(column_names[:n]) + [f"col_{i}" for i in range(n, df.shape[1])]

    return df


def classify_scan(filepath: str) -> dict:
    """Classify a scan file and return its metadata with scan type info.

    Returns
    -------
    dict with all parse_spec_header fields plus:
        classification : str
            'xas' - energy scan (gscan energy)
            'alignment_z' - sample Z position scan
            'alignment_xy' - sample X/Y position scan
            'emission' - emission energy scan
            'other' - unrecognized scan type
            'aborted' - scan was aborted
        n_points : int (number of data points)
    """
    meta = parse_spec_header(filepath)

    if meta.get("is_aborted", False):
        meta["classification"] = "aborted"
        meta["n_points"] = 0
        return meta

    # Count data points
    n_points = 0
    with open(filepath, "r") as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                n_points += 1
    meta["n_points"] = n_points

    scan_type = meta.get("scan_type", "")
    scan_motor = meta.get("scan_motor", "")

    if meta["is_xas"]:
        meta["classification"] = "xas"
    elif scan_motor in ("Sz", "Sy", "Sx"):
        meta["classification"] = "alignment_z" if scan_motor == "Sz" else "alignment_xy"
    elif scan_motor == "emiss":
        meta["classification"] = "emission"
    elif scan_type == "a2scan":
        meta["classification"] = "alignment_xy"
    else:
        meta["classification"] = "other"

    return meta


# ---------------------------------------------------------------------------
# Directory / Sample Discovery
# ---------------------------------------------------------------------------

# Directories to skip when auto-discovering samples
_SKIP_DIRS = {"alignment_dir", "test_dir", "test2_dir", "test3_dir",
              "average_pymca", "exported_data"}

# Minimum number of data points for a scan to be included in averaging
MIN_XAS_POINTS = 20


def list_sample_dirs(data_dir: str = DATA_DIR) -> list[str]:
    """List all sample data directories (ending with _dir).

    Excludes known non-sample directories (alignment, test, etc.).
    Returns sorted list of directory names relative to data_dir.
    """
    dirs = []
    base = Path(data_dir)
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and entry.name.endswith("_dir"):
            if entry.name not in _SKIP_DIRS:
                dirs.append(entry.name)
    return dirs


def list_scans_in_dir(dir_path: str) -> list[dict]:
    """List and classify all .dat scan files in a directory.

    Returns list of classification dicts, sorted by scan number.
    """
    scans = []
    for f in sorted(glob.glob(os.path.join(dir_path, "*.dat"))):
        try:
            info = classify_scan(f)
            scans.append(info)
        except Exception as e:
            scans.append({
                "filepath": f,
                "filename": os.path.basename(f),
                "classification": "error",
                "error": str(e),
            })
    return sorted(scans, key=lambda s: s.get("scan_number", 0))


def get_xas_scans(dir_path: str, min_points: int = MIN_XAS_POINTS) -> list[dict]:
    """Return only XAS scan metadata from a directory.
    
    Filters out scans with fewer than min_points data points.
    """
    return [s for s in list_scans_in_dir(dir_path) 
            if s.get("classification") == "xas" and s.get("n_points", 0) >= min_points]


def parse_sample_name(dir_name: str) -> dict:
    """Parse sample information from directory name.

    Examples:
        '20260519_RuO2-TiO2_100_grazing_dir' ->
            {'date': '20260519', 'sample': 'RuO2-TiO2', 'orientation': '100',
             'geometry': 'grazing', 'temperature': None}
        '20260521_RuO2-TiO2_110_500C_normal_dir' ->
            {'date': '20260521', 'sample': 'RuO2-TiO2', 'orientation': '110',
             'geometry': 'normal', 'temperature': '500C'}
    """
    name = dir_name.replace("_dir", "")
    parts = name.split("_")

    info = {
        "date": parts[0] if parts else "",
        "sample": parts[1] if len(parts) > 1 else "",
        "orientation": "",
        "geometry": "",
        "temperature": None,
        "full_name": name,
    }

    # Find orientation (3-digit number like 100, 101, 110, 111)
    for i, p in enumerate(parts[2:], start=2):
        if re.match(r"^\d{3}$", p):
            info["orientation"] = p
            remaining = parts[i + 1:]
            break
    else:
        remaining = parts[2:]

    # Parse remaining parts for temperature and geometry
    for p in remaining:
        if re.match(r"^\d+C$", p):
            info["temperature"] = p
        elif p in ("grazing", "normal"):
            info["geometry"] = p

    return info


# ---------------------------------------------------------------------------
# Signal Extraction & Averaging
# ---------------------------------------------------------------------------

def get_herfd_signal(df: pd.DataFrame, energy_col: str = "energy",
                     use_absev: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Extract energy and HERFD signal (vortDT/I0) from a scan DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded scan data.
    energy_col : str
        Which energy column to use. Default 'energy' (mono setpoint).
    use_absev : bool
        If True and 'absev' column exists, use calibrated energy instead.

    Returns
    -------
    (energy, mu) : tuple of np.ndarray
        Energy in eV and normalized HERFD signal.
    """
    # Choose energy column
    if use_absev and "absev" in df.columns:
        energy = df["absev"].values.astype(float)
    elif energy_col in df.columns:
        energy = df[energy_col].values.astype(float)
    else:
        # Fallback: first column
        energy = df.iloc[:, 0].values.astype(float)

    # HERFD signal: vortDT / I0
    if "vortDT" in df.columns and "I0" in df.columns:
        vort = df["vortDT"].values.astype(float)
        i0 = df["I0"].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = np.where(i0 != 0, vort / i0, 0.0)
    elif "vortDT" in df.columns:
        mu = df["vortDT"].values.astype(float)
    elif "I1" in df.columns and "I0" in df.columns:
        # Fallback: transmission mode
        i1 = df["I1"].values.astype(float)
        i0 = df["I0"].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = np.where((i0 != 0) & (i1 != 0), -np.log(i1 / i0), 0.0)
    else:
        raise ValueError(f"Cannot find HERFD signal columns. Available: {list(df.columns)}")

    return energy, mu


def average_scans(
    filepaths: list[str],
    energy_col: str = "energy",
    use_absev: bool = True,
    e_min: float | None = None,
    e_max: float | None = None,
    e_step: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Average multiple XAS scans by interpolating onto a common energy grid.

    Uses the energy grid from the scan with the most data points as the
    reference, then interpolates all other scans onto it. This handles
    heterogeneous energy grids (e.g. coarse survey scans mixed with fine
    near-edge scans) by using the densest grid as the template.

    Parameters
    ----------
    filepaths : list[str]
        Paths to .dat scan files.
    energy_col : str
        Energy column name.
    use_absev : bool
        Use calibrated energy ('absev') if available.
    e_min, e_max : float or None
        Energy range for the common grid. If None, uses the range of
        the densest scan.
    e_step : float or None
        Energy step for interpolation grid. If None, uses the median
        step from the densest scan.

    Returns
    -------
    (energy_grid, mu_avg, used_files) : tuple
        Common energy grid, averaged mu, and list of files actually used.
    """
    spectra = []
    used_files = []

    for fp in filepaths:
        try:
            df = load_spec_scan(fp)
            if df.empty:
                continue
            energy, mu = get_herfd_signal(df, energy_col, use_absev)
            if len(energy) < MIN_XAS_POINTS:
                continue
            # Sort by energy
            sort_idx = np.argsort(energy)
            energy = energy[sort_idx]
            mu = mu[sort_idx]
            # Remove duplicates
            unique_mask = np.diff(energy, prepend=-np.inf) > 0
            energy = energy[unique_mask]
            mu = mu[unique_mask]
            spectra.append((energy, mu))
            used_files.append(fp)
        except Exception as e:
            print(f"Warning: skipping {fp}: {e}")
            continue

    if not spectra:
        raise ValueError("No valid scans to average")

    # Find the scan with the most points (densest grid) as reference
    ref_idx = max(range(len(spectra)), key=lambda i: len(spectra[i][0]))
    ref_energy = spectra[ref_idx][0]

    # Determine energy range
    grid_emin = ref_energy.min()
    grid_emax = ref_energy.max()
    if e_min is not None:
        grid_emin = max(grid_emin, e_min)
    if e_max is not None:
        grid_emax = min(grid_emax, e_max)

    # Determine step size from the reference scan
    if e_step is None:
        mask = (ref_energy >= grid_emin) & (ref_energy <= grid_emax)
        e_sub = ref_energy[mask]
        if len(e_sub) > 1:
            e_step = np.median(np.diff(e_sub))
        else:
            e_step = 0.3  # default for HERFD near-edge

    # Create common grid
    energy_grid = np.arange(grid_emin, grid_emax + e_step / 2, e_step)

    # Interpolate each spectrum onto the common grid and average
    interpolated = []
    for energy, mu in spectra:
        # Only interpolate within the scan's own energy range
        f_interp = interp1d(energy, mu, kind="linear", bounds_error=False,
                            fill_value=np.nan)
        mu_interp = f_interp(energy_grid)
        interpolated.append(mu_interp)

    mu_stack = np.array(interpolated)
    # Average with nanmean to handle scans that don't cover the full range
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_avg = np.nanmean(mu_stack, axis=0)

    # Remove any NaN points at edges
    valid = ~np.isnan(mu_avg)
    energy_grid = energy_grid[valid]
    mu_avg = mu_avg[valid]

    return energy_grid, mu_avg, used_files


def average_sample_dir(
    dir_path: str,
    scan_indices: list[int] | None = None,
    use_absev: bool = True,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Average all XAS scans in a sample directory.

    Parameters
    ----------
    dir_path : str
        Path to the sample _dir directory.
    scan_indices : list[int] or None
        If provided, only average scans with these scan numbers.
        If None, average all XAS scans.
    use_absev : bool
        Use calibrated energy.
    **kwargs
        Additional arguments passed to average_scans().

    Returns
    -------
    (energy, mu_avg, used_files)
    """
    xas_scans = get_xas_scans(dir_path)

    if scan_indices is not None:
        xas_scans = [s for s in xas_scans if s["scan_number"] in scan_indices]

    if not xas_scans:
        raise ValueError(f"No XAS scans found in {dir_path}")

    filepaths = [s["filepath"] for s in xas_scans]
    return average_scans(filepaths, use_absev=use_absev, **kwargs)


# ---------------------------------------------------------------------------
# XAS Normalization (Athena-style)
# ---------------------------------------------------------------------------

def find_e0(energy: np.ndarray, mu: np.ndarray) -> float:
    """Find E0 as the energy of the maximum of the 1st derivative of mu.

    Uses a Savitzky-Golay smoothed derivative to avoid noise artifacts.
    """
    window = max(5, len(energy) // 20)
    if window % 2 == 0:
        window += 1
    window = min(window, len(energy) - 2)
    if window < 5:
        window = 5
    deriv = savgol_filter(mu, window_length=window, polyorder=3, deriv=1,
                          delta=np.mean(np.diff(energy)))
    return float(energy[np.argmax(np.abs(deriv))])


def normalize_xanes(
    energy: np.ndarray,
    mu: np.ndarray,
    e0: float | None = None,
    pre_edge_range: tuple[float, float] = (-50, -15),
    post_edge_range: tuple[float, float] = (30, None),
) -> dict:
    """Athena-style pre-edge subtraction and post-edge normalization.

    Adapted for hard X-ray HERFD-XANES with wider energy ranges than
    soft X-ray data.

    Parameters
    ----------
    energy : array
        Energy in eV.
    mu : array
        Absorption signal (vortDT/I0).
    e0 : float or None
        Edge energy. If None, determined automatically.
    pre_edge_range : (float, float)
        Energy range relative to E0 for pre-edge line fit (eV).
        Default: (-50, -15) means E0-50 to E0-15.
    post_edge_range : (float, float)
        Energy range relative to E0 for post-edge fit (eV).
        Default: (30, None) means E0+30 to end of data.

    Returns
    -------
    dict with keys:
        e0, pre_edge_line, post_edge_line, edge_step,
        norm (normalized mu), flat (flattened mu)
    """
    if e0 is None:
        e0 = find_e0(energy, mu)

    # ── Pre-edge line (linear fit) ────────────────────────────────────────
    pre_lo = e0 + pre_edge_range[0]
    pre_hi = e0 + pre_edge_range[1]
    pre_lo = max(pre_lo, energy.min())
    pre_hi = min(pre_hi, e0 - 1)
    pre_mask = (energy >= pre_lo) & (energy <= pre_hi)
    if pre_mask.sum() < 3:
        # Fallback: use first 10% of data
        n10 = max(3, len(energy) // 10)
        pre_mask = np.zeros(len(energy), dtype=bool)
        pre_mask[:n10] = True
    pre_coeffs = np.polyfit(energy[pre_mask], mu[pre_mask], 1)
    pre_edge_line = np.polyval(pre_coeffs, energy)

    # ── Post-edge line (quadratic fit) ────────────────────────────────────
    post_lo = e0 + post_edge_range[0]
    post_hi = e0 + post_edge_range[1] if post_edge_range[1] is not None else energy.max()
    post_lo = max(post_lo, e0 + 5)
    post_hi = min(post_hi, energy.max())
    post_mask = (energy >= post_lo) & (energy <= post_hi)
    if post_mask.sum() < 3:
        # Fallback: use last 20% of data
        n20 = max(3, len(energy) // 5)
        post_mask = np.zeros(len(energy), dtype=bool)
        post_mask[-n20:] = True
    post_coeffs = np.polyfit(energy[post_mask], mu[post_mask], 2)
    post_edge_line = np.polyval(post_coeffs, energy)

    # ── Edge step ─────────────────────────────────────────────────────────
    edge_step = np.polyval(post_coeffs, e0) - np.polyval(pre_coeffs, e0)
    if abs(edge_step) < 1e-12:
        edge_step = 1.0  # avoid division by zero

    # ── Normalized mu ─────────────────────────────────────────────────────
    norm = (mu - pre_edge_line) / edge_step

    # ── Flattened (remove post-edge slope from normalized) ────────────────
    norm_post = norm[post_mask]
    if len(norm_post) >= 3:
        flat_coeffs = np.polyfit(energy[post_mask], norm_post, 1)
        flat = norm.copy()
        above_e0 = energy >= e0
        flat[above_e0] = norm[above_e0] - (np.polyval(flat_coeffs, energy[above_e0]) - 1.0)
    else:
        flat = norm

    return {
        "e0": e0,
        "pre_edge_line": pre_edge_line,
        "post_edge_line": post_edge_line,
        "edge_step": edge_step,
        "norm": norm,
        "flat": flat,
    }


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------

def smooth_derivative(
    energy: np.ndarray,
    mu: np.ndarray,
    order: int = 1,
    window: int | None = None,
    polyorder: int = 3,
) -> np.ndarray:
    """Compute the smoothed nth derivative of mu(E) using Savitzky-Golay filter.

    Parameters
    ----------
    energy : array
        Energy in eV.
    mu : array
        Absorption signal.
    order : int
        Derivative order (1 or 2).
    window : int or None
        Savitzky-Golay window length (must be odd). If None, auto-selected.
    polyorder : int
        Polynomial order for the filter (default 3).

    Returns
    -------
    array : the derivative dⁿmu/dEⁿ
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    if window is None:
        window = max(7, len(energy) // 20)
        if window % 2 == 0:
            window += 1
    window = min(window, len(energy) - 2)
    if window % 2 == 0:
        window += 1
    if polyorder >= window:
        polyorder = window - 1

    delta = np.mean(np.diff(energy))
    return savgol_filter(mu, window_length=window, polyorder=polyorder,
                         deriv=order, delta=delta)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def ensure_export_dir(subdir: str | None = None) -> str:
    """Create export directory if needed and return its path."""
    path = os.path.join(EXPORT_DIR, subdir) if subdir else EXPORT_DIR
    os.makedirs(path, exist_ok=True)
    return path


def export_spectrum(
    energy: np.ndarray,
    mu: np.ndarray,
    filename: str,
    subdir: str | None = None,
    header: str = "",
    extra_columns: dict[str, np.ndarray] | None = None,
) -> str:
    """Export a spectrum (energy, mu) to a text file.

    Parameters
    ----------
    energy : array
        Energy in eV.
    mu : array
        Signal values.
    filename : str
        Output filename.
    subdir : str or None
        Subdirectory within EXPORT_DIR.
    header : str
        Header comment to include.
    extra_columns : dict or None
        Additional columns to include (name -> array).

    Returns
    -------
    str : path to the exported file.
    """
    out_dir = ensure_export_dir(subdir)

    if not filename.endswith((".dat", ".txt", ".csv")):
        filename += ".dat"

    out_path = os.path.join(out_dir, filename)

    # Build data array
    columns = {"energy_eV": energy, "mu": mu}
    if extra_columns:
        columns.update(extra_columns)

    col_names = list(columns.keys())
    data = np.column_stack([columns[c] for c in col_names])

    header_line = header + "\n" if header else ""
    header_line += "\t".join(col_names)

    np.savetxt(out_path, data, delimiter="\t", header=header_line,
               comments="# ", fmt="%.10g")
    return out_path


def export_normalized(
    energy: np.ndarray,
    mu: np.ndarray,
    norm_result: dict,
    filename: str,
    subdir: str = "normalized",
) -> str:
    """Export a normalized spectrum with all normalization components.

    Exports columns: energy_eV, mu_raw, mu_norm, mu_flat, pre_edge, post_edge
    """
    extra = {
        "mu_norm": norm_result["norm"],
        "mu_flat": norm_result["flat"],
        "pre_edge": norm_result["pre_edge_line"],
        "post_edge": norm_result["post_edge_line"],
    }
    header = f"E0 = {norm_result['e0']:.2f} eV, edge_step = {norm_result['edge_step']:.6g}"
    return export_spectrum(energy, mu, filename, subdir=subdir,
                           header=header, extra_columns=extra)


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def process_all_samples(
    data_dir: str = DATA_DIR,
    use_absev: bool = True,
    normalize: bool = True,
    export: bool = True,
    e0: float | None = None,
    pre_edge_range: tuple[float, float] = (-50, -15),
    post_edge_range: tuple[float, float] = (30, None),
) -> list[dict]:
    """Process all sample directories: average XAS scans and normalize.

    Parameters
    ----------
    data_dir : str
        Root data directory.
    use_absev : bool
        Use calibrated energy.
    normalize : bool
        Apply XANES normalization.
    export : bool
        Export results to files.
    e0 : float or None
        Edge energy for normalization. If None, auto-detect.
    pre_edge_range, post_edge_range : tuple
        Ranges for normalization fitting.

    Returns
    -------
    list[dict] : Results for each sample, with keys:
        sample_name, dir_path, n_scans, energy, mu_avg,
        norm_result (if normalize=True), export_path (if export=True)
    """
    results = []
    sample_dirs = list_sample_dirs(data_dir)

    for dir_name in sample_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        sample_info = parse_sample_name(dir_name)

        try:
            energy, mu_avg, used_files = average_sample_dir(
                dir_path, use_absev=use_absev
            )
        except ValueError as e:
            print(f"Skipping {dir_name}: {e}")
            continue

        result = {
            "sample_name": sample_info["full_name"],
            "sample_info": sample_info,
            "dir_path": dir_path,
            "dir_name": dir_name,
            "n_scans": len(used_files),
            "used_files": [os.path.basename(f) for f in used_files],
            "energy": energy,
            "mu_avg": mu_avg,
        }

        if normalize:
            norm_result = normalize_xanes(
                energy, mu_avg, e0=e0,
                pre_edge_range=pre_edge_range,
                post_edge_range=post_edge_range,
            )
            result["norm_result"] = norm_result
            result["e0"] = norm_result["e0"]
            result["edge_step"] = norm_result["edge_step"]

        if export:
            # Export averaged spectrum
            avg_path = export_spectrum(
                energy, mu_avg,
                f"{sample_info['full_name']}_avg.dat",
                subdir="averaged",
                header=f"Average of {len(used_files)} scans from {dir_name}",
            )
            result["avg_export_path"] = avg_path

            if normalize:
                # Export normalized spectrum
                norm_path = export_normalized(
                    energy, mu_avg, norm_result,
                    f"{sample_info['full_name']}_norm.dat",
                    subdir="normalized",
                )
                result["norm_export_path"] = norm_path

        results.append(result)
        print(f"✓ {sample_info['full_name']}: {len(used_files)} scans averaged"
              + (f", E0={norm_result['e0']:.1f} eV" if normalize else ""))

    return results


# ---------------------------------------------------------------------------
# Summary / Reporting
# ---------------------------------------------------------------------------

def summarize_directory(dir_path: str) -> str:
    """Generate a text summary of all scans in a directory.

    Returns a formatted string with scan classifications.
    """
    scans = list_scans_in_dir(dir_path)
    lines = [f"Directory: {dir_path}", f"Total files: {len(scans)}", ""]

    xas_count = 0
    for s in scans:
        cls = s.get("classification", "unknown")
        num = s.get("scan_number", "?")
        cmd = s.get("scan_command", "")[:60]
        npts = s.get("n_points", 0)
        marker = "  ✓" if cls == "xas" else "   "
        if cls == "xas":
            xas_count += 1
        lines.append(f"{marker} #{num:3d}  [{cls:12s}]  {npts:4d} pts  {cmd}")

    lines.insert(3, f"XAS scans: {xas_count}")
    lines.insert(4, "")
    return "\n".join(lines)


def print_all_summaries(data_dir: str = DATA_DIR):
    """Print scan summaries for all sample directories."""
    for dir_name in list_sample_dirs(data_dir):
        dir_path = os.path.join(data_dir, dir_name)
        print(summarize_directory(dir_path))
        print("=" * 70)


# ---------------------------------------------------------------------------
# Convenience: Quick Plot Data (returns data for plotting)
# ---------------------------------------------------------------------------

def get_plot_data(
    dir_path: str,
    scan_indices: list[int] | None = None,
    normalize_flag: bool = False,
    use_absev: bool = True,
) -> dict:
    """Get averaged (and optionally normalized) data ready for plotting.

    Returns
    -------
    dict with keys:
        energy, mu_avg, mu_norm (if normalized), mu_flat (if normalized),
        e0 (if normalized), sample_name, n_scans
    """
    energy, mu_avg, used_files = average_sample_dir(
        dir_path, scan_indices=scan_indices, use_absev=use_absev
    )

    result = {
        "energy": energy,
        "mu_avg": mu_avg,
        "n_scans": len(used_files),
        "used_files": used_files,
    }

    if normalize_flag:
        norm = normalize_xanes(energy, mu_avg)
        result["mu_norm"] = norm["norm"]
        result["mu_flat"] = norm["flat"]
        result["e0"] = norm["e0"]
        result["edge_step"] = norm["edge_step"]

    return result


# ---------------------------------------------------------------------------
# Macro Generation for SPEC
# ---------------------------------------------------------------------------

def generate_energy_grid(element: str, edge: str = "K",
                         pre_edge_start: float = -50,
                         xanes_step: float = 0.3,
                         exafs_end: float = 500) -> str:
    """Generate a gscan energy grid string for a given element and edge.

    Creates a multi-region grid with fine steps near the edge and coarser
    steps in the pre-edge and post-edge regions.

    Parameters
    ----------
    element : str
        Element symbol (e.g. 'Ru', 'Fe', 'Pt').
    edge : str
        Edge name (e.g. 'K', 'L3', 'L2'). Default 'K'.
    pre_edge_start : float
        Start of pre-edge region relative to E0 (eV). Default -50.
    xanes_step : float
        Step size in the XANES region (eV). Default 0.3.
    exafs_end : float
        End of scan relative to E0 (eV). Default 500.

    Returns
    -------
    str : gscan energy grid string (e.g. "22070 22100 2 22110 1 ...")
    """
    # Look up edge energy
    if element not in XAS_EDGE_DB:
        raise ValueError(f"Element '{element}' not in database. "
                         f"Available: {sorted(XAS_EDGE_DB.keys())}")
    edges = XAS_EDGE_DB[element]
    if edge not in edges:
        raise ValueError(f"Edge '{edge}' not available for {element}. "
                         f"Available: {list(edges.keys())}")
    e0 = edges[edge]

    # Build energy grid regions
    e_start = int(e0 + pre_edge_start)
    regions = []

    # Start energy
    regions.append(f"{e_start}")

    # Pre-edge: coarse (2 eV steps)
    pre_end = int(e0 - 20)
    regions.append(f"{pre_end} 2")

    # Approaching edge: medium (1 eV steps)
    approach_end = int(e0 - 7)
    regions.append(f"{approach_end} 1")

    # Edge region: fine (xanes_step eV)
    edge_end = round(e0 + 10, 1)
    regions.append(f"{edge_end} {xanes_step}")

    # Near-edge / white line: medium-fine (0.5 eV)
    near_end = round(e0 + 30, 1)
    regions.append(f"{near_end} 0.5")

    # Post-edge XANES: medium (1 eV)
    post_xanes_end = round(e0 + 80, 1)
    regions.append(f"{post_xanes_end} 1")

    # Extended post-edge: coarse (4 eV)
    ext_end = round(e0 + 200, 1)
    regions.append(f"{ext_end} 4")

    # Far post-edge: very coarse (10 eV)
    far_end = round(e0 + exafs_end, 1)
    regions.append(f"{far_end} 10")

    return " ".join(regions)


def generate_xas_macro(element: str, edge: str = "K",
                       macro_name: str = None,
                       count_time: float = 1.0,
                       xanes_step: float = 0.3,
                       pre_edge_start: float = -50,
                       exafs_end: float = 500,
                       include_exafs: bool = False) -> str:
    """Generate a SPEC macro for XAS data collection.

    Parameters
    ----------
    element : str
        Element symbol.
    edge : str
        Edge name. Default 'K'.
    macro_name : str or None
        Name for the macro function. Default: '{element}_xas'.
    count_time : float
        Default counting time per point (seconds).
    xanes_step : float
        Step size in XANES region (eV).
    pre_edge_start : float
        Pre-edge start relative to E0 (eV).
    exafs_end : float
        End of scan relative to E0 (eV).
    include_exafs : bool
        If True, also generate an EXAFS macro.

    Returns
    -------
    str : Complete SPEC macro text.
    """
    if macro_name is None:
        macro_name = f"{element}_xas"

    e0 = XAS_EDGE_DB[element][edge]
    grid = generate_energy_grid(element, edge, pre_edge_start, xanes_step, exafs_end)

    SQ = "'"  # single quote for SPEC macro syntax

    lines = []
    lines.append(f"# {macro_name} cntSec  nbrScan  emission  nbrFilter")
    lines.append(f"# Element: {element}, Edge: {edge}, E0 = {e0:.1f} eV")
    lines.append(f"def {macro_name} {SQ}{{")
    lines.append(f"    local scan_ctime scan_repeat")
    lines.append(f"    local ndx nbrFilter")
    lines.append(f"    global XAS_MAIN_GRID")
    lines.append(f"    local  arr_tmp xas_start")
    lines.append(f"")
    lines.append(f'    XAS_MAIN_GRID  = "{grid}"')
    lines.append(f"")
    lines.append(f'    split(XAS_MAIN_GRID, arr_tmp)')
    lines.append(f'    xas_start = arr_tmp["0"]')
    lines.append(f"")
    lines.append(f"    if ($# < 4) {{")
    lines.append(f'        printf("\\n\\nSyntax:\\n")')
    lines.append(f'        printf("    {macro_name}  cntSec  nbrScan  emission  nbrFilter\\n\\n")')
    lines.append(f"        return")
    lines.append(f"    }}")
    lines.append(f"    if ( $1 <= 0) {{")
    lines.append(f'        print "Counting time needs to be non-zero positive."')
    lines.append(f'        print "Default {count_time} sec is used now."')
    lines.append(f"        scan_ctime = {count_time}")
    lines.append(f"    }} else {{")
    lines.append(f"        scan_ctime = $1")
    lines.append(f"    }}")
    lines.append(f"    if ( $2 == 0) {{")
    lines.append(f"        scan_repeat = 1")
    lines.append(f"    }} else {{")
    lines.append(f"        scan_repeat = $2")
    lines.append(f"    }}")
    lines.append(f"    if ( $3 == 0 ) {{")
    lines.append(f'        printf("Emission energy stays at %f eV\\n", A[emiss])')
    lines.append(f"    }} else {{")
    lines.append(f'        printf("Moving emission energy to %f eV\\n", $3)')
    lines.append(f'        eval(sprintf("mv emiss %f",$3))')
    lines.append(f"    }}")
    lines.append(f"    if ( $4 >= 0 ) {{")
    lines.append(f'        printf("FILTER=%d will be applied for the scan.\\n",$4)')
    lines.append(f"        nbrFilter = $4")
    lines.append(f"    }} else {{")
    lines.append(f'        printf("\\n!!!!!! Error !!!!!!\\n")')
    lines.append(f'        printf("Number of filter needs to be positive.\\n")')
    lines.append(f"        return")
    lines.append(f"    }}")
    lines.append(f"    for (ndx=0; ndx< scan_repeat; ndx++) {{")
    lines.append(f'        eval(sprintf("umv energy %f",xas_start))')
    lines.append(f'        eval(sprintf("mv filter %d",nbrFilter))')
    lines.append(f"        sleep(0.25)")
    lines.append(f'        eval(sprintf("gscan energy %s %f", XAS_MAIN_GRID, scan_ctime))')
    lines.append(f"    }}")
    lines.append(f"}}{SQ}")

    if include_exafs:
        e_start = int(e0 - 70)
        lines.append(f"")
        lines.append(f"")
        lines.append(f"def {element}_exafs {SQ}{{")
        lines.append(f"# EXAFS scan for {element} {edge}-edge (E0 = {e0:.1f} eV)")
        lines.append(f"# Usage: kscan_energy start0 step1 sec1 end1 [...] E0 k1 kstep k2 ksec1 ksec2 kweight")
        lines.append(f"        umv energy {e_start}")
        lines.append(f"        kscan_energy {e_start} 2 1 {int(e0-20)} 1.0 1 {int(e0-7)} 0.3 1 {int(e0+10)} 0.5 1 {int(e0+20)} {e0:.0f} 1.0 0.05 14 1 15 2")
        lines.append(f"}}{SQ}")

    return "\n".join(lines)


def generate_run_macro(samples: list, xas_macro: str = "Ru_xas") -> str:
    """Generate a batch run macro for multiple samples.

    Parameters
    ----------
    samples : list of dict
        Each dict should have:
            name : str - sample/file name
            Sx, Sy, Sz, Sr : float - motor positions
            emiss : float - emission energy (eV)
            count_time : float - counting time per point (sec)
            n_scans : int - number of repeat scans
            filter : int - number of filters (default 0)
            geometry : str - 'grazing' or 'normal' (optional)
    xas_macro : str
        Name of the XAS macro to call.

    Returns
    -------
    str : Complete run macro text.
    """
    lines = [f"# Batch run macro generated by HERFD Agent",
             f"# XAS macro: {xas_macro}",
             f"# Samples: {len(samples)}",
             ""]

    for i, s in enumerate(samples):
        name = s.get("name", f"sample_{i+1}")
        geometry = s.get("geometry", "")
        lines.append("######################################")
        lines.append(f"# Sample {i+1}: {name}" + (f" ({geometry})" if geometry else ""))
        lines.append(f"newfile {name}")
        lines.append("")

        # Motor positions
        sx = s.get("Sx", 0)
        sy = s.get("Sy", 0)
        sz = s.get("Sz", 0)
        sr = s.get("Sr", 0)
        lines.append(f"umv Sx {sx} Sy {sy} Sz {sz} Sr {sr}")

        # Emission energy
        emiss = s.get("emiss", 0)
        if emiss:
            lines.append(f"umv emiss {emiss}")

        lines.append("")

        # XAS scan
        ct = s.get("count_time", 1)
        ns = s.get("n_scans", 4)
        filt = s.get("filter", 0)
        emiss_arg = s.get("emission_arg", 0)
        lines.append(f"#{xas_macro}  cntSec  nbrScan  emission  nbrFilter")
        lines.append(f"{xas_macro} {ct} {ns} {emiss_arg} {filt}")
        lines.append("")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main entry point for command-line usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HERFD XAS Data Processing Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all sample directories and scan summaries
  python herfd_utils.py --summary

  # Process all samples (average + normalize + export)
  python herfd_utils.py --process-all

  # Process a specific sample directory
  python herfd_utils.py --process 2026-05_Yano/20260519_RuO2-TiO2_100_normal_dir

  # Average specific scans from a directory
  python herfd_utils.py --average 2026-05_Yano/20260519_RuO2-TiO2_100_normal_dir --scans 6 7 8 9 10
        """,
    )
    parser.add_argument("--summary", action="store_true",
                        help="Print scan summaries for all sample directories")
    parser.add_argument("--process-all", action="store_true",
                        help="Process all samples: average + normalize + export")
    parser.add_argument("--process", type=str, default=None,
                        help="Process a specific sample directory")
    parser.add_argument("--average", type=str, default=None,
                        help="Average scans in a specific directory")
    parser.add_argument("--scans", nargs="+", type=int, default=None,
                        help="Specific scan numbers to include")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help=f"Root data directory (default: {DATA_DIR})")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip normalization")
    parser.add_argument("--e0", type=float, default=None,
                        help="Edge energy for normalization (eV)")
    parser.add_argument("--use-mono-energy", action="store_true",
                        help="Use mono setpoint energy instead of calibrated absev")

    args = parser.parse_args()
    use_absev = not args.use_mono_energy

    if args.summary:
        print_all_summaries(args.data_dir)

    elif args.process_all:
        results = process_all_samples(
            data_dir=args.data_dir,
            use_absev=use_absev,
            normalize=not args.no_normalize,
            e0=args.e0,
        )
        print(f"\nProcessed {len(results)} samples.")

    elif args.process:
        dir_path = args.process
        sample_info = parse_sample_name(os.path.basename(dir_path))
        energy, mu_avg, used_files = average_sample_dir(
            dir_path, scan_indices=args.scans, use_absev=use_absev
        )
        print(f"Averaged {len(used_files)} scans from {dir_path}")

        if not args.no_normalize:
            norm = normalize_xanes(energy, mu_avg, e0=args.e0)
            print(f"E0 = {norm['e0']:.2f} eV, edge_step = {norm['edge_step']:.6g}")
            export_normalized(
                energy, mu_avg, norm,
                f"{sample_info['full_name']}_norm.dat",
            )
            print(f"Exported to {EXPORT_DIR}/normalized/")
        else:
            export_spectrum(
                energy, mu_avg,
                f"{sample_info['full_name']}_avg.dat",
                subdir="averaged",
            )
            print(f"Exported to {EXPORT_DIR}/averaged/")

    elif args.average:
        dir_path = args.average
        energy, mu_avg, used_files = average_sample_dir(
            dir_path, scan_indices=args.scans, use_absev=use_absev
        )
        print(f"Averaged {len(used_files)} scans")
        sample_info = parse_sample_name(os.path.basename(dir_path))
        export_spectrum(
            energy, mu_avg,
            f"{sample_info['full_name']}_avg.dat",
            subdir="averaged",
        )
        print(f"Exported to {EXPORT_DIR}/averaged/")

    else:
        parser.print_help()
