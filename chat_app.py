"""
HERFD Agent – Interactive Chat Web App
=======================================
A Flask-based chat interface for the HERFD XAS AI Agent.
Run with:  python chat_app.py
Then open http://localhost:5051 in your browser.
"""

import os
import sys
import re
import json
import textwrap
import base64
import io
import datetime
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# ── Load environment ─────────────────────────────────────────────────────────
load_dotenv()

# ── Ensure local imports work ─────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import herfd_utils as hu

# ── LLM Provider Configuration ──────────────────────────────────────────────
PROVIDER_DEFAULTS = {
    "cborg": {
        "key_env": "CBORG_API_KEY",
        "base_url": "https://api.cborg.lbl.gov/v1",
        "model": "claude-sonnet",
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
    },
    "claude": {
        "key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-20250514",
    },
}


def _configure_llm():
    """Detect and configure the LLM provider from environment variables."""
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if provider and provider in PROVIDER_DEFAULTS:
        cfg = PROVIDER_DEFAULTS[provider]
        api_key = os.environ.get(cfg["key_env"], "")
        if not api_key:
            raise ValueError(
                f"LLM_PROVIDER={provider} but {cfg['key_env']} is not set in .env"
            )
    elif provider:
        api_key = os.environ.get("LLM_API_KEY", "")
        base_url = os.environ.get("LLM_BASE_URL", "")
        model = os.environ.get("LLM_MODEL", "")
        if not all([api_key, base_url, model]):
            raise ValueError(
                f"Custom LLM_PROVIDER='{provider}' requires LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL in .env"
            )
        return OpenAI(api_key=api_key, base_url=base_url), model, provider
    else:
        for pname, cfg in PROVIDER_DEFAULTS.items():
            api_key = os.environ.get(cfg["key_env"], "")
            if api_key and api_key != "your-api-key-here":
                provider = pname
                break
        else:
            raise ValueError(
                "No LLM API key found. Set one of: CBORG_API_KEY, OPENAI_API_KEY, "
                "GEMINI_API_KEY, or ANTHROPIC_API_KEY in your .env file."
            )
        cfg = PROVIDER_DEFAULTS[provider]

    base_url = os.environ.get("LLM_BASE_URL", cfg["base_url"])
    model = os.environ.get("LLM_MODEL", cfg["model"])

    return OpenAI(api_key=api_key, base_url=base_url), model, provider


client, MODEL, LLM_PROVIDER = _configure_llm()
print(f"[OK] LLM Provider: {LLM_PROVIDER} | Model: {MODEL}")

DATA_DIR = os.environ.get("HERFD_DATA_DIR", "2026-05_Yano")
EXPORT_DIR = os.environ.get("HERFD_EXPORT_DIR", "exported_data")

# ── Matplotlib defaults ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ── State ─────────────────────────────────────────────────────────────────────
_last_plot = {}
_last_plot_b64 = ""
_cache = {}
_pending_images = []


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _apply_axis_style(ax, axis_style: dict = None):
    """Apply axis styling (fonts, colors, sizes) to a matplotlib axes."""
    if not axis_style:
        return
    s = axis_style
    font_kw = {}
    if "font_family" in s and s["font_family"]:
        font_kw["fontfamily"] = s["font_family"]

    if "title_size" in s or "title_color" in s or font_kw:
        ax.title.set_fontsize(s.get("title_size", ax.title.get_fontsize()))
        if "title_color" in s and s["title_color"]:
            ax.title.set_color(s["title_color"])
        if font_kw:
            ax.title.set_fontfamily(font_kw.get("fontfamily"))

    label_size = s.get("label_size")
    label_color = s.get("label_color")
    if label_size:
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
    if label_color:
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)

    tick_size = s.get("tick_size")
    tick_color = s.get("tick_color")
    if tick_size:
        ax.xaxis.set_tick_params(labelsize=tick_size)
        ax.yaxis.set_tick_params(labelsize=tick_size)
    if tick_color:
        ax.xaxis.set_tick_params(labelcolor=tick_color)
        ax.yaxis.set_tick_params(labelcolor=tick_color)

    legend_size = s.get("legend_size")
    if legend_size:
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(legend_size)


# ── Tool definitions ─────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_samples",
            "description": "List all available sample directories with their scan counts. Each sample directory contains multiple XAS scans that can be averaged together.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_sample_info",
            "description": "Show detailed information about a sample directory: list all scans with their types (XAS, alignment, emission, etc.), scan numbers, and point counts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample": {"type": "string", "description": "Sample directory name (e.g. '20260519_RuO2-TiO2_100_normal_dir') or a partial match (e.g. '100_normal', 'RuO2_Ref')."},
                },
                "required": ["sample"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_sample",
            "description": "Average all XAS scans in a sample directory and plot the averaged HERFD spectrum (vortDT/I0 vs energy). Optionally select specific scan numbers to include.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample": {"type": "string", "description": "Sample directory name or partial match."},
                    "scans": {"type": "array", "items": {"type": "integer"}, "description": "Optional list of specific scan numbers to include. If not given, all XAS scans are averaged."},
                    "e_min": {"type": "number", "description": "Minimum energy in eV for the plot range (zoom)."},
                    "e_max": {"type": "number", "description": "Maximum energy in eV for the plot range (zoom)."},
                    "y_min": {"type": "number", "description": "Minimum Y-axis value for the plot range."},
                    "y_max": {"type": "number", "description": "Maximum Y-axis value for the plot range."},
                    "color": {"type": "string", "description": "Line color. Default: blue."},
                    "title": {"type": "string", "description": "Custom plot title."},
                    "label": {"type": "string", "description": "Custom legend label."},
                    "axis_style": {
                        "type": "object",
                        "description": "Customize axis appearance.",
                        "properties": {
                            "font_family": {"type": "string"},
                            "title_size": {"type": "number"},
                            "title_color": {"type": "string"},
                            "label_size": {"type": "number"},
                            "label_color": {"type": "string"},
                            "tick_size": {"type": "number"},
                            "tick_color": {"type": "string"},
                            "legend_size": {"type": "number"},
                        },
                    },
                },
                "required": ["sample"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_samples",
            "description": "Overlay averaged HERFD spectra from multiple samples on one plot for comparison. Supports auto-scaling, offsets, and per-curve styling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "samples": {"type": "array", "items": {"type": "string"}, "description": "List of sample directory names or partial matches."},
                    "e_min": {"type": "number", "description": "Minimum energy in eV for the plot range."},
                    "e_max": {"type": "number", "description": "Maximum energy in eV for the plot range."},
                    "y_min": {"type": "number", "description": "Minimum Y-axis value."},
                    "y_max": {"type": "number", "description": "Maximum Y-axis value."},
                    "offset": {"type": "number", "description": "Vertical offset between curves. Default 0."},
                    "auto_scale": {"type": "string", "enum": ["overlay", "offset"], "description": "Auto-scale mode. 'overlay': normalize to [0,1] and overlay. 'offset': normalize and stack."},
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "Custom legend labels, one per sample."},
                    "styles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "color": {"type": "string"},
                                "linestyle": {"type": "string"},
                                "linewidth": {"type": "number"},
                            },
                        },
                        "description": "Per-curve style overrides.",
                    },
                    "title": {"type": "string", "description": "Custom plot title."},
                    "axis_style": {
                        "type": "object",
                        "description": "Customize axis appearance.",
                        "properties": {
                            "font_family": {"type": "string"},
                            "title_size": {"type": "number"},
                            "label_size": {"type": "number"},
                            "tick_size": {"type": "number"},
                            "legend_size": {"type": "number"},
                        },
                    },
                },
                "required": ["samples"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "normalize_sample",
            "description": "Perform Athena-style XANES normalization on a sample's averaged spectrum: pre-edge subtraction, post-edge normalization, edge step = 1. Plots the normalized spectrum and optionally saves it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample": {"type": "string", "description": "Sample directory name or partial match."},
                    "e0": {"type": "number", "description": "Optional edge energy in eV. If not given, determined automatically."},
                    "flatten": {"type": "boolean", "description": "If true (default), use flattened spectrum with corrected post-edge slope. Set false for raw normalized."},
                    "save": {"type": "boolean", "description": "If true, save the normalized data. Default false."},
                    "scans": {"type": "array", "items": {"type": "integer"}, "description": "Optional specific scan numbers to include."},
                },
                "required": ["sample"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_normalized",
            "description": "Compare normalized XANES spectra from multiple samples on one plot. Each sample is averaged and normalized before plotting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "samples": {"type": "array", "items": {"type": "string"}, "description": "List of sample names or partial matches."},
                    "e_min": {"type": "number", "description": "Minimum energy in eV."},
                    "e_max": {"type": "number", "description": "Maximum energy in eV."},
                    "y_min": {"type": "number", "description": "Minimum Y-axis value."},
                    "y_max": {"type": "number", "description": "Maximum Y-axis value."},
                    "flatten": {"type": "boolean", "description": "If true (default), use flattened spectra with corrected post-edge slope. Set false for raw normalized."},
                    "offset": {"type": "number", "description": "Vertical offset between curves. Default 0."},
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "Custom legend labels."},
                    "title": {"type": "string", "description": "Custom plot title."},
                    "styles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "color": {"type": "string"},
                                "linestyle": {"type": "string"},
                                "linewidth": {"type": "number"},
                            },
                        },
                    },
                    "axis_style": {
                        "type": "object",
                        "properties": {
                            "font_family": {"type": "string"},
                            "title_size": {"type": "number"},
                            "label_size": {"type": "number"},
                            "tick_size": {"type": "number"},
                            "legend_size": {"type": "number"},
                        },
                    },
                },
                "required": ["samples"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "derivative_sample",
            "description": "Compute and plot the 1st or 2nd derivative of a sample's averaged HERFD spectrum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample": {"type": "string", "description": "Sample name or partial match."},
                    "order": {"type": "integer", "enum": [1, 2], "description": "Derivative order: 1 or 2."},
                    "smooth_window": {"type": "integer", "description": "Optional Savitzky-Golay window size (odd integer)."},
                },
                "required": ["sample", "order"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_single_scan",
            "description": "Plot a single individual scan (not averaged) from a sample directory. Useful for inspecting individual scans for quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample": {"type": "string", "description": "Sample name or partial match."},
                    "scan_number": {"type": "integer", "description": "Scan number to plot."},
                    "color": {"type": "string", "description": "Line color. Default: blue."},
                    "title": {"type": "string", "description": "Custom plot title."},
                },
                "required": ["sample", "scan_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_all_scans",
            "description": "Overlay all individual XAS scans from a sample directory on one plot. Useful for checking scan-to-scan reproducibility and identifying outliers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample": {"type": "string", "description": "Sample name or partial match."},
                    "e_min": {"type": "number", "description": "Minimum energy in eV."},
                    "e_max": {"type": "number", "description": "Maximum energy in eV."},
                    "y_min": {"type": "number", "description": "Minimum Y-axis value."},
                    "y_max": {"type": "number", "description": "Maximum Y-axis value."},
                    "title": {"type": "string", "description": "Custom plot title."},
                },
                "required": ["sample"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "identify_element",
            "description": "Identify the element and absorption edge being measured from the energy range and sample metadata. Uses the XAS edge database to find matching elements. Call this when first examining data or when the element is unclear.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample": {"type": "string", "description": "Sample name or partial match. If not given, uses the first available sample."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_data",
            "description": "Save the last plotted data to a text file in exported_data/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Optional filename."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_image",
            "description": "Save the last plot as a PNG image file in exported_data/images/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Optional filename (e.g. 'my_plot.png')."},
                    "dpi": {"type": "integer", "description": "Image resolution. Default: 150."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_all",
            "description": "Process all sample directories: average XAS scans, normalize, and export results. Returns a summary of all processed samples with E0 values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "normalize": {"type": "boolean", "description": "Apply XANES normalization. Default true."},
                    "export": {"type": "boolean", "description": "Export results to files. Default true."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_file",
            "description": "Plot a generic data file (txt, csv, dat) from exported_data/ or any path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file."},
                    "e_min": {"type": "number", "description": "Minimum X value."},
                    "e_max": {"type": "number", "description": "Maximum X value."},
                    "y_min": {"type": "number", "description": "Minimum Y-axis value."},
                    "y_max": {"type": "number", "description": "Maximum Y-axis value."},
                    "title": {"type": "string", "description": "Custom plot title."},
                    "color": {"type": "string", "description": "Line color. Default: blue."},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_exports",
            "description": "List all files in the exported_data/ directory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ── Tool implementations ─────────────────────────────────────────────────────

def _resolve_sample(sample_name: str) -> str:
    """Resolve a partial sample name to a full directory path."""
    sample_dirs = hu.list_sample_dirs(DATA_DIR)

    # Exact match
    if sample_name in sample_dirs:
        return os.path.join(DATA_DIR, sample_name)

    # Try with _dir suffix
    if sample_name + "_dir" in sample_dirs:
        return os.path.join(DATA_DIR, sample_name + "_dir")

    # Partial match
    matches = [d for d in sample_dirs if sample_name.lower() in d.lower()]
    if len(matches) == 1:
        return os.path.join(DATA_DIR, matches[0])
    elif len(matches) > 1:
        raise ValueError(
            f"Ambiguous sample name '{sample_name}'. Matches: {', '.join(matches)}"
        )
    else:
        raise ValueError(
            f"Sample '{sample_name}' not found. Available: {', '.join(sample_dirs)}"
        )


def _get_sample_label(dir_path: str) -> str:
    """Get a short label for a sample from its directory name."""
    info = hu.parse_sample_name(os.path.basename(dir_path))
    parts = []
    if info["orientation"]:
        parts.append(info["orientation"])
    if info["temperature"]:
        parts.append(info["temperature"])
    if info["geometry"]:
        parts.append(info["geometry"])
    if not parts:
        parts.append(info["full_name"])
    return " ".join(parts)


def _get_style(styles: list, index: int, defaults: dict) -> dict:
    """Get style for curve at given index, merging with defaults."""
    style = dict(defaults)
    if styles and index < len(styles) and styles[index]:
        s = styles[index]
        if "color" in s and s["color"]:
            style["color"] = s["color"]
        if "linestyle" in s and s["linestyle"]:
            style["linestyle"] = s["linestyle"]
        if "linewidth" in s and s["linewidth"]:
            style["linewidth"] = float(s["linewidth"])
    return style


def tool_list_samples(**kw) -> str:
    sample_dirs = hu.list_sample_dirs(DATA_DIR)
    if not sample_dirs:
        return f"No sample directories found in {DATA_DIR}."

    lines = [f"Found {len(sample_dirs)} sample directories:\n"]
    for d in sample_dirs:
        dir_path = os.path.join(DATA_DIR, d)
        xas_scans = hu.get_xas_scans(dir_path)
        info = hu.parse_sample_name(d)
        label = f"{info['sample']} {info['orientation']}"
        if info["temperature"]:
            label += f" {info['temperature']}"
        label += f" ({info['geometry']})"
        lines.append(f"  {d}  —  {len(xas_scans)} XAS scans  [{label}]")

    return "\n".join(lines)


def tool_show_sample_info(sample: str, **kw) -> str:
    try:
        dir_path = _resolve_sample(sample)
    except ValueError as e:
        return str(e)

    return hu.summarize_directory(dir_path)


def tool_plot_sample(sample: str, scans: list = None, e_min: float = None,
                     e_max: float = None, y_min: float = None, y_max: float = None,
                     color: str = "blue",
                     title: str = None, label: str = None,
                     axis_style: dict = None, **kw) -> str:
    global _last_plot, _last_plot_b64
    try:
        dir_path = _resolve_sample(sample)
    except ValueError as e:
        return str(e)

    try:
        energy, mu_avg, used_files = hu.average_sample_dir(
            dir_path, scan_indices=scans
        )
    except ValueError as e:
        return str(e)

    # Apply energy range
    mask = np.ones(len(energy), dtype=bool)
    if e_min is not None:
        mask &= energy >= e_min
    if e_max is not None:
        mask &= energy <= e_max
    ep, mp = energy[mask], mu_avg[mask]

    if len(ep) == 0:
        return f"No data in range {e_min}–{e_max} eV."

    sample_label = label or _get_sample_label(dir_path)

    fig, ax = plt.subplots()
    ax.plot(ep, mp, color=color, linewidth=1.2, label=sample_label)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("HERFD (vortDT / I0)")
    ax.set_title(title or f"HERFD — {sample_label} (avg of {len(used_files)} scans)")
    ax.legend(fontsize=9)
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    _apply_axis_style(ax, axis_style)
    plt.tight_layout()

    img_b64 = _fig_to_base64(fig)
    _pending_images.append(img_b64)
    _last_plot_b64 = img_b64
    _last_plot = {"energy": ep, "signal": mp, "signal_name": "HERFD",
                  "sample": os.path.basename(dir_path)}

    return (f"Plotted averaged HERFD for {sample_label} "
            f"({len(used_files)} scans, {len(ep)} pts, "
            f"{ep.min():.1f}–{ep.max():.1f} eV).")


def tool_compare_samples(samples: list, e_min: float = None, e_max: float = None,
                         y_min: float = None, y_max: float = None,
                         offset: float = 0, auto_scale: str = None,
                         labels: list = None, styles: list = None,
                         title: str = None, axis_style: dict = None, **kw) -> str:
    global _last_plot, _last_plot_b64

    if auto_scale is True:
        auto_scale = "overlay"
    elif auto_scale is False or auto_scale is None:
        auto_scale = None

    colors = list(plt.cm.tab10.colors)
    fig, ax = plt.subplots()
    plot_info = []

    for i, sample_name in enumerate(samples):
        try:
            dir_path = _resolve_sample(sample_name)
        except ValueError as e:
            plot_info.append(f"  ✗ {sample_name}: {e}")
            continue

        try:
            energy, mu_avg, used_files = hu.average_sample_dir(dir_path)
        except ValueError as e:
            plot_info.append(f"  ✗ {sample_name}: {e}")
            continue

        mask = np.ones(len(energy), dtype=bool)
        if e_min is not None:
            mask &= energy >= e_min
        if e_max is not None:
            mask &= energy <= e_max
        ep, mp = energy[mask], mu_avg[mask]

        if len(ep) == 0:
            continue

        # Auto-scale
        if auto_scale == "overlay":
            dmin, dmax = mp.min(), mp.max()
            mp = (mp - dmin) / (dmax - dmin + 1e-30)
            v_offset = 0
        elif auto_scale == "offset":
            dmin, dmax = mp.min(), mp.max()
            mp = (mp - dmin) / (dmax - dmin + 1e-30)
            v_offset = i * 1.1
        else:
            v_offset = offset * i

        lbl = labels[i] if labels and i < len(labels) else _get_sample_label(dir_path)
        sty = _get_style(styles, i, {
            "color": colors[i % len(colors)],
            "linestyle": "-", "linewidth": 1.2
        })

        ax.plot(ep, mp + v_offset, label=lbl, **sty)
        plot_info.append(f"  ✓ {lbl}: {len(used_files)} scans")

    ax.set_xlabel("Energy (eV)")
    ylabel = "Normalized Intensity" if auto_scale else "HERFD (vortDT / I0)"
    ax.set_ylabel(ylabel)
    ax.set_title(title or "HERFD Comparison")
    ax.legend(fontsize=9)
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    _apply_axis_style(ax, axis_style)
    plt.tight_layout()

    img_b64 = _fig_to_base64(fig)
    _pending_images.append(img_b64)
    _last_plot_b64 = img_b64

    return "Comparison plot:\n" + "\n".join(plot_info)


def tool_normalize_sample(sample: str, e0: float = None, flatten: bool = True,
                          save: bool = False, scans: list = None, **kw) -> str:
    global _last_plot, _last_plot_b64
    try:
        dir_path = _resolve_sample(sample)
    except ValueError as e:
        return str(e)

    try:
        energy, mu_avg, used_files = hu.average_sample_dir(
            dir_path, scan_indices=scans
        )
    except ValueError as e:
        return str(e)

    norm_result = hu.normalize_xanes(energy, mu_avg, e0=e0)
    mu_plot = norm_result["flat"] if flatten else norm_result["norm"]

    sample_label = _get_sample_label(dir_path)

    fig, ax = plt.subplots()
    ax.plot(energy, mu_plot, color="blue", linewidth=1.2, label=sample_label)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=norm_result["e0"], color="red", linestyle=":", alpha=0.5,
               label=f"E0 = {norm_result['e0']:.1f} eV")
    ax.set_xlabel("Energy (eV)")
    ylabel = "Flattened μ(E)" if flatten else "Normalized μ(E)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Normalized HERFD — {sample_label}")
    ax.legend(fontsize=9)
    plt.tight_layout()

    img_b64 = _fig_to_base64(fig)
    _pending_images.append(img_b64)
    _last_plot_b64 = img_b64
    _last_plot = {"energy": energy, "signal": mu_plot,
                  "signal_name": ylabel, "sample": os.path.basename(dir_path)}

    result = (f"Normalized {sample_label}: E0 = {norm_result['e0']:.2f} eV, "
              f"edge_step = {norm_result['edge_step']:.6g}")

    if save:
        info = hu.parse_sample_name(os.path.basename(dir_path))
        path = hu.export_normalized(
            energy, mu_avg, norm_result,
            f"{info['full_name']}_norm.dat",
            subdir="normalized",
        )
        result += f"\nSaved to {path}"

    return result


def tool_compare_normalized(samples: list, e_min: float = None, e_max: float = None,
                            y_min: float = None, y_max: float = None,
                            flatten: bool = True, offset: float = 0,
                            labels: list = None, title: str = None,
                            styles: list = None, axis_style: dict = None, **kw) -> str:
    global _last_plot, _last_plot_b64

    colors = list(plt.cm.tab10.colors)
    fig, ax = plt.subplots()
    plot_info = []

    for i, sample_name in enumerate(samples):
        try:
            dir_path = _resolve_sample(sample_name)
        except ValueError as e:
            plot_info.append(f"  ✗ {sample_name}: {e}")
            continue

        try:
            energy, mu_avg, used_files = hu.average_sample_dir(dir_path)
        except ValueError as e:
            plot_info.append(f"  ✗ {sample_name}: {e}")
            continue

        norm = hu.normalize_xanes(energy, mu_avg)
        mu_plot = norm["flat"] if flatten else norm["norm"]

        mask = np.ones(len(energy), dtype=bool)
        if e_min is not None:
            mask &= energy >= e_min
        if e_max is not None:
            mask &= energy <= e_max
        ep, mp = energy[mask], mu_plot[mask]

        if len(ep) == 0:
            continue

        v_offset = offset * i
        lbl = labels[i] if labels and i < len(labels) else _get_sample_label(dir_path)
        sty = _get_style(styles, i, {
            "color": colors[i % len(colors)],
            "linestyle": "-", "linewidth": 1.2
        })

        ax.plot(ep, mp + v_offset, label=lbl, **sty)
        plot_info.append(f"  ✓ {lbl}: E0={norm['e0']:.1f} eV")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Energy (eV)")
    ylabel = "Flattened μ(E)" if flatten else "Normalized μ(E)"
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Normalized HERFD Comparison")
    ax.legend(fontsize=9)
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    _apply_axis_style(ax, axis_style)
    plt.tight_layout()

    img_b64 = _fig_to_base64(fig)
    _pending_images.append(img_b64)
    _last_plot_b64 = img_b64

    return "Normalized comparison:\n" + "\n".join(plot_info)


def tool_derivative_sample(sample: str, order: int = 1,
                           smooth_window: int = None, **kw) -> str:
    global _last_plot, _last_plot_b64
    try:
        dir_path = _resolve_sample(sample)
    except ValueError as e:
        return str(e)

    try:
        energy, mu_avg, used_files = hu.average_sample_dir(dir_path)
    except ValueError as e:
        return str(e)

    deriv = hu.smooth_derivative(energy, mu_avg, order=order, window=smooth_window)
    sample_label = _get_sample_label(dir_path)

    fig, ax = plt.subplots()
    ax.plot(energy, deriv, color="blue", linewidth=1.2, label=sample_label)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(f"d{'²' if order == 2 else ''}μ/dE{'²' if order == 2 else ''}")
    ax.set_title(f"{'2nd' if order == 2 else '1st'} Derivative — {sample_label}")
    ax.legend(fontsize=9)
    plt.tight_layout()

    img_b64 = _fig_to_base64(fig)
    _pending_images.append(img_b64)
    _last_plot_b64 = img_b64
    _last_plot = {"energy": energy, "signal": deriv,
                  "signal_name": f"d{order}mu/dE{order}", "sample": os.path.basename(dir_path)}

    return f"Plotted {'2nd' if order == 2 else '1st'} derivative for {sample_label}."


def tool_plot_single_scan(sample: str, scan_number: int,
                          color: str = "blue", title: str = None, **kw) -> str:
    global _last_plot, _last_plot_b64
    try:
        dir_path = _resolve_sample(sample)
    except ValueError as e:
        return str(e)

    scans = hu.list_scans_in_dir(dir_path)
    target = [s for s in scans if s.get("scan_number") == scan_number]
    if not target:
        available = [s["scan_number"] for s in scans]
        return f"Scan #{scan_number} not found. Available: {available}"

    scan_info = target[0]
    df = hu.load_spec_scan(scan_info["filepath"])
    if df.empty:
        return f"Scan #{scan_number} has no data."

    energy, mu = hu.get_herfd_signal(df)

    fig, ax = plt.subplots()
    ax.plot(energy, mu, color=color, linewidth=1.2,
            label=f"Scan #{scan_number} ({scan_info['classification']})")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("HERFD (vortDT / I0)")
    ax.set_title(title or f"Scan #{scan_number} — {os.path.basename(dir_path)}")
    ax.legend(fontsize=9)
    plt.tight_layout()

    img_b64 = _fig_to_base64(fig)
    _pending_images.append(img_b64)
    _last_plot_b64 = img_b64
    _last_plot = {"energy": energy, "signal": mu, "signal_name": "HERFD",
                  "sample": os.path.basename(dir_path)}

    return (f"Plotted scan #{scan_number} ({scan_info['classification']}, "
            f"{scan_info.get('n_points', 0)} pts).")


def tool_plot_all_scans(sample: str, e_min: float = None, e_max: float = None,
                        y_min: float = None, y_max: float = None,
                        title: str = None, **kw) -> str:
    global _last_plot, _last_plot_b64
    try:
        dir_path = _resolve_sample(sample)
    except ValueError as e:
        return str(e)

    xas_scans = hu.get_xas_scans(dir_path)
    if not xas_scans:
        return f"No XAS scans found in {sample}."

    colors = plt.cm.viridis(np.linspace(0, 1, len(xas_scans)))
    fig, ax = plt.subplots()

    for i, scan_info in enumerate(xas_scans):
        try:
            df = hu.load_spec_scan(scan_info["filepath"])
            energy, mu = hu.get_herfd_signal(df)

            mask = np.ones(len(energy), dtype=bool)
            if e_min is not None:
                mask &= energy >= e_min
            if e_max is not None:
                mask &= energy <= e_max

            ax.plot(energy[mask], mu[mask], color=colors[i], alpha=0.6,
                    linewidth=0.8, label=f"#{scan_info['scan_number']}")
        except Exception:
            continue

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("HERFD (vortDT / I0)")
    sample_label = _get_sample_label(dir_path)
    ax.set_title(title or f"All XAS Scans — {sample_label}")
    if len(xas_scans) <= 15:
        ax.legend(fontsize=7, ncol=2)
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    plt.tight_layout()

    img_b64 = _fig_to_base64(fig)
    _pending_images.append(img_b64)
    _last_plot_b64 = img_b64

    return f"Plotted {len(xas_scans)} individual XAS scans for {sample_label}."


def tool_identify_element(sample: str = None, **kw) -> str:
    """Identify element and edge from energy range and metadata."""
    if sample:
        try:
            dir_path = _resolve_sample(sample)
        except ValueError as e:
            return str(e)
    else:
        # Use first available sample
        sample_dirs = hu.list_sample_dirs(DATA_DIR)
        if not sample_dirs:
            return "No sample directories found."
        dir_path = os.path.join(DATA_DIR, sample_dirs[0])

    # Get energy range from first XAS scan
    xas_scans = hu.get_xas_scans(dir_path)
    if not xas_scans:
        return f"No XAS scans found in {os.path.basename(dir_path)}."

    df = hu.load_spec_scan(xas_scans[0]["filepath"])
    energy, mu = hu.get_herfd_signal(df)
    e_range = (energy.min(), energy.max())

    # Get metadata text from directory name and scan command
    metadata = os.path.basename(dir_path) + " " + xas_scans[0].get("scan_command", "")

    candidates = hu.identify_element_from_energy(e_range, metadata)

    if not candidates:
        return (f"Could not identify element. Energy range: {e_range[0]:.1f}–{e_range[1]:.1f} eV. "
                f"No matching edges found in the database.")

    lines = [f"Energy range: {e_range[0]:.1f}–{e_range[1]:.1f} eV"]
    lines.append(f"Sample: {os.path.basename(dir_path)}\n")
    lines.append("Candidate elements:")
    for i, c in enumerate(candidates[:5]):
        marker = "⭐" if i == 0 else "  "
        lines.append(f"  {marker} {c['element']} {c['edge']}-edge "
                     f"({c['ref_energy']:.0f} eV) — matched from {c['source']}")

    if candidates[0]["source"] == "metadata":
        lines.append(f"\nBest match: **{candidates[0]['element']} {candidates[0]['edge']}-edge** "
                     f"(found in sample name)")
    else:
        lines.append(f"\nBest guess: **{candidates[0]['element']} {candidates[0]['edge']}-edge** "
                     f"(based on energy range — please confirm)")

    return "\n".join(lines)


def tool_save_data(filename: str = None, **kw) -> str:
    if not _last_plot:
        return "No data to save. Plot something first."

    if filename is None:
        sample = _last_plot.get("sample", "data")
        sig = _last_plot.get("signal_name", "signal")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sample}_{sig}_{ts}.dat"

    path = hu.export_spectrum(
        _last_plot["energy"], _last_plot["signal"],
        filename, subdir=None,
    )
    return f"Saved to {path}"


def tool_save_image(filename: str = None, dpi: int = 150, **kw) -> str:
    if not _last_plot_b64:
        return "No plot to save. Plot something first."

    out_dir = hu.ensure_export_dir("images")
    if filename is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{ts}.png"
    if not filename.endswith(".png"):
        filename += ".png"

    out_path = os.path.join(out_dir, filename)
    img_data = base64.b64decode(_last_plot_b64)
    with open(out_path, "wb") as f:
        f.write(img_data)

    return f"Saved plot to {out_path}"


def tool_process_all(normalize: bool = True, export: bool = True, **kw) -> str:
    results = hu.process_all_samples(
        data_dir=DATA_DIR, normalize=normalize, export=export
    )

    lines = [f"Processed {len(results)} samples:\n"]
    for r in results:
        line = f"  ✓ {r['sample_name']}: {r['n_scans']} scans"
        if "e0" in r:
            line += f", E0={r['e0']:.1f} eV"
        lines.append(line)

    if export:
        lines.append(f"\nExported to {EXPORT_DIR}/averaged/ and {EXPORT_DIR}/normalized/")

    return "\n".join(lines)


def tool_plot_file(filepath: str, e_min: float = None, e_max: float = None,
                   y_min: float = None, y_max: float = None,
                   title: str = None, color: str = "blue", **kw) -> str:
    global _last_plot, _last_plot_b64

    if not os.path.isfile(filepath):
        # Try in exported_data
        alt = os.path.join(EXPORT_DIR, filepath)
        if os.path.isfile(alt):
            filepath = alt
        else:
            return f"File not found: {filepath}"

    try:
        data = np.loadtxt(filepath, comments="#")
    except Exception as e:
        return f"Error loading {filepath}: {e}"

    if data.ndim != 2 or data.shape[1] < 2:
        return f"File must have at least 2 columns. Shape: {data.shape}"

    x, y = data[:, 0], data[:, 1]

    mask = np.ones(len(x), dtype=bool)
    if e_min is not None:
        mask &= x >= e_min
    if e_max is not None:
        mask &= x <= e_max

    fig, ax = plt.subplots()
    ax.plot(x[mask], y[mask], color=color, linewidth=1.2)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Signal")
    ax.set_title(title or os.path.basename(filepath))
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    plt.tight_layout()

    img_b64 = _fig_to_base64(fig)
    _pending_images.append(img_b64)
    _last_plot_b64 = img_b64
    _last_plot = {"energy": x[mask], "signal": y[mask],
                  "signal_name": "signal", "sample": os.path.basename(filepath)}

    return f"Plotted {os.path.basename(filepath)} ({len(x[mask])} pts)."


def tool_list_exports(**kw) -> str:
    if not os.path.isdir(EXPORT_DIR):
        return "No exported_data directory found."

    lines = ["Exported files:"]
    for root, dirs, files in os.walk(EXPORT_DIR):
        rel = os.path.relpath(root, EXPORT_DIR)
        level = rel.count(os.sep)
        indent = "  " * level
        if rel != ".":
            lines.append(f"{indent}📁 {os.path.basename(root)}/")
        for f in sorted(files):
            if f.startswith("."):
                continue
            lines.append(f"{indent}  📄 {f}")

    return "\n".join(lines) if len(lines) > 1 else "No exported files found."


# ── Tool dispatch ─────────────────────────────────────────────────────────────
TOOL_DISPATCH = {
    "list_samples": tool_list_samples,
    "show_sample_info": tool_show_sample_info,
    "plot_sample": tool_plot_sample,
    "compare_samples": tool_compare_samples,
    "normalize_sample": tool_normalize_sample,
    "compare_normalized": tool_compare_normalized,
    "derivative_sample": tool_derivative_sample,
    "plot_single_scan": tool_plot_single_scan,
    "plot_all_scans": tool_plot_all_scans,
    "identify_element": tool_identify_element,
    "save_data": tool_save_data,
    "save_image": tool_save_image,
    "process_all": tool_process_all,
    "plot_file": tool_plot_file,
    "list_exports": tool_list_exports,
}


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent(f"""\
You are a HERFD (High Energy Resolution Fluorescence Detected) XAS data analysis assistant.
You help the user visualize and analyze hard X-ray HERFD-XAS data collected at SSRL beamline 15-2.

Data directory: {DATA_DIR}
The data contains HERFD-XAS measurements in SPEC format.
Samples are organized in directories named like: 20260519_RuO2-TiO2_100_normal_dir
Each directory contains multiple SPEC-format .dat scan files.

Sample naming convention:
  - Typically: Date_Sample_Orientation_[Temperature]_Geometry
  - The agent auto-parses directory names to extract sample info
  - Use identify_element to determine the element and edge from the energy range and metadata

Scan types in each directory:
  - XAS scans (gscan energy): The main HERFD-XANES measurements. Multiple repeats are averaged.
  - Alignment scans (ascan Sz/Sx/Sy): Sample position optimization — NOT XAS data.
  - Emission scans (ascan emiss): Emission energy optimization — NOT XAS data.
  - Some scans may be aborted.

Signal: HERFD signal = vortDT / I0 (deadtime-corrected vortex detector counts normalized by ion chamber)
Energy: Calibrated energy from 'absev' column (monochromator setpoint is in 'energy' column)

Available tools:
- list_samples: List all sample directories with XAS scan counts
- show_sample_info: Show all scans in a sample directory with their types
- plot_sample: Average all XAS scans in a sample and plot the HERFD spectrum
- compare_samples: Overlay averaged HERFD spectra from multiple samples
- normalize_sample: Athena-style XANES normalization (pre-edge subtraction, post-edge normalization)
- compare_normalized: Compare normalized XANES spectra from multiple samples
- derivative_sample: Compute 1st or 2nd derivative of a sample's averaged spectrum
- plot_single_scan: Plot one individual scan (for quality checking)
- plot_all_scans: Overlay all individual XAS scans from a sample (for reproducibility checking)
- save_data: Save the last plotted data to a file
- save_image: Save the last plot as a PNG image
- process_all: Batch process all samples (average + normalize + export)
- plot_file: Plot a generic data file from exported_data/
- list_exports: List exported files

Rules:
- The user may refer to samples by full directory name or partial match (e.g. '100_normal', '110_grazing', 'RuO2_Ref', '500C')
- When the user says "plot", use plot_sample to show the averaged HERFD spectrum
- When the user says "compare", use compare_samples or compare_normalized
- When the user says "normalize", use normalize_sample. Flattening is on by default to correct the post-edge slope for hard X-ray data
- When the user says "derivative", use derivative_sample
- When the user asks to see individual scans or check quality, use plot_all_scans or plot_single_scan
- When the user says "process all" or "batch process", use process_all
- When the user asks to "zoom in" on energy, use e_min/e_max parameters
- When the user asks to set the Y-axis range or "zoom in" on intensity, use y_min/y_max parameters
- When the user asks to "offset" or "stack" curves, use the offset parameter
- When the user asks to "auto scale" or "normalize and compare", use auto_scale in compare_samples
- When the user asks "what element" or "identify element" or "what edge", use identify_element
- When first loading data, consider using identify_element to determine the element being measured
- If the element is guessed from energy (not metadata), ask the user to confirm
- When the user says "save", use save_data or save_image as appropriate
- Be helpful and concise
- If the request is ambiguous, make a reasonable assumption and explain what you did

NUMBERED OPTIONS — When presenting choices:
- Present options as a numbered list (1. 2. 3.)
- End with: "Or type your own request."
- The user can type a number to select that option.
""")

conversation = [{"role": "system", "content": SYSTEM_PROMPT}]


def _expand_numbered_choice(user_message: str) -> str:
    """If the user typed just a number, expand it to the full option text."""
    stripped = user_message.strip()
    if not re.match(r'^\d+\.?$', stripped):
        return user_message

    choice_num = int(stripped.rstrip('.'))

    for entry in reversed(conversation):
        role = entry.get("role") if isinstance(entry, dict) else getattr(entry, "role", None)
        content = entry.get("content") if isinstance(entry, dict) else getattr(entry, "content", None)
        if role != "assistant" or not content:
            continue

        pattern = re.compile(
            r'^\s*(?:\*\*)?(\d+)[.)]\s*(?:\*\*\s*)?(.+)$',
            re.MULTILINE
        )
        options = {}
        for m in pattern.finditer(content):
            num = int(m.group(1))
            text = m.group(2).strip()
            text = re.sub(r'\*+$', '', text).strip()
            options[num] = text

        if choice_num in options:
            return options[choice_num]

    return user_message


def agent_chat(user_message: str) -> dict:
    """Send a message to the agent and return {text, images, tools_used}."""
    global _pending_images
    _pending_images = []
    tools_used = []

    user_message = _expand_numbered_choice(user_message)
    conversation.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL, messages=conversation, tools=TOOLS, tool_choice="auto",
    )
    msg = response.choices[0].message

    while msg.tool_calls:
        conversation.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            tools_used.append(f"🔧 {fn_name}({json.dumps(fn_args)})")
            fn = TOOL_DISPATCH.get(fn_name)
            try:
                result = fn(**fn_args) if fn else f"Error: Unknown tool '{fn_name}'"
            except Exception as exc:
                result = f"Error in {fn_name}: {exc}"
            conversation.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        response = client.chat.completions.create(
            model=MODEL, messages=conversation, tools=TOOLS, tool_choice="auto",
        )
        msg = response.choices[0].message

    conversation.append(msg)

    return {
        "text": msg.content or "",
        "images": _pending_images[:],
        "tools_used": tools_used,
    }


# ── Flask App ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🔬 HERFD Agent — SSRL BL 15-2</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #2e7d32;
    color: white;
    padding: 12px 20px;
    font-size: 18px;
    font-weight: 600;
    flex-shrink: 0;
  }
  header small { font-weight: 400; opacity: 0.8; font-size: 13px; }

  #main-container {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  /* ── Left sidebar ─────────────────────────────────────────────────── */
  #sidebar {
    width: 280px;
    min-width: 200px;
    max-width: 450px;
    background: #252526;
    color: #cccccc;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #1e1e1e;
    flex-shrink: 0;
    overflow: hidden;
  }
  #sidebar-header {
    padding: 10px 14px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #888;
    border-bottom: 1px solid #333;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  #sidebar-header button {
    background: none;
    border: none;
    color: #888;
    cursor: pointer;
    font-size: 14px;
    padding: 2px 6px;
    border-radius: 4px;
  }
  #sidebar-header button:hover { background: #3c3c3c; color: #ccc; }

  #file-tree {
    flex: 1;
    overflow-y: auto;
    padding: 4px 0;
    font-size: 13px;
  }
  #file-tree::-webkit-scrollbar { width: 8px; }
  #file-tree::-webkit-scrollbar-track { background: #252526; }
  #file-tree::-webkit-scrollbar-thumb { background: #424242; border-radius: 4px; }

  .tree-item {
    display: flex;
    align-items: center;
    padding: 3px 8px;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  .tree-item:hover { background: #2a2d2e; }
  .tree-item.selected { background: #094771; color: #fff; }
  .tree-item .icon {
    width: 18px;
    text-align: center;
    margin-right: 4px;
    font-size: 12px;
    flex-shrink: 0;
  }
  .tree-item .label {
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .tree-children { display: none; }
  .tree-children.open { display: block; }

  #clean-exports-panel {
    border-top: 1px solid #333;
    padding: 8px 10px;
  }
  #clean-exports-btn {
    width: 100%;
    padding: 6px 0;
    background: #5c2020;
    color: #e0e0e0;
    border: 1px solid #7a3030;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
  }
  #clean-exports-btn:hover { background: #7a3030; color: #fff; }

  #confirm-overlay {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.55);
    z-index: 9999;
    align-items: center;
    justify-content: center;
  }
  #confirm-overlay.visible { display: flex; }
  #confirm-dialog {
    background: #2d2d2d;
    border: 1px solid #555;
    border-radius: 8px;
    padding: 24px 28px;
    min-width: 320px;
    text-align: center;
    color: #e0e0e0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }
  #confirm-dialog p { margin: 0 0 18px 0; font-size: 14px; line-height: 1.5; }
  #confirm-dialog .confirm-buttons { display: flex; gap: 12px; justify-content: center; }
  #confirm-dialog .confirm-buttons button {
    padding: 7px 22px; border-radius: 4px; border: 1px solid #555;
    font-size: 13px; cursor: pointer;
  }
  .btn-cancel { background: #3c3c3c; color: #ccc; }
  .btn-cancel:hover { background: #555; }
  .btn-confirm { background: #a03030; color: #fff; border-color: #c04040; }
  .btn-confirm:hover { background: #c04040; }

  /* ── Resize handle ──────────────────────────────────────────────── */
  #resize-handle {
    width: 4px;
    cursor: col-resize;
    background: transparent;
    flex-shrink: 0;
  }
  #resize-handle:hover { background: #2e7d32; }

  /* ── Right panel: chat ──────────────────────────────────────────── */
  #chat-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  #chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; word-wrap: break-word; }
  .msg.user {
    align-self: flex-end;
    background: #e8f5e9;
    border: 1px solid #c8e6c9;
    color: #1b5e20;
  }
  .msg.assistant {
    align-self: flex-start;
    background: white;
    border: 1px solid #e0e0e0;
    color: #333;
  }
  .msg.tool {
    align-self: flex-start;
    background: #fff3e0;
    border: 1px solid #ffe0b2;
    color: #e65100;
    font-family: monospace;
    font-size: 13px;
    padding: 6px 10px;
  }
  .msg img {
    max-width: 100%;
    border-radius: 8px;
    margin-top: 8px;
    border: 1px solid #ddd;
    cursor: pointer;
  }
  .msg img:hover { opacity: 0.9; }
  .msg pre { background: #f5f5f5; padding: 8px; border-radius: 6px; overflow-x: auto; font-size: 13px; margin: 6px 0; }
  .msg code { font-size: 13px; }
  .msg p { margin: 4px 0; }
  .msg table { border-collapse: collapse; margin: 8px 0; font-size: 13px; }
  .msg th, .msg td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
  .msg th { background: #f5f5f5; }

  #input-area {
    flex-shrink: 0;
    display: flex;
    gap: 8px;
    padding: 12px 20px;
    background: white;
    border-top: 1px solid #ddd;
  }
  #msg-input {
    flex: 1;
    padding: 10px 14px;
    border: 1px solid #ccc;
    border-radius: 8px;
    font-size: 15px;
    outline: none;
  }
  #msg-input:focus { border-color: #2e7d32; box-shadow: 0 0 0 2px rgba(46,125,50,0.2); }
  #send-btn {
    padding: 10px 24px;
    background: #2e7d32;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    cursor: pointer;
    font-weight: 600;
  }
  #send-btn:hover { background: #1b5e20; }
  #send-btn:disabled { background: #a5d6a7; cursor: not-allowed; }
  #clear-btn {
    padding: 10px 16px;
    background: #ff9800;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
  }
  #clear-btn:hover { background: #f57c00; }
  .typing { color: #888; font-style: italic; }
  .welcome {
    align-self: center;
    color: #888;
    font-style: italic;
    text-align: center;
    padding: 20px;
  }
  .welcome b { color: #555; }

  /* Image modal */
  #img-modal {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.8);
    z-index: 9999;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }
  #img-modal.visible { display: flex; }
  #img-modal img { max-width: 95%; max-height: 95%; border-radius: 8px; }
</style>
</head>
<body>
  <header>
    🔬 HERFD Agent
    <small>— SSRL BL 15-2 HERFD-XAS Analysis</small>
  </header>

  <div id="main-container">
    <!-- ── Left sidebar: file explorer ──────────────────────────────── -->
    <div id="sidebar">
      <div id="sidebar-header">
        <span>📁 Data Explorer</span>
        <button id="refresh-tree" title="Refresh file list">⟳</button>
      </div>
      <div id="file-tree"></div>
      <div id="clean-exports-panel">
        <button id="clean-exports-btn">🗑 Clean Exported Data</button>
      </div>
    </div>

    <!-- Confirm dialog -->
    <div id="confirm-overlay">
      <div id="confirm-dialog">
        <p>⚠️ This will <strong>permanently delete</strong> all files in <code>exported_data</code>.<br>Are you sure?</p>
        <div class="confirm-buttons">
          <button class="btn-cancel" id="confirm-cancel">Cancel</button>
          <button class="btn-confirm" id="confirm-yes">Delete All</button>
        </div>
      </div>
    </div>

    <div id="resize-handle"></div>

    <!-- ── Right panel: chat ────────────────────────────────────────── -->
    <div id="chat-panel">
      <div id="chat-area">
        <div class="welcome">
          🔬 HERFD Agent ready! (SSRL BL 15-2) (SSRL BL 15-2)<br>
          Try: <b>List samples</b> · <b>Plot a sample</b> · <b>Compare samples</b> · <b>Normalize a spectrum</b><br>
          <small>💡 Double-click a file in the left panel to paste its name into the chat.</small>
        </div>
      </div>
      <div id="input-area">
        <input type="text" id="msg-input" placeholder="Ask about your HERFD data…" autocomplete="off" autofocus />
        <button id="send-btn">Send</button>
        <button id="clear-btn">Clear</button>
      </div>
    </div>
  </div>

  <!-- Image modal -->
  <div id="img-modal" onclick="this.classList.remove('visible')">
    <img id="modal-img" src="">
  </div>

<script>
const chatArea = document.getElementById("chat-area");
const msgInput = document.getElementById("msg-input");
const sendBtn  = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

function addMessage(role, html) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.innerHTML = html;
  // Make images clickable for modal
  div.querySelectorAll("img").forEach(img => {
    img.addEventListener("click", (e) => {
      e.stopPropagation();
      document.getElementById("modal-img").src = img.src;
      document.getElementById("img-modal").classList.add("visible");
    });
  });
  chatArea.appendChild(div);
  scrollToBottom();
}

function addImages(images) {
  images.forEach(b64 => {
    const div = document.createElement("div");
    div.className = "msg assistant";
    const img = document.createElement("img");
    img.src = "data:image/png;base64," + b64;
    img.addEventListener("click", (e) => {
      e.stopPropagation();
      document.getElementById("modal-img").src = img.src;
      document.getElementById("img-modal").classList.add("visible");
    });
    div.appendChild(img);
    chatArea.appendChild(div);
  });
  scrollToBottom();
}

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text) return;

  // Save to input history
  inputHistory.push(text);
  historyIndex = -1;
  savedInput = "";

  addMessage("user", text);
  msgInput.value = "";
  sendBtn.disabled = true;
  msgInput.disabled = true;

  const typingDiv = document.createElement("div");
  typingDiv.className = "msg assistant typing";
  typingDiv.textContent = "Thinking…";
  chatArea.appendChild(typingDiv);
  scrollToBottom();

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: text}),
    });
    const data = await resp.json();
    typingDiv.remove();

    if (data.tools_used && data.tools_used.length > 0) {
      data.tools_used.forEach(t => addMessage("tool", t));
    }
    if (data.images && data.images.length > 0) {
      addImages(data.images);
    }
    if (data.text) {
      const rendered = typeof marked !== "undefined" ? marked.parse(data.text) : data.text;
      addMessage("assistant", rendered);
    }
    if (data.error) {
      addMessage("tool", "❌ " + data.error);
    }
    // Refresh file tree after any tool call (in case files were saved)
    if (data.tools_used && data.tools_used.length > 0) {
      setTimeout(() => loadFileTree(), 500);
    }
  } catch (err) {
    typingDiv.remove();
    addMessage("tool", "❌ Network error: " + err.message);
  }

  sendBtn.disabled = false;
  msgInput.disabled = false;
  msgInput.focus();
}

async function clearChat() {
  try { await fetch("/clear", {method: "POST"}); } catch(e) {}
  chatArea.innerHTML = '<div class="welcome">Chat cleared. Start a new conversation!</div>';
}

// ── Input history (Up/Down arrow) ──────────────────────────────────────
const inputHistory = [];
let historyIndex = -1;
let savedInput = "";

sendBtn.addEventListener("click", sendMessage);
msgInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  } else if (e.key === "ArrowUp") {
    if (inputHistory.length === 0) return;
    if (historyIndex === -1) {
      savedInput = msgInput.value;
      historyIndex = inputHistory.length - 1;
    } else if (historyIndex > 0) {
      historyIndex--;
    }
    msgInput.value = inputHistory[historyIndex];
    e.preventDefault();
  } else if (e.key === "ArrowDown") {
    if (historyIndex === -1) return;
    if (historyIndex < inputHistory.length - 1) {
      historyIndex++;
      msgInput.value = inputHistory[historyIndex];
    } else {
      historyIndex = -1;
      msgInput.value = savedInput;
    }
    e.preventDefault();
  }
});
clearBtn.addEventListener("click", clearChat);

// ── File Explorer ─────────────────────────────────────────────────────
const fileTree = document.getElementById("file-tree");
const refreshBtn = document.getElementById("refresh-tree");

function createTreeItem(node, depth) {
  const container = document.createElement("div");

  if (node.type === "dir") {
    const item = document.createElement("div");
    item.className = "tree-item";
    item.style.paddingLeft = (8 + depth * 16) + "px";
    item.innerHTML =
      '<span class="icon">▶</span>' +
      '<span class="label">📁 ' + node.name + '</span>';

    const childrenDiv = document.createElement("div");
    childrenDiv.className = "tree-children";
    (node.children || []).forEach(child => {
      childrenDiv.appendChild(createTreeItem(child, depth + 1));
    });

    item.addEventListener("click", () => {
      const isOpen = childrenDiv.classList.toggle("open");
      item.querySelector(".icon").textContent = isOpen ? "▼" : "▶";
    });

    container.appendChild(item);
    container.appendChild(childrenDiv);
  } else {
    const item = document.createElement("div");
    item.className = "tree-item";
    item.style.paddingLeft = (8 + depth * 16) + "px";
    item.innerHTML =
      '<span class="icon"> </span>' +
      '<span class="label">📄 ' + node.name + '</span>';

    // Double-click to paste filename into chat input
    item.addEventListener("dblclick", (e) => {
      e.preventDefault();
      const fullName = node.name;
      const input = document.getElementById("msg-input");
      const start = input.selectionStart;
      const end = input.selectionEnd;
      const val = input.value;
      input.value = val.substring(0, start) + fullName + val.substring(end);
      input.focus();
      input.selectionStart = input.selectionEnd = start + fullName.length;
      document.querySelectorAll(".tree-item.selected").forEach(el => el.classList.remove("selected"));
      item.classList.add("selected");
    });

    item.addEventListener("click", () => {
      document.querySelectorAll(".tree-item.selected").forEach(el => el.classList.remove("selected"));
      item.classList.add("selected");
    });

    container.appendChild(item);
  }

  return container;
}

async function loadFileTree() {
  fileTree.innerHTML = '<div style="padding:12px;color:#888;font-size:12px;">Loading…</div>';
  try {
    const resp = await fetch("/api/files");
    const data = await resp.json();
    fileTree.innerHTML = "";

    data.trees.forEach(rootNode => {
      const rootItem = document.createElement("div");
      rootItem.className = "tree-item";
      rootItem.style.paddingLeft = "8px";
      rootItem.innerHTML =
        '<span class="icon">▼</span>' +
        '<span class="label" style="font-weight:600;">📁 ' + rootNode.name + '</span>';

      const rootChildren = document.createElement("div");
      rootChildren.className = "tree-children open";
      (rootNode.children || []).forEach(child => {
        rootChildren.appendChild(createTreeItem(child, 1));
      });

      rootItem.addEventListener("click", () => {
        const isOpen = rootChildren.classList.toggle("open");
        rootItem.querySelector(".icon").textContent = isOpen ? "▼" : "▶";
      });

      fileTree.appendChild(rootItem);
      fileTree.appendChild(rootChildren);
    });
  } catch (err) {
    fileTree.innerHTML = '<div style="padding:12px;color:#f44;font-size:12px;">Failed to load files</div>';
  }
}

refreshBtn.addEventListener("click", loadFileTree);

// ── Sidebar resize ────────────────────────────────────────────────────
const sidebar = document.getElementById("sidebar");
const resizeHandle = document.getElementById("resize-handle");
let isResizing = false;

resizeHandle.addEventListener("mousedown", (e) => {
  isResizing = true;
  document.body.style.cursor = "col-resize";
  document.body.style.userSelect = "none";
  e.preventDefault();
});

document.addEventListener("mousemove", (e) => {
  if (!isResizing) return;
  const newWidth = e.clientX;
  if (newWidth >= 150 && newWidth <= 500) {
    sidebar.style.width = newWidth + "px";
  }
});

document.addEventListener("mouseup", () => {
  if (isResizing) {
    isResizing = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }
});

// ── Clean exported data ──────────────────────────────────────────────
const cleanBtn = document.getElementById("clean-exports-btn");
const confirmOverlay = document.getElementById("confirm-overlay");
const confirmCancel = document.getElementById("confirm-cancel");
const confirmYes = document.getElementById("confirm-yes");

cleanBtn.addEventListener("click", () => { confirmOverlay.classList.add("visible"); });
confirmCancel.addEventListener("click", () => { confirmOverlay.classList.remove("visible"); });
confirmOverlay.addEventListener("click", (e) => {
  if (e.target === confirmOverlay) confirmOverlay.classList.remove("visible");
});
confirmYes.addEventListener("click", async () => {
  confirmOverlay.classList.remove("visible");
  try {
    const resp = await fetch("/api/clear-exports", { method: "POST" });
    const data = await resp.json();
    if (data.status === "ok") {
      addMessage("assistant", "🗑 Exported data cleaned: " + data.deleted + " file(s) removed.");
      loadFileTree();
    } else {
      addMessage("assistant", "⚠️ Error: " + (data.error || "unknown"));
    }
  } catch (err) {
    addMessage("assistant", "⚠️ Network error: " + err.message);
  }
});

// Load file tree on startup
loadFileTree();
</script>
</body>
</html>
"""



@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400
    try:
        result = agent_chat(user_msg)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "text": "", "images": [], "tools_used": []}), 500


@app.route("/clear", methods=["POST"])
def clear_endpoint():
    global conversation, _cache
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    _cache.clear()
    return jsonify({"status": "ok"})



@app.route("/api/files", methods=["GET"])
def list_data_files():
    """Return the data directory tree as JSON for the file explorer panel."""
    def _build_tree(root_dir: str) -> list:
        entries = []
        try:
            items = sorted(os.listdir(root_dir))
        except OSError:
            return entries
        dirs = sorted([i for i in items if os.path.isdir(os.path.join(root_dir, i))], reverse=True)
        files = [i for i in items if os.path.isfile(os.path.join(root_dir, i))
                 and i.endswith(".dat") and not i.startswith(".")]
        for d in dirs:
            if d.startswith("."):
                continue
            children = _build_tree(os.path.join(root_dir, d))
            entries.append({"name": d, "type": "dir", "children": children})
        for f in files:
            entries.append({"name": f, "type": "file"})
        return entries

    trees = [{"name": os.path.basename(DATA_DIR), "type": "dir",
              "children": _build_tree(DATA_DIR)}]
    if os.path.isdir(EXPORT_DIR):
        trees.append({"name": "exported_data", "type": "dir",
                       "children": _build_tree(EXPORT_DIR)})
    return jsonify({"trees": trees})


@app.route("/api/clear-exports", methods=["POST"])
def clear_exports():
    """Delete all files and subdirectories inside exported_data/."""
    if not os.path.isdir(EXPORT_DIR):
        return jsonify({"status": "ok", "deleted": 0})
    count = 0
    for entry in os.listdir(EXPORT_DIR):
        entry_path = os.path.join(EXPORT_DIR, entry)
        try:
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.unlink(entry_path)
                count += 1
            elif os.path.isdir(entry_path):
                n = sum(len(files) for _, _, files in os.walk(entry_path))
                shutil.rmtree(entry_path)
                count += n
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500
    return jsonify({"status": "ok", "deleted": count})


if __name__ == "__main__":
    print("=" * 60)
    print("  HERFD Agent Chat")
    print("  Open http://localhost:5051 in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5051, debug=False)
