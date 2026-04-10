#!/usr/bin/env python3
"""
AI CSV Profiler
===============
A powerful, resilient tool for analyzing the structure and content of CSV
files. Generates a detailed, AI-friendly JSON report covering column types,
statistics, data quality warnings, and file metadata.

Operates in two modes:
    GUI Mode : A Tkinter interface for interactive use.
    CLI Mode : A command-line interface for scripting and automation.

Key design principles:
    - Resilience   : Every operation is wrapped in error handlers. The
                     profiler never crashes regardless of input quality.
    - Accuracy     : Heuristic type detection with currency, boolean,
                     datetime, categorical, and text support.
    - Transparency : Warnings vs. info are kept separate so that a clean
                     run always shows zero warnings.

Dependencies:
    Standard library : argparse, json, os, queue, re, sys, threading,
                       traceback, pathlib, tkinter
    Third-party      : pandas, numpy (install via: pip install pandas)
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import argparse
import json
import os
import queue
import re
import sys
import threading
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# GUI Library Import
# ---------------------------------------------------------------------------
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---------------------------------------------------------------------------
# Third-Party Library Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import warnings as _warnings

# ---------------------------------------------------------------------------
# Global Warning Suppression
# ---------------------------------------------------------------------------
# DtypeWarning : Raised when pandas infers mixed types. We read all columns
#                as strings initially, so this is expected and handled.
# RuntimeWarning: Can occur during statistical calculations on columns with
#                 extreme outlier values. Our safe analyzers handle these.
_warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
_warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Currency Detection Constants
# ---------------------------------------------------------------------------
# Covers the full Unicode Currency Symbols block (U+20A0 to U+20CF) as well
# as the four most common symbols that fall outside it: $ £ ¥ €.
# This handles USD, EUR, GBP, JPY, INR, KRW, ILS, TRY, RUB, PHP, THB,
# VND, UAH, NGN, GHS, PYG, CRC, and more without any external library.
_CURRENCY_SYMBOLS = r"\$€£¥\u20a0-\u20cf"

# Matches an optional leading symbol, optional negative indicator (minus or
# opening parenthesis), digits with separators, and an optional closing paren.
_CURRENCY_VALUE_RE = re.compile(
    r"^\s*["
    + _CURRENCY_SYMBOLS
    + r"]?\s*"   # optional leading currency symbol
    r"[\(\-]?\s*"           # optional negative: '-' or opening '('
    r"[\d,\.\s]+"           # digits, commas, dots, spaces (value body)
    r"\s*\)?\s*$"           # optional closing ')' for accounting negatives
)

# Used to strip currency symbols and surrounding whitespace before float().
_CURRENCY_SYMBOL_STRIP_RE = re.compile(r"[" + _CURRENCY_SYMBOLS + r"\s]")

# ---------------------------------------------------------------------------
# Outlier Warning Thresholds
# ---------------------------------------------------------------------------
# These control how many standard deviations from the clipped mean a raw
# min or max must be before a data quality warning is emitted.
#
# Two separate thresholds are used because financial data naturally has long
# tails — a column containing both a $225 part-time allocation and a $467K
# annual salary is real data, not corruption. Plain numeric columns (like
# FTE, which should sit between 0 and 1) are held to a tighter standard.
#
# HOW TO TUNE:
#   If currency columns warn on legitimate salary spread:
#       Raise _OUTLIER_SD_THRESHOLD_CURRENCY (try 30-50)
#   If FTE or count columns are not catching obvious corruption:
#       Lower _OUTLIER_SD_THRESHOLD_NUMERIC (try 3)
#   If all warnings are noise for your dataset:
#       Raise both thresholds
#   If you want warnings on any value beyond 2 SDs:
#       Lower both to 2-3
_OUTLIER_SD_THRESHOLD_NUMERIC = 5    # for plain numeric columns (FTE, counts, etc.)
_OUTLIER_SD_THRESHOLD_CURRENCY = 30  # for currency columns (salary, revenue, etc.)


class DefensiveAnalyzer:
    """
    Utility class providing static helper methods for safe data operations.

    Every method is designed to return a sensible default rather than raise
    an exception, making callers resilient to unexpected or malformed input.
    """

    @staticmethod
    def safe_execute(func, *args, default=None, **kwargs):
        """
        Execute *func* and return its result; return *default* on any error.

        This is the foundational safety wrapper used throughout the profiler.
        It also guards against non-finite numeric results (NaN, +/-inf) which
        can silently corrupt downstream JSON serialization.

        Parameters
        ----------
        func : callable
            The function to execute.
        *args :
            Positional arguments forwarded to *func*.
        default :
            Value returned if *func* raises or returns a non-finite number.
        **kwargs :
            Keyword arguments forwarded to *func*.

        Returns
        -------
        Any
            Result of *func*, or *default* on failure.
        """
        try:
            result = func(*args, **kwargs)
            if isinstance(result, (int, float)):
                if pd.isna(result) or np.isinf(result):
                    return default
            return result
        except Exception:
            return default

    @staticmethod
    def safe_convert_numeric(series, fill_value=np.nan):
        """
        Convert a pandas Series to numeric with maximum safety.

        Cleans common null placeholders (NULL, N/A, #DIV/0!, etc.) and
        replaces +/-inf with NaN. If fewer than 10% of values convert
        successfully the original series is returned unchanged, preventing
        ID-like columns from being misclassified as numeric.

        Parameters
        ----------
        series : pd.Series
            Input series (typically dtype object/string).
        fill_value : scalar, optional
            Value used to fill unconvertible entries. Default: np.nan.

        Returns
        -------
        pd.Series
            Numeric series, or the original series on widespread failure.
        """
        try:
            if series.dtype == "object":
                cleaned = series.astype(str).str.strip()
                null_placeholders = [
                    "", "NULL", "null", "None", "none", "N/A", "n/a",
                    "#N/A", "#NULL!", "#DIV/0!", "#REF!", "#NAME?",
                    "#NUM!", "#VALUE!", "inf", "-inf", "infinity",
                    "-infinity",
                ]
                cleaned = cleaned.replace(null_placeholders, np.nan)
                result = pd.to_numeric(cleaned, errors="coerce")
            else:
                result = pd.to_numeric(series, errors="coerce")

            # Treat +/-inf as missing rather than letting them skew stats.
            result = result.replace([np.inf, -np.inf], np.nan)

            # Bail out if conversion produced very few valid numbers —
            # this prevents columns like 'product_code' being mis-typed.
            if result.notna().sum() < len(result) * 0.1:
                return series.fillna(fill_value)

            return result.fillna(fill_value)

        except Exception:
            return series.fillna(fill_value)

    @staticmethod
    def safe_convert_currency(series, fill_value=np.nan):
        """
        Convert a currency-formatted Series to float.

        Handles the following formats:
            - Leading/trailing symbols : $, EUR, GBP, JPY, all Unicode currency
            - Thousands separators     : commas (e.g. 1,234.56)
            - Accounting negatives     : (1,234.56) becomes -1234.56
            - Standard negatives       : -1234.56

        Parameters
        ----------
        series : pd.Series
            Input series containing currency-formatted strings.
        fill_value : scalar, optional
            Value used for values that cannot be parsed. Default: np.nan.

        Returns
        -------
        pd.Series
            Float series with currency formatting removed.
        """
        null_strings = {"", "nan", "None", "NULL", "N/A"}

        def _parse_one(val):
            """Parse a single currency string to float."""
            s = str(val).strip()
            if s in null_strings:
                return fill_value

            # Accounting-style negative: value wrapped in parentheses.
            is_negative = s.startswith("(") and s.endswith(")")

            # Strip currency symbols and surrounding whitespace.
            s = _CURRENCY_SYMBOL_STRIP_RE.sub("", s)
            # Remove parentheses used for accounting negatives.
            s = s.strip("()")
            # Remove thousands-separator commas.
            s = s.replace(",", "").strip()

            try:
                value = float(s)
                return -value if is_negative else value
            except (ValueError, TypeError):
                return fill_value

        try:
            return series.apply(_parse_one)
        except Exception:
            return series.fillna(fill_value)

    @staticmethod
    def safe_statistics(data, stat_name):
        """
        Compute a single statistic on *data* with outlier clipping.

        For mean, std, min, and max the computation is performed on the
        inner 98% of values (1st-99th percentile) to prevent extreme
        outliers from producing misleading summary statistics. The raw
        extremes are reported separately via raw_statistics.

        Parameters
        ----------
        data : pd.Series
            Numeric data (may contain NaN; these are dropped internally).
        stat_name : str
            One of: 'min', 'max', 'mean', 'median', 'std', 'count', or
            'quantile_<fraction>' (e.g. 'quantile_0.25').

        Returns
        -------
        float or int or None
            Computed statistic, or None if data is empty or an error occurs.
        """
        try:
            if len(data) == 0:
                return None

            clean = data.dropna()
            if len(clean) == 0:
                return None

            # Clip to the inner 98% for stats sensitive to extreme values.
            if stat_name in ("mean", "std", "min", "max"):
                q01 = clean.quantile(0.01)
                q99 = clean.quantile(0.99)
                if pd.notna(q01) and pd.notna(q99):
                    clean = clean[(clean >= q01) & (clean <= q99)]

            if len(clean) == 0:
                return None

            dispatch = {
                "min":    lambda d: float(d.min()),
                "max":    lambda d: float(d.max()),
                "mean":   lambda d: float(d.mean()),
                "median": lambda d: float(d.median()),
                "std":    lambda d: float(d.std()) if len(d) > 1 else 0.0,
                "count":  lambda d: len(d),
            }

            if stat_name in dispatch:
                return dispatch[stat_name](clean)

            if stat_name.startswith("quantile_"):
                q = float(stat_name.split("_")[1])
                return float(clean.quantile(q))

            return None

        except Exception:
            return None


class UltraRobustCSVProfiler:
    """
    Core profiling engine.

    Orchestrates the full pipeline:
        1. Read the CSV using a multi-strategy fallback approach.
        2. Strip leading/trailing whitespace from column names.
        3. Detect and analyze each column individually.
        4. Emit data quality warnings for suspect values.
        5. Compile and return the final JSON-serializable report dict.
    """

    def __init__(self):
        """Initialize the profiler with encoding and separator candidates."""
        self.analyzer = DefensiveAnalyzer()

        # Encodings to attempt in priority order. utf-8-sig is first so that
        # BOM-encoded files (common in Windows CSV exports) are handled
        # cleanly without producing a junk character in the first column name.
        self.encoding_attempts = [
            "utf-8-sig", "utf-8", "cp1252", "iso-8859-1",
            "latin1", "ascii", "utf-16", "utf-32",
        ]

        # Separators to attempt in priority order.
        self.separator_attempts = [",", ";", "\t", "|", ":", " "]

        # Maximum file size (bytes) before switching to chunked reading.
        self.chunk_threshold_bytes = 100 * 1024 * 1024  # 100 MB

        # Number of rows per chunk when reading large files.
        self.chunk_size = 50_000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, file_path: str) -> dict:
        """
        Generate a comprehensive profile of *file_path*.

        This is the single public entry point. It always returns a
        JSON-serializable dict; on critical failure the dict carries
        status='error' and a traceback key rather than raising.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the target CSV file.

        Returns
        -------
        dict
            Profile report with keys: file, status, shape, columns,
            warnings, info, metadata.
        """
        base_profile = {
            "file": os.path.basename(file_path),
            "status": "unknown",
            "shape": {"rows": 0, "columns": 0},
            "columns": [],
            "warnings": [],
            "info": [],
            "metadata": {},
        }

        try:
            df, read_warnings, read_info = self._read_csv_ultra_safe(
                file_path
            )
            base_profile["warnings"].extend(read_warnings)
            base_profile["info"].extend(read_info)

            if df is None or df.empty:
                base_profile["status"] = "empty_or_unreadable"
                base_profile["error"] = (
                    "Could not read file or file is empty"
                )
                return base_profile

            # Strip leading/trailing whitespace from all column names.
            # This prevents downstream issues when names are used as dict
            # keys, Power BI field references, or pandas column selectors.
            df.columns = df.columns.str.strip()

            base_profile["shape"] = {
                "rows": len(df),
                "columns": len(df.columns),
            }
            base_profile["status"] = "processed"
            base_profile["columns"] = self._analyze_all_columns_safe(df)
            base_profile["warnings"].extend(
                self._check_data_quality(base_profile["columns"])
            )
            base_profile["metadata"] = self._get_file_metadata_safe(
                file_path, df
            )

            return base_profile

        except Exception as exc:
            base_profile["status"] = "error"
            base_profile["error"] = f"Critical error: {exc}"
            base_profile["traceback"] = traceback.format_exc()
            return base_profile

    # ------------------------------------------------------------------
    # CSV Reading
    # ------------------------------------------------------------------

    def _read_csv_ultra_safe(self, file_path: str) -> tuple:
        """
        Read a CSV using a four-stage fallback strategy.

        Stages
        ------
        1. Explicit encoding x separator x engine matrix.
           Tries every combination of encoding, separator, and pandas
           engine (C engine first for speed, Python engine as fallback).
           Covers the vast majority of real-world files.
        2. Common encodings with auto-separator detection.
           Catches BOM files and other edge cases that slip past stage 1.
        3. UTF-8 full auto-separator fallback.
           A warning is emitted if this stage is reached.
        4. Manual line-by-line text parsing.
           Last resort for severely malformed files.

        For files larger than self.chunk_threshold_bytes the read is
        performed in chunks of self.chunk_size rows to keep memory
        usage predictable.

        Returns
        -------
        tuple[pd.DataFrame | None, list[str], list[str]]
            (dataframe, warnings_list, info_list)
        """
        warnings_list = []
        info_list = []

        # ---- Pre-flight size check ----------------------------------------
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return None, ["File is empty"], info_list
            if file_size > 500 * 1024 * 1024:
                warnings_list.append(
                    f"Very large file ({file_size / 1024 / 1024:.1f} MB) "
                    f"— consider splitting before profiling."
                )
            elif file_size > self.chunk_threshold_bytes:
                warnings_list.append(
                    f"Large file ({file_size / 1024 / 1024:.1f} MB) "
                    f"— using chunked reading."
                )
        except OSError:
            warnings_list.append("Could not determine file size.")

        use_chunks = (
            os.path.getsize(file_path) > self.chunk_threshold_bytes
        )

        # ---- Stage 1: explicit encoding x separator x engine matrix --------
        for encoding in self.encoding_attempts:
            for separator in self.separator_attempts:
                for engine in ("c", "python"):
                    read_kwargs = dict(
                        encoding=encoding,
                        sep=separator,
                        on_bad_lines="skip",
                        low_memory=False,
                        dtype=str,
                        na_values=[""],
                        keep_default_na=False,
                        engine=engine,
                    )
                    df = self._attempt_read(
                        file_path, read_kwargs, use_chunks
                    )
                    if df is not None and len(df.columns) > 1 and len(df) > 0:
                        info_list.append(
                            f"Read with encoding={encoding}, "
                            f"separator='{separator}', engine={engine}"
                        )
                        return df, warnings_list, info_list

        # ---- Stage 1.5: common encodings with auto-separator ---------------
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1"):
            read_kwargs = dict(
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines="skip",
                dtype=str,
                na_values=[""],
                keep_default_na=False,
            )
            df = self._attempt_read(file_path, read_kwargs, use_chunks)
            if df is not None and len(df.columns) > 1 and len(df) > 0:
                info_list.append(
                    f"Read with encoding={encoding}, "
                    f"separator=auto-detected"
                )
                return df, warnings_list, info_list

        # ---- Stage 2: utf-8 full auto-separator fallback -------------------
        try:
            df = pd.read_csv(
                file_path,
                encoding="utf-8",
                sep=None,
                engine="python",
                on_bad_lines="skip",
                dtype=str,
            )
            warnings_list.append(
                "Used fallback reading method (auto-separator)."
            )
            return df, warnings_list, info_list
        except Exception:
            pass

        # ---- Stage 3: manual line-by-line text parsing ---------------------
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()[:1000]

            if not lines:
                return None, ["File appears empty or unreadable."], info_list

            first_line = lines[0].strip()
            best_sep = max(
                self.separator_attempts,
                key=lambda s: first_line.count(s),
            )
            headers = first_line.split(best_sep)

            rows = []
            for line in lines[1:]:
                row = line.strip().split(best_sep)
                # Pad short rows and truncate long rows to match header.
                row = (row + [""] * len(headers))[: len(headers)]
                rows.append(row)

            df = pd.DataFrame(rows, columns=headers)
            warnings_list.append(
                "Used manual text parsing as last resort."
            )
            return df, warnings_list, info_list

        except Exception:
            return (
                None,
                warnings_list + ["All reading methods failed."],
                info_list,
            )

    def _attempt_read(
        self,
        file_path: str,
        read_kwargs: dict,
        use_chunks: bool,
    ):
        """
        Attempt a single pd.read_csv call, optionally reading in chunks.

        Chunked reading concatenates self.chunk_size rows at a time so
        that large files do not need to fit entirely in RAM before the
        profiler can begin working.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.
        read_kwargs : dict
            Keyword arguments passed directly to pd.read_csv.
        use_chunks : bool
            If True, read via chunksize iterator and concatenate.

        Returns
        -------
        pd.DataFrame or None
            Parsed dataframe, or None if the attempt fails.
        """
        try:
            if use_chunks:
                chunks = pd.read_csv(
                    file_path,
                    chunksize=self.chunk_size,
                    **read_kwargs,
                )
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path, **read_kwargs)
            return df
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Data Quality Checks
    # ------------------------------------------------------------------

    def _check_data_quality(self, columns: list) -> list:
        """
        Scan analyzed column profiles for data quality issues.

        Currently checks:
            - Numeric/currency columns where the raw min or max is more
              than the type-specific SD threshold from the clipped mean,
              indicating likely data entry errors or corrupt values.

        Thresholds are intentionally different per type:
            - Numeric  : _OUTLIER_SD_THRESHOLD_NUMERIC  (default 5)
            - Currency : _OUTLIER_SD_THRESHOLD_CURRENCY (default 20)

        Financial data naturally has long tails (e.g. a $225 part-time
        allocation alongside a $467K salary in the same column), so the
        currency threshold is set much higher to avoid false positives.
        If warnings are still too noisy or too quiet for your specific
        data, adjust the two threshold constants at the top of this file.

        Parameters
        ----------
        columns : list[dict]
            Column profile dicts produced by _analyze_all_columns_safe.

        Returns
        -------
        list[str]
            Zero or more human-readable warning strings.
        """
        quality_warnings = []

        for col in columns:
            col_name = col.get("name", "unknown")
            col_type = col.get("type", "")

            # Only check types that carry numeric statistics.
            if col_type not in ("numeric", "currency"):
                continue

            raw_stats = col.get("raw_statistics")
            stats = col.get("statistics", {})

            if not raw_stats or not stats:
                continue

            mean = stats.get("mean")
            std = stats.get("std")
            raw_min = raw_stats.get("min")
            raw_max = raw_stats.get("max")

            # Skip if we lack the values needed to compute the threshold.
            if any(v is None for v in (mean, std, raw_min, raw_max)):
                continue

            # Avoid division by zero for columns with zero variance.
            if std == 0:
                continue

            # Use a looser threshold for currency columns since financial
            # data naturally has a wider spread than plain numeric columns.
            threshold = (
                _OUTLIER_SD_THRESHOLD_CURRENCY
                if col_type == "currency"
                else _OUTLIER_SD_THRESHOLD_NUMERIC
            )

            min_sd = abs((raw_min - mean) / std)
            max_sd = abs((raw_max - mean) / std)
            worst_sd = max(min_sd, max_sd)

            if worst_sd > threshold:
                quality_warnings.append(
                    f"Column '{col_name}': raw min/max values are "
                    f"{worst_sd:.0f} standard deviations from the mean "
                    f"(threshold: {threshold} SDs for type '{col_type}') "
                    f"— possible data entry error or corrupt values. "
                    f"See raw_statistics for details."
                )

        return quality_warnings

    # ------------------------------------------------------------------
    # Column Analysis
    # ------------------------------------------------------------------

    def _analyze_all_columns_safe(self, df: pd.DataFrame) -> list:
        """
        Analyze every column in *df*, isolating errors per column.

        A failure in one column's analysis never prevents the remaining
        columns from being processed — the failed column receives an error
        entry instead.

        Parameters
        ----------
        df : pd.DataFrame
            The fully loaded (and column-name-stripped) dataframe.

        Returns
        -------
        list[dict]
            One profile dict per column.
        """
        column_profiles = []
        for position, col_name in enumerate(df.columns):
            try:
                profile = self._analyze_single_column_safe(
                    df[col_name], str(col_name)
                )
                profile["position"] = position
                column_profiles.append(profile)
            except Exception as exc:
                column_profiles.append({
                    "name": str(col_name),
                    "position": position,
                    "type": "error",
                    "status": "analysis_failed",
                    "error": str(exc),
                    "samples": [],
                })

        return column_profiles

    def _analyze_single_column_safe(
        self, series: pd.Series, col_name: str
    ) -> dict:
        """
        Analyze a single column and return its profile dict.

        Computes base metrics (missing count, unique count, samples) for
        all columns, then dispatches to the type-specific analyzer.

        Parameters
        ----------
        series : pd.Series
            Column data.
        col_name : str
            Column name (used for type-detection heuristics).

        Returns
        -------
        dict
            Profile with keys: name, type, status, total_values, samples,
            missing, empty_strings, unique, analysis, plus type-specific
            fields.
        """
        base = {
            "name": col_name,
            "type": "unknown",
            "status": "analyzing",
            "total_values": len(series),
            "samples": [],
        }

        try:
            base["samples"] = self._get_safe_samples(series)
            base["missing"] = self.analyzer.safe_execute(
                lambda: int(series.isnull().sum()), default=0
            )
            base["empty_strings"] = self.analyzer.safe_execute(
                lambda: int((series.astype(str) == "").sum()), default=0
            )
            base["unique"] = self.analyzer.safe_execute(
                lambda: int(series.nunique()), default=0
            )

            detected_type = self._detect_column_type_safe(series)
            base["type"] = detected_type

            # Dispatch to the appropriate type-specific analyzer.
            type_dispatch = {
                "currency":    self._analyze_currency_safe,
                "numeric":     self._analyze_numeric_ultra_safe,
                "boolean":     self._analyze_boolean_safe,
                "datetime":    self._analyze_datetime_safe,
                "categorical": self._analyze_categorical_safe,
            }
            analyzer_fn = type_dispatch.get(
                detected_type, self._analyze_text_safe
            )
            base.update(analyzer_fn(series))

            base["status"] = "completed"
            return base

        except Exception as exc:
            base["status"] = "error"
            base["error"] = str(exc)
            base["type"] = "error"
            return base

    # ------------------------------------------------------------------
    # Type Detection
    # ------------------------------------------------------------------

    def _detect_column_type_safe(self, series: pd.Series) -> str:
        """
        Determine the most appropriate type label for a column.

        Heuristics are applied in this order to resolve ambiguities:
            1. Boolean  : small set of known true/false patterns.
            2. Datetime : column name contains date/time keywords.
            3. ID/key   : column name contains id/uuid/key -> text.
            4. Currency : values match the currency regex pattern.
            5. Numeric  : >90% of values parse as float.
            6. Datetime : general datetime attempt (no name hint).
            7. Categorical / Text : based on cardinality ratio.

        Parameters
        ----------
        series : pd.Series
            Column data (raw strings).

        Returns
        -------
        str
            One of: 'boolean', 'datetime', 'text', 'currency', 'numeric',
            'categorical', 'empty'.
        """
        try:
            clean = series.dropna()
            if len(clean) == 0:
                return "empty"

            str_vals = clean.astype(str).str.strip()
            str_vals = str_vals[str_vals != ""]
            if len(str_vals) == 0:
                return "empty"

            lower_unique = set(str_vals.str.lower().unique())

            # ---- 1. Boolean ------------------------------------------------
            boolean_patterns = [
                {"true", "false"},
                {"yes", "no"},
                {"y", "n"},
                {"1", "0"},
                {"on", "off"},
                {"enabled", "disabled"},
                {"active", "inactive"},
            ]
            for pattern in boolean_patterns:
                if len(lower_unique) <= 2 and lower_unique.issubset(pattern):
                    return "boolean"

            col_name_lower = (
                str(series.name).lower() if series.name else ""
            )

            # ---- 2. Datetime (name hint) ------------------------------------
            date_keywords = ("date", "dt", "time", "start", "end")
            if any(kw in col_name_lower for kw in date_keywords):
                if self._could_be_datetime(str_vals):
                    return "datetime"

            # ---- 3. ID / key -> always text ---------------------------------
            id_keywords = ("id", "response", "uuid", "key")
            if any(kw in col_name_lower for kw in id_keywords):
                return "text"

            # ---- 4. Currency ------------------------------------------------
            is_currency, _ = self._is_currency_column(str_vals)
            if is_currency:
                return "currency"

            # ---- 5. Numeric -------------------------------------------------
            try:
                numeric_series = pd.to_numeric(str_vals, errors="coerce")
                numeric_ratio = numeric_series.notna().sum() / len(str_vals)
                if numeric_ratio > 0.9:
                    unique_numeric = set(numeric_series.dropna().unique())
                    # A numeric column with only 0 and 1 is boolean.
                    if len(unique_numeric) <= 2 and unique_numeric.issubset(
                        {0, 1, 0.0, 1.0}
                    ):
                        return "boolean"
                    return "numeric"
            except Exception:
                pass

            # ---- 6. Datetime (general) --------------------------------------
            if self._could_be_datetime(str_vals):
                return "datetime"

            # ---- 7. Categorical vs text (cardinality) -----------------------
            unique_count = len(lower_unique)
            total_count = len(str_vals)
            if unique_count <= min(50, total_count * 0.5):
                return "categorical"

            return "text"

        except Exception:
            # Safest fallback — never crash the type detection step.
            return "text"

    def _is_currency_column(self, str_values: pd.Series) -> tuple:
        """
        Determine whether a Series of strings looks like currency values.

        Sampling strategy: examine up to 100 values. A column qualifies
        when at least 80% of the sample matches _CURRENCY_VALUE_RE AND
        at least one actual currency symbol is found in the sample
        (preventing false positives on plain comma-formatted numbers).

        Parameters
        ----------
        str_values : pd.Series
            Non-null, stripped string values.

        Returns
        -------
        tuple[bool, str | None]
            (is_currency, detected_symbol) where detected_symbol is the
            first matched symbol or None.
        """
        try:
            sample = str_values.head(100)
            if len(sample) == 0:
                return False, None

            match_count = sum(
                1 for v in sample if _CURRENCY_VALUE_RE.match(str(v))
            )
            if match_count / len(sample) < 0.8:
                return False, None

            # Confirm at least one explicit currency symbol is present.
            joined = " ".join(sample.astype(str))
            symbol_match = re.search(
                r"[" + _CURRENCY_SYMBOLS + r"]", joined
            )
            if not symbol_match:
                return False, None

            return True, symbol_match.group(0)

        except Exception:
            return False, None

    def _could_be_datetime(self, str_series: pd.Series) -> bool:
        """
        Heuristically check whether a sample of strings look like dates.

        Samples the first 10 values and attempts pd.to_datetime on each.
        A column is considered datetime if at least 70% of the sample
        parses successfully.

        Parameters
        ----------
        str_series : pd.Series
            Non-null, stripped string values.

        Returns
        -------
        bool
            True if the sample is likely datetime-formatted.
        """
        try:
            sample = str_series.head(10)
            success = 0
            for val in sample:
                s = str(val).strip()
                # Short strings (< 8 chars) are unlikely to be full dates.
                if len(s) < 8:
                    continue
                # Quick pre-filter: real dates contain at least one separator.
                if any(ch in s for ch in ("-", "/", ":", "T", " ")):
                    try:
                        if pd.notna(pd.to_datetime(s, errors="raise")):
                            success += 1
                    except Exception:
                        continue
            return success >= len(sample) * 0.7
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Shared Numeric Statistics Builder
    # ------------------------------------------------------------------

    def _build_numeric_stats(self, clean_data: pd.Series) -> dict:
        """
        Compute the standard numeric statistics block.

        Shared between _analyze_numeric_ultra_safe and
        _analyze_currency_safe to avoid duplication.

        Parameters
        ----------
        clean_data : pd.Series
            Numeric series with NaN already dropped.

        Returns
        -------
        dict
            Contains: statistics, raw_statistics (if applicable),
            quartiles, zero_count, negative_count.
        """
        result = {}

        # Clipped statistics (1%-99%).
        stats = {}
        for stat_name in ("min", "max", "mean", "median", "std"):
            val = self.analyzer.safe_statistics(clean_data, stat_name)
            if val is not None:
                stats[stat_name] = (
                    round(val, 4) if isinstance(val, float) else val
                )
        result["statistics"] = stats

        # Raw extremes — only emitted when they differ from clipped values,
        # signalling the presence of genuine outliers in the data.
        try:
            raw_min = float(clean_data.min())
            raw_max = float(clean_data.max())
            if raw_min != stats.get("min") or raw_max != stats.get("max"):
                result["raw_statistics"] = {
                    "min": raw_min,
                    "max": raw_max,
                    "note": "Raw min/max before 1%-99% outlier clipping",
                }
        except Exception:
            pass

        # Quartiles.
        quartiles = {}
        for q_frac, q_name in ((0.25, "q25"), (0.75, "q75")):
            q_val = self.analyzer.safe_statistics(
                clean_data, f"quantile_{q_frac}"
            )
            if q_val is not None:
                quartiles[q_name] = round(q_val, 4)
        result["quartiles"] = quartiles

        result["zero_count"] = self.analyzer.safe_execute(
            lambda: int((clean_data == 0).sum()), default=0
        )
        result["negative_count"] = self.analyzer.safe_execute(
            lambda: int((clean_data < 0).sum()), default=0
        )

        return result

    # ------------------------------------------------------------------
    # Type-Specific Analyzers
    # ------------------------------------------------------------------

    def _analyze_currency_safe(self, series: pd.Series) -> dict:
        """
        Analyze a currency-formatted column.

        Strips symbols and formatting via safe_convert_currency, then
        runs the standard numeric statistics block. Also records the
        detected currency symbol and the count of values that could not
        be parsed.

        Parameters
        ----------
        series : pd.Series
            Raw currency-formatted strings.

        Returns
        -------
        dict
            Profile with analysis='currency', currency_symbol, and full
            numeric stats.
        """
        result = {"analysis": "currency"}
        try:
            str_vals = series.dropna().astype(str).str.strip()
            _, detected_symbol = self._is_currency_column(str_vals)
            if detected_symbol:
                result["currency_symbol"] = detected_symbol

            numeric_data = self.analyzer.safe_convert_currency(series)
            clean_data = numeric_data.dropna()

            if len(clean_data) == 0:
                result["status"] = "no_valid_values"
                return result

            result["valid_numbers"] = len(clean_data)
            result["invalid_numbers"] = len(series) - len(clean_data)
            result.update(self._build_numeric_stats(clean_data))

            return result

        except Exception as exc:
            result["error"] = str(exc)
            return result

    def _analyze_numeric_ultra_safe(self, series: pd.Series) -> dict:
        """
        Analyze a plain numeric column.

        Converts values via safe_convert_numeric then runs the shared
        _build_numeric_stats block. Also detects likely-ID columns
        (all unique values) and likely-categorical numeric columns
        (20 or fewer distinct values).

        Parameters
        ----------
        series : pd.Series
            Raw strings that are predominantly numeric.

        Returns
        -------
        dict
            Profile with analysis='numeric' and full numeric stats.
        """
        result = {"analysis": "numeric"}
        try:
            numeric_data = self.analyzer.safe_convert_numeric(series)
            clean_data = numeric_data.dropna()

            if len(clean_data) == 0:
                result["status"] = "no_valid_numbers"
                return result

            if len(clean_data) <= 1:
                return {
                    "analysis": "numeric",
                    "count": len(clean_data),
                    "value": (
                        float(clean_data.iloc[0]) if len(clean_data) == 1
                        else None
                    ),
                    "status": "insufficient_data_for_analysis",
                }

            result["valid_numbers"] = len(clean_data)
            result["invalid_numbers"] = len(series) - len(clean_data)
            result.update(self._build_numeric_stats(clean_data))

            # Pattern hints for downstream consumers.
            unique_count = self.analyzer.safe_execute(
                lambda: clean_data.nunique(), default=0
            )
            if unique_count and unique_count == len(clean_data):
                result["pattern"] = "likely_id"
            elif unique_count and unique_count <= 20:
                result["pattern"] = "likely_categorical"
                try:
                    counts = clean_data.value_counts().head(10)
                    result["value_counts"] = {
                        str(k): int(v) for k, v in counts.items()
                    }
                except Exception:
                    pass

            return result

        except Exception as exc:
            result["error"] = str(exc)
            return result

    def _analyze_boolean_safe(self, series: pd.Series) -> dict:
        """
        Analyze a boolean column.

        Counts true-like and false-like values using canonical string sets
        so the column is handled correctly regardless of case or format
        (e.g. 'Yes', 'YES', 'y', '1', 'True' are all treated as true).

        Parameters
        ----------
        series : pd.Series
            Raw boolean strings.

        Returns
        -------
        dict
            Profile with true_count, false_count, other_count,
            true_percentage.
        """
        result = {"analysis": "boolean"}
        try:
            clean = series.dropna()
            if len(clean) == 0:
                return result

            str_data = clean.astype(str).str.lower().str.strip()
            true_values = {
                "true", "yes", "y", "1", "on", "enabled", "active", "1.0"
            }
            false_values = {
                "false", "no", "n", "0", "off", "disabled", "inactive", "0.0"
            }

            true_count = sum(1 for v in str_data if v in true_values)
            false_count = sum(1 for v in str_data if v in false_values)
            total = len(str_data)

            result.update({
                "true_count": true_count,
                "false_count": false_count,
                "other_count": total - true_count - false_count,
                "true_percentage": (
                    round(true_count / total * 100, 1) if total > 0 else 0
                ),
            })
            return result

        except Exception as exc:
            result["error"] = str(exc)
            return result

    def _analyze_datetime_safe(self, series: pd.Series) -> dict:
        """
        Analyze a datetime column.

        Uses pd.to_datetime with errors='coerce' to parse values and
        reports the valid/invalid split, the date range, and a sample
        parsed timestamp for format reference.

        Parameters
        ----------
        series : pd.Series
            Raw datetime strings.

        Returns
        -------
        dict
            Profile with valid_dates, invalid_dates, success_rate,
            date_range, sample_format.
        """
        result = {"analysis": "datetime"}
        try:
            clean = series.dropna().astype(str)
            if len(clean) == 0:
                return result

            parsed = pd.to_datetime(clean, errors="coerce")
            valid = parsed.dropna()

            result.update({
                "valid_dates": len(valid),
                "invalid_dates": len(clean) - len(valid),
                "success_rate": (
                    round(len(valid) / len(clean) * 100, 1)
                    if len(clean) > 0 else 0
                ),
            })

            if len(valid) > 0:
                result.update({
                    "date_range": [str(valid.min()), str(valid.max())],
                    "sample_format": str(valid.iloc[0]),
                })

            return result

        except Exception as exc:
            result["error"] = str(exc)
            return result

    def _analyze_categorical_safe(self, series: pd.Series) -> dict:
        """
        Analyze a categorical column.

        Reports the number of distinct categories, the top 20 by
        frequency, and the percentage share of the most common value.

        Parameters
        ----------
        series : pd.Series
            Raw categorical strings.

        Returns
        -------
        dict
            Profile with categories, values (top 20),
            most_common_percentage.
        """
        result = {"analysis": "categorical"}
        try:
            clean = series.dropna()
            if len(clean) == 0:
                return result

            try:
                counts = clean.value_counts()
                result["categories"] = len(counts)
                result["values"] = {
                    str(k): int(v) for k, v in counts.head(20).items()
                }
                if len(counts) > 0:
                    result["most_common_percentage"] = round(
                        counts.iloc[0] / len(clean) * 100, 1
                    )
            except Exception:
                # Fallback if value_counts fails (e.g. unhashable types).
                result["categories"] = len(set(clean.astype(str)))

            return result

        except Exception as exc:
            result["error"] = str(exc)
            return result

    def _analyze_text_safe(self, series: pd.Series) -> dict:
        """
        Analyze a free-text column.

        Computes string length statistics and flags the presence of
        numeric characters and special characters, which are useful
        signals for downstream data cleaning decisions.

        Parameters
        ----------
        series : pd.Series
            Raw text strings.

        Returns
        -------
        dict
            Profile with avg_length, max_length, min_length,
            contains_numbers, contains_special_chars.
        """
        result = {"analysis": "text"}
        try:
            clean = series.dropna().astype(str)
            if len(clean) == 0:
                return result

            lengths = clean.str.len()
            result.update({
                "avg_length": (
                    round(lengths.mean(), 1) if len(lengths) > 0 else 0
                ),
                "max_length": int(lengths.max()) if len(lengths) > 0 else 0,
                "min_length": int(lengths.min()) if len(lengths) > 0 else 0,
                "contains_numbers": int(
                    clean.str.contains(r"\d", na=False).sum()
                ),
                "contains_special_chars": int(
                    clean.str.contains(r"[^a-zA-Z0-9\s]", na=False).sum()
                ),
            })
            return result

        except Exception as exc:
            result["error"] = str(exc)
            return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_safe_samples(self, series: pd.Series, n: int = 3) -> list:
        """
        Return up to *n* representative non-null sample values.

        Samples are drawn from the start, middle, and end of the series
        to capture potential variability. Values longer than 100
        characters are truncated for readability.

        Parameters
        ----------
        series : pd.Series
            Any column series.
        n : int, optional
            Maximum number of samples to return. Default: 3.

        Returns
        -------
        list[str]
            Up to *n* stringified sample values.
        """
        try:
            non_null = series.dropna()
            if len(non_null) == 0:
                return []

            if len(non_null) <= n:
                raw_samples = non_null.tolist()
            else:
                indices = [0, len(non_null) // 2, len(non_null) - 1]
                raw_samples = [non_null.iloc[i] for i in indices[:n]]

            safe = []
            for sample in raw_samples:
                try:
                    s = str(sample)
                    safe.append(s[:97] + "..." if len(s) > 100 else s)
                except Exception:
                    safe.append("<unprintable>")

            return safe[:n]

        except Exception:
            return []

    def _get_file_metadata_safe(
        self, file_path: str, df: pd.DataFrame
    ) -> dict:
        """
        Collect file system and in-memory metadata.

        Parameters
        ----------
        file_path : str
            Path to the source CSV.
        df : pd.DataFrame
            The loaded dataframe (used for memory usage calculation).

        Returns
        -------
        dict
            File size (bytes and MB), modification time, memory usage
            (MB), and total cell count.
        """
        metadata = {}

        try:
            stat = os.stat(file_path)
            metadata.update({
                "file_size_bytes": stat.st_size,
                "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_time": stat.st_mtime,
            })
        except OSError:
            pass

        try:
            metadata.update({
                "memory_usage_mb": round(
                    df.memory_usage(deep=True).sum() / (1024 * 1024), 2
                ),
                "total_cells": len(df) * len(df.columns),
            })
        except Exception:
            pass

        return metadata


# ===========================================================================
# GUI
# ===========================================================================

class RobustProfilerGUI:
    """
    Tkinter GUI for the CSV profiler.

    Provides file/output selection, option checkboxes, a progress bar,
    and a scrollable log pane. Analysis is performed in a background
    daemon thread; results are passed back to the GUI thread via a
    queue.Queue to avoid Tkinter thread-safety issues.
    """

    def __init__(self, master: tk.Tk) -> None:
        """
        Initialize the GUI and bind it to *master*.

        Parameters
        ----------
        master : tk.Tk
            The root Tkinter window.
        """
        self.master = master
        self.profiler = UltraRobustCSVProfiler()
        self.result_queue: queue.Queue = queue.Queue()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build and layout all widgets."""
        self.master.title("Robust CSV Profiler")
        self.master.geometry("800x500")
        self.master.resizable(True, True)

        main = ttk.Frame(self.master, padding="15")
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(1, weight=1)

        # ---- File selection row -------------------------------------------
        ttk.Label(main, text="CSV File:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        self.file_var = tk.StringVar()
        ttk.Entry(
            main, textvariable=self.file_var, state="readonly"
        ).grid(row=0, column=1, sticky=tk.EW, padx=(0, 10), pady=5)
        ttk.Button(
            main, text="Browse...", command=self._browse_file
        ).grid(row=0, column=2, pady=5)

        # ---- Output selection row -----------------------------------------
        ttk.Label(main, text="Save to:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        self.output_var = tk.StringVar()
        ttk.Entry(
            main, textvariable=self.output_var, state="readonly"
        ).grid(row=1, column=1, sticky=tk.EW, padx=(0, 10), pady=5)
        self.save_btn = ttk.Button(
            main, text="Save As...", command=self._save_as, state="disabled"
        )
        self.save_btn.grid(row=1, column=2, pady=5)

        # ---- Options frame ------------------------------------------------
        opts = ttk.LabelFrame(main, text="Options", padding="10")
        opts.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=10)
        self.include_samples_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts,
            text="Include sample values",
            variable=self.include_samples_var,
        ).pack(side=tk.LEFT, padx=10)
        self.detailed_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts,
            text="Detailed statistical analysis",
            variable=self.detailed_analysis_var,
        ).pack(side=tk.LEFT, padx=10)

        # ---- Analyze button and progress bar ------------------------------
        self.analyze_btn = ttk.Button(
            main,
            text="Analyze CSV",
            command=self._start_analysis,
            state="disabled",
        )
        self.analyze_btn.grid(row=3, column=0, columnspan=3, pady=(15, 10))
        self.progress = ttk.Progressbar(main, mode="indeterminate")

        # ---- Status / log frame -------------------------------------------
        status_frame = ttk.LabelFrame(main, text="Status", padding="10")
        status_frame.grid(
            row=4, column=0, columnspan=3, sticky=tk.EW, pady=10
        )
        status_frame.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(
            value="Select a CSV file to begin analysis"
        )
        ttk.Label(status_frame, textvariable=self.status_var).grid(
            row=0, column=0, sticky=tk.W
        )

        self.log_text = tk.Text(status_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(
            status_frame, orient=tk.VERTICAL, command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=1, column=0, sticky=tk.EW, pady=(5, 0))
        scrollbar.grid(row=1, column=1, sticky=tk.NS, pady=(5, 0))

        # trace_add() replaces the deprecated trace() for Tcl 9 / Python 3.14+
        self.file_var.trace_add("write", self._update_buttons)
        self.output_var.trace_add("write", self._update_buttons)

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------

    def _browse_file(self) -> None:
        """Open a file dialog to select the input CSV."""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[
                ("CSV, TSV, Text", "*.csv *.tsv *.txt"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.file_var.set(filename)
            self.output_var.set(str(Path(filename).with_suffix(".json")))
            self.status_var.set("File selected — ready to analyze")
            self._log(f"Selected: {os.path.basename(filename)}")

    def _save_as(self) -> None:
        """Open a save dialog to choose the output JSON path."""
        filename = filedialog.asksaveasfilename(
            title="Save profile as",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.output_var.set(filename)

    def _update_buttons(self, *_args) -> None:
        """Enable/disable buttons based on whether paths are set."""
        if self.file_var.get():
            self.save_btn.config(state="normal")
            if self.output_var.get():
                self.analyze_btn.config(state="normal")
                self.status_var.set("Ready to analyze")
            else:
                self.analyze_btn.config(state="disabled")
                self.status_var.set("Choose output location")
        else:
            self.save_btn.config(state="disabled")
            self.analyze_btn.config(state="disabled")
            self.status_var.set("Select a CSV file to begin analysis")

    def _log(self, message: str) -> None:
        """Append *message* to the log pane and scroll to the bottom."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.master.update_idletasks()

    # ------------------------------------------------------------------
    # Analysis lifecycle
    # ------------------------------------------------------------------

    def _start_analysis(self) -> None:
        """Disable the button, start the progress bar, launch the thread."""
        self.analyze_btn.config(state="disabled")
        self.log_text.delete(1.0, tk.END)
        self.progress.grid(
            row=5, column=0, columnspan=3, sticky=tk.EW, pady=(5, 0)
        )
        self.progress.start(10)
        self.status_var.set("Analyzing CSV file...")
        self._log("Starting analysis...")

        threading.Thread(
            target=self._run_analysis,
            args=(self.file_var.get(), self.output_var.get()),
            daemon=True,
        ).start()

        # Poll the result queue every 100 ms on the main thread.
        self.master.after(100, self._check_results)

    def _run_analysis(self, input_path: str, output_path: str) -> None:
        """
        Worker thread: run the profiler, apply options, write JSON.

        Results (or errors) are placed on self.result_queue for the
        main thread to consume.
        """
        try:
            profile = self.profiler.profile(input_path)

            # Apply GUI options before saving.
            if not self.include_samples_var.get():
                for col in profile.get("columns", []):
                    col.pop("samples", None)

            if not self.detailed_analysis_var.get():
                for col in profile.get("columns", []):
                    if "statistics" in col:
                        col["statistics"] = {
                            k: v
                            for k, v in col["statistics"].items()
                            if k in ("min", "max", "mean", "count")
                        }
                    col.pop("quartiles", None)

            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(profile, fh, indent=2, ensure_ascii=False)

            self.result_queue.put(("SUCCESS", output_path, profile))

        except Exception as exc:
            self.result_queue.put((
                "ERROR",
                f"Critical analysis failure: {exc}",
                traceback.format_exc(),
            ))

    def _check_results(self) -> None:
        """Poll the result queue; reschedule if nothing is ready yet."""
        try:
            result = self.result_queue.get_nowait()
            self._handle_result(result)
        except queue.Empty:
            self.master.after(100, self._check_results)

    def _handle_result(self, result: tuple) -> None:
        """Process the worker result and update the GUI accordingly."""
        self.progress.stop()
        self.progress.grid_remove()
        self.analyze_btn.config(state="normal")

        status, payload1, payload2 = result

        if status == "SUCCESS":
            output_path, profile = payload1, payload2
            shape = profile.get("shape", {})
            warn_list = profile.get("warnings", [])
            info_list = profile.get("info", [])

            self.status_var.set("Analysis completed successfully!")
            self._log(
                f"Analysis completed! Status: "
                f"{profile.get('status', 'unknown')}"
            )

            if info_list:
                self._log("\nInfo:")
                for msg in info_list:
                    self._log(f"  i {msg}")

            if warn_list:
                self._log("\nWarnings:")
                for w in warn_list:
                    self._log(f"  ! {w}")

            # Summarize detected column types.
            columns = profile.get("columns", [])
            type_counts: dict = {}
            error_count = sum(
                1 for c in columns if c.get("type") == "error"
            )
            for col in columns:
                t = col.get("type", "unknown")
                if t != "error":
                    type_counts[t] = type_counts.get(t, 0) + 1

            self._log("\nColumn types detected:")
            for t, count in sorted(type_counts.items()):
                self._log(f"  - {t}: {count}")
            if error_count:
                self._log(f"  - errors: {error_count}")

            messagebox.showinfo(
                "Analysis Complete",
                (
                    f"Analysis completed successfully!\n\n"
                    f"File: {profile.get('file', 'N/A')}\n"
                    f"Size: {shape.get('rows', 0):,} rows x "
                    f"{shape.get('columns', 0)} columns\n"
                    f"Saved to: {os.path.basename(output_path)}\n"
                    f"Warnings: {len(warn_list)}"
                ),
            )

        elif status == "ERROR":
            error_msg, tb = payload1, payload2
            self.status_var.set("Analysis failed")
            self._log(f"ERROR: {error_msg}")
            if tb:
                self._log("Full error details:")
                self._log(tb)
            messagebox.showerror(
                "Analysis Failed",
                f"Analysis failed:\n\n{error_msg}\n\n"
                f"Check the log for details.",
            )


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> int:
    """
    Entry point for both GUI and CLI modes.

    If command-line arguments are provided the script runs headlessly and
    writes a JSON file. Otherwise the Tkinter GUI is launched.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    if len(sys.argv) > 1:
        # ---- CLI mode ------------------------------------------------------
        parser = argparse.ArgumentParser(
            description=(
                "A highly robust CSV profiler that handles a wide range "
                "of data formats and encodings."
            ),
            epilog=(
                "Designed to process most CSV files gracefully, with "
                "extensive error handling to prevent crashes."
            ),
        )
        parser.add_argument(
            "csv_file", help="Path to the CSV file to analyze."
        )
        parser.add_argument(
            "-o", "--output",
            help="Output JSON file path (default: <input>.json).",
        )
        parser.add_argument(
            "--no-samples",
            action="store_true",
            help="Exclude sample values from the output.",
        )
        parser.add_argument(
            "--simple",
            action="store_true",
            help="Simplified analysis — faster, less detail.",
        )
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Show progress messages and warnings.",
        )
        args = parser.parse_args()

        if not os.path.exists(args.csv_file):
            print(f"Error: File not found at '{args.csv_file}'")
            return 1

        if args.verbose:
            print(f"Analyzing: {args.csv_file}")

        profiler = UltraRobustCSVProfiler()
        profile = profiler.profile(args.csv_file)

        if profile.get("status") == "error":
            print(f"Error: {profile.get('error', 'Unknown error')}")
            if args.verbose and "traceback" in profile:
                print(profile["traceback"])
            return 1

        # Apply CLI flags.
        if args.no_samples:
            for col in profile.get("columns", []):
                col.pop("samples", None)
        if args.simple:
            for col in profile.get("columns", []):
                if "statistics" in col:
                    col["statistics"] = {
                        k: v
                        for k, v in col["statistics"].items()
                        if k in ("min", "max", "mean", "count")
                    }
                col.pop("quartiles", None)

        output_file = args.output or str(
            Path(args.csv_file).with_suffix(".json")
        )

        try:
            with open(output_file, "w", encoding="utf-8") as fh:
                json.dump(profile, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            print(f"Error saving to {output_file}: {exc}")
            return 1

        shape = profile.get("shape", {})
        warn_list = profile.get("warnings", [])
        info_list = profile.get("info", [])

        print("\nAnalysis completed successfully!")
        print(
            f"Shape: {shape.get('rows', 0):,} rows x "
            f"{shape.get('columns', 0)} columns"
        )
        print(f"Profile saved to: {output_file}")

        if args.verbose:
            if info_list:
                print("Info:")
                for msg in info_list:
                    print(f"   i {msg}")
            if warn_list:
                print(f"Warnings ({len(warn_list)}):")
                for w in warn_list[:5]:
                    print(f"   ! {w}")
                if len(warn_list) > 5:
                    print(f"   ... and {len(warn_list) - 5} more")

            columns = profile.get("columns", [])
            type_counts: dict = {}
            error_count = sum(
                1 for c in columns if c.get("type") == "error"
            )
            for col in columns:
                t = col.get("type", "unknown")
                if t != "error":
                    type_counts[t] = type_counts.get(t, 0) + 1

            print("Column Types Detected:")
            for t, count in sorted(type_counts.items()):
                print(f"   - {t}: {count}")
            if error_count:
                print(f"   - errors: {error_count}")

        return 0

    else:
        # ---- GUI mode ------------------------------------------------------
        try:
            root = tk.Tk()
            RobustProfilerGUI(root)
            root.update_idletasks()
            x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
            y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
            root.geometry(f"+{x}+{y}")
            root.title("Robust CSV Profiler - Resilient Data Analysis")
            root.mainloop()
            return 0
        except Exception as exc:
            print(f"GUI failed to start: {exc}")
            print("Try CLI mode instead:")
            print(f"  python {sys.argv[0]} your_file.csv")
            return 1


if __name__ == "__main__":
    sys.exit(main())