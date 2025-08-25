#!/usr/bin/env python3
"""
AI CSV Profiler

This script provides a powerful and resilient tool for analyzing the structure and
content of CSV files. Its core design philosophy is resilience and graceful error
handling. It is built to handle malformed, messy, and unpredictable data
without terminating unexpectedly.

The script operates in two modes:
1.  **GUI Mode:** A user-friendly Tkinter interface for interactive use.
2.  **CLI Mode:** A command-line interface for scripting and automation.

Key Features:
-   **Defensive Analysis:** Every operation is wrapped in extensive error handlers.
-   **Smart CSV Reading:** Automatically attempts to detect separators and encodings.
-   **Accurate Type Inference:** Employs a sophisticated, heuristic-based
    algorithm to identify numeric, boolean, datetime, categorical, and text data.
-   **Comprehensive Profiling:** Generates a detailed JSON report including file
    metadata, column statistics, and data quality warnings.
"""

# --- Standard Library Imports ---
import argparse
import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

# --- GUI Library Import ---
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# --- Data Handling Library Imports ---
import pandas as pd
import numpy as np
import warnings

# --- Global Warning Suppression ---
# These warnings are suppressed to provide a cleaner user experience, as the
# script is designed to handle these issues internally.
# DtypeWarning: Occurs when pandas infers mixed types in a column. We handle this
#               by initially reading all columns as strings.
# RuntimeWarning: Can occur during statistical calculations with invalid values
#                 (e.g., infinity). Our safe analyzers handle these cases.
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class DefensiveAnalyzer:
    """
    A utility class with static methods for executing operations with extensive
    error handling, aiming for fail-safe execution. It provides tools for
    data manipulation that are designed to be highly resilient to unexpected inputs.
    """

    @staticmethod
    def safe_execute(func, *args, default=None, **kwargs):
        """
        Executes a given function within a try-except block, returning a default
        value if any exception occurs. This is the foundational safety wrapper.

        Args:
            func (callable): The function to execute.
            *args: Positional arguments for the function.
            default: The value to return on any exception.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function or the default value on error.
        """
        try:
            result = func(*args, **kwargs)
            # Additional check for non-finite numbers which can cause issues later.
            if isinstance(result, (int, float)):
                if pd.isna(result) or np.isinf(result):
                    return default
            return result
        except Exception:
            # If anything goes wrong, we return the safe default value.
            return default

    @staticmethod
    def safe_convert_numeric(series, fill_value=np.nan):
        """
        Converts a pandas Series to a numeric type with maximum safety. It handles
        various non-standard null values and avoids misinterpreting ID columns
        as numeric data.

        Args:
            series (pd.Series): The series to convert.
            fill_value: The value to use for elements that cannot be converted.

        Returns:
            pd.Series: The converted numeric series, or the original on failure.
        """
        try:
            # Handle object types, which are likely strings.
            if series.dtype == 'object':
                # First, clean up common non-numeric placeholders and whitespace.
                cleaned = series.astype(str).str.strip()
                null_placeholders = [
                    '', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', '#N/A',
                    '#NULL!', '#DIV/0!', '#REF!', '#NAME?', '#NUM!', '#VALUE!',
                    'inf', '-inf', 'infinity', '-infinity'
                ]
                cleaned = cleaned.replace(null_placeholders, np.nan)
                result = pd.to_numeric(cleaned, errors='coerce')
            else:
                # For non-object types, a direct conversion is usually safe.
                result = pd.to_numeric(series, errors='coerce')

            # Ensure infinite values are treated as NaN.
            result = result.replace([np.inf, -np.inf], np.nan)

            # Heuristic: If less than 10% of values are valid numbers, it's likely
            # not a numeric column (e.g., IDs with some numbers). Revert to original.
            if result.notna().sum() < len(result) * 0.1:
                return series.fillna(fill_value)

            return result.fillna(fill_value)

        except Exception:
            # Fallback for any unexpected error during conversion.
            return series.fillna(fill_value)

    @staticmethod
    def safe_statistics(data, stat_name):
        """
        Computes a single statistical measure on a dataset with high safety.
        It handles empty data and removes extreme outliers for mean and std dev.

        Args:
            data (pd.Series): The data to analyze (pre-filtered for NaNs).
            stat_name (str): The name of the statistic to compute (e.g., 'mean').

        Returns:
            float or int: The computed statistic, or None on failure.
        """
        try:
            # Cannot compute stats on empty data.
            if len(data) == 0:
                return None

            clean_data = data.dropna()
            if len(clean_data) == 0:
                return None

            # For mean and standard deviation, remove extreme outliers (top/bottom 1%)
            # to get a more representative measure of the central tendency and spread.
            if stat_name in ['mean', 'std']:
                q99 = clean_data.quantile(0.99)
                q01 = clean_data.quantile(0.01)
                if pd.notna(q99) and pd.notna(q01):
                    clean_data = clean_data[(clean_data >= q01) & (clean_data <= q99)]

            if len(clean_data) == 0:
                return None  # All data was filtered out as outliers.

            # Dispatch to the correct statistical function.
            if stat_name == 'min':
                return float(clean_data.min())
            elif stat_name == 'max':
                return float(clean_data.max())
            elif stat_name == 'mean':
                return float(clean_data.mean())
            elif stat_name == 'median':
                return float(clean_data.median())
            elif stat_name == 'std':
                # Standard deviation requires at least 2 data points.
                return float(clean_data.std()) if len(clean_data) > 1 else 0.0
            elif stat_name == 'count':
                return len(clean_data)
            elif stat_name.startswith('quantile_'):
                q = float(stat_name.split('_')[1])
                return float(clean_data.quantile(q))
            else:
                return None

        except Exception:
            return None


class UltraRobustCSVProfiler:
    """
    The core engine of the application. This class orchestrates the entire
    profiling process, from reading the CSV with multiple fallback strategies
    to analyzing each column individually and compiling the final JSON report.
    """

    def __init__(self):
        """Initializes the profiler with common settings."""
        self.analyzer = DefensiveAnalyzer()
        # A list of common text encodings to attempt when reading a file.
        self.encoding_attempts = ['utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1',
                                  'latin1', 'ascii', 'utf-16', 'utf-32']
        # A list of common column separators to attempt.
        self.separator_attempts = [',', ';', '\t', '|', ':', ' ']

    def profile(self, file_path: str) -> dict:
        """
        Generates a comprehensive and safe profile of a CSV file. This is the
        main public method of the class.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            dict: A JSON-serializable dictionary containing the profile.
        """
        # Start with a base profile structure to ensure consistency, even on failure.
        base_profile = {
            "file": os.path.basename(file_path),
            "status": "unknown",
            "shape": {"rows": 0, "columns": 0},
            "columns": [],
            "warnings": [],
            "metadata": {}
        }

        try:
            # Step 1: Read the CSV using a highly resilient method.
            df, read_warnings = self._read_csv_ultra_safe(file_path)
            base_profile["warnings"].extend(read_warnings)

            # If the file is unreadable or empty, return a corresponding status.
            if df is None or df.empty:
                base_profile["status"] = "empty_or_unreadable"
                base_profile["error"] = "Could not read file or file is empty"
                return base_profile

            # Step 2: If reading succeeded, populate the profile.
            base_profile["shape"] = {"rows": len(df), "columns": len(df.columns)}
            base_profile["status"] = "processed"
            base_profile["columns"] = self._analyze_all_columns_safe(df)
            base_profile["metadata"] = self._get_file_metadata_safe(file_path, df)

            return base_profile

        except Exception as e:
            # A top-level catch-all for any unexpected critical failure.
            base_profile["status"] = "error"
            base_profile["error"] = f"Critical error: {str(e)}"
            base_profile["traceback"] = traceback.format_exc()
            return base_profile

    def _read_csv_ultra_safe(self, file_path: str) -> tuple:
        """
        Reads a CSV file using a multi-stage fallback strategy to handle
        various encodings, separators, and file corruptions.

        Returns:
            tuple: A pandas DataFrame (or None) and a list of warnings.
        """
        warnings_list = []

        # Pre-check: Check file size to warn about large files.
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return None, ["File is empty"]
            if file_size > 500 * 1024 * 1024:  # Warn for files > 500MB
                warnings_list.append(f"Large file ({file_size / 1024 / 1024:.1f}MB)")
        except Exception:
            warnings_list.append("Could not determine file size")

        # Strategy 1: Iterate through common encodings and separators.
        # This is the most reliable method for well-formed but non-standard CSVs.
        for encoding in self.encoding_attempts:
            for separator in self.separator_attempts:
                try:
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        sep=separator,
                        on_bad_lines='skip',  # Skip rows with errors.
                        low_memory=False,  # Helps with mixed types.
                        dtype=str,  # Read everything as string first.
                        na_values=[''],  # Treat empty strings as NaN initially.
                        keep_default_na=False,
                        engine='python'  # Slower but more feature-complete.
                    )
                    # A successful read should have more than one column.
                    if len(df.columns) > 1 and len(df) > 0:
                        warnings_list.append(f"Successfully read with {encoding} and '{separator}' separator")
                        return df, warnings_list
                except Exception:
                    continue  # Try the next combination.

        # Strategy 2: Fallback using pandas' automatic separator detection.
        try:
            df = pd.read_csv(file_path, encoding='utf-8', sep=None, engine='python', on_bad_lines='skip', dtype=str)
            warnings_list.append("Used fallback reading method (auto-separator)")
            return df, warnings_list
        except Exception:
            pass

        # Strategy 3: Last resort - manual text parsing.
        # This handles cases where pandas fails completely, such as with
        # severe corruption or unusual line terminators.
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:1000]  # Read a sample of the file.

            if not lines:
                return None, ["File appears empty or unreadable"]

            # Guess the separator by finding the most frequent candidate in the header.
            first_line = lines[0].strip()
            best_sep = max(self.separator_attempts, key=lambda sep: first_line.count(sep))

            # Manually construct a DataFrame.
            data = []
            headers = first_line.split(best_sep)
            for line in lines[1:]:
                row = line.strip().split(best_sep)
                # Pad rows that are shorter than the header.
                while len(row) < len(headers):
                    row.append('')
                row = row[:len(headers)]  # Truncate rows longer than the header.
                data.append(row)

            df = pd.DataFrame(data, columns=headers)
            warnings_list.append("Used manual text parsing as last resort")
            return df, warnings_list
        except Exception:
            return None, warnings_list + ["All reading methods failed"]

    def _analyze_all_columns_safe(self, df: pd.DataFrame) -> list:
        """
        Analyzes all columns in a DataFrame, isolating errors so that a
        problem in one column does not prevent others from being analyzed.

        Returns:
            list: A list of profile dictionaries, one for each column.
        """
        columns = []
        for i, col_name in enumerate(df.columns):
            try:
                # Analyze each column individually.
                col_info = self._analyze_single_column_safe(df[col_name], str(col_name))
                col_info["position"] = i
                columns.append(col_info)
            except Exception as e:
                # If analysis for a column fails, append an error entry for it.
                columns.append({
                    "name": str(col_name),
                    "position": i,
                    "type": "error",
                    "status": "analysis_failed",
                    "error": str(e),
                    "samples": []
                })
        return columns

    def _analyze_single_column_safe(self, series: pd.Series, col_name: str) -> dict:
        """
        Analyzes a single column (Series), calculating common metrics and then
        dispatching to a type-specific analysis function.

        Returns:
            dict: A profile dictionary for the single column.
        """
        base_info = {
            "name": col_name,
            "type": "unknown",
            "status": "analyzing",
            "total_values": len(series),
            "samples": []
        }
        try:
            # Basic stats are calculated for all columns.
            base_info["samples"] = self._get_safe_samples(series)
            base_info["missing"] = self.analyzer.safe_execute(lambda: int(series.isnull().sum()), default=0)
            base_info["empty_strings"] = self.analyzer.safe_execute(lambda: int((series.astype(str) == '').sum()),
                                                                    default=0)
            base_info["unique"] = self.analyzer.safe_execute(lambda: int(series.nunique()), default=0)

            # Core logic: Detect the column type and then perform specific analysis.
            detected_type = self._detect_column_type_safe(series)
            base_info["type"] = detected_type

            if detected_type == "numeric":
                base_info.update(self._analyze_numeric_ultra_safe(series))
            elif detected_type == "boolean":
                base_info.update(self._analyze_boolean_safe(series))
            elif detected_type == "datetime":
                base_info.update(self._analyze_datetime_safe(series))
            elif detected_type == "categorical":
                base_info.update(self._analyze_categorical_safe(series))
            else:  # Includes 'text', 'empty', and 'unknown'
                base_info.update(self._analyze_text_safe(series))

            base_info["status"] = "completed"
            return base_info
        except Exception as e:
            # If anything goes wrong, populate the dict with error info.
            base_info["status"] = "error"
            base_info["error"] = str(e)
            base_info["type"] = "error"
            return base_info

    def _detect_column_type_safe(self, series: pd.Series) -> str:
        """
        Determines the data type of a column using a series of heuristics.
        The order of these checks is crucial for accuracy.

        Returns:
            str: The detected type (e.g., 'numeric', 'boolean', 'datetime').
        """
        try:
            # Start with a clean series, dropping nulls and empty strings.
            clean_series = series.dropna()
            if len(clean_series) == 0:
                return "empty"

            str_values = clean_series.astype(str).str.strip()
            str_values = str_values[str_values != '']
            if len(str_values) == 0:
                return "empty"

            unique_str_lower = set(str_values.str.lower().unique())

            # Heuristic 1: Check for boolean patterns FIRST.
            # This is critical because booleans (e.g., 1/0) can be misidentified as numeric.
            boolean_patterns = [{'true', 'false'}, {'yes', 'no'}, {'y', 'n'}, {'1', '0'},
                                {'on', 'off'}, {'enabled', 'disabled'}, {'active', 'inactive'}]
            for pattern in boolean_patterns:
                if len(unique_str_lower) <= 2 and unique_str_lower.issubset(pattern):
                    return "boolean"

            # Heuristic 2: Check for datetime hints in the column name.
            # If the name suggests a date, attempt a datetime check early.
            col_name = str(series.name).lower() if series.name else ""
            if any(keyword in col_name for keyword in ['date', 'dt', 'time', 'start', 'end']):
                if self._could_be_datetime(str_values):
                    return "datetime"

            # Heuristic 3: Check for ID hints in the column name.
            # This prevents columns like 'product_id' from being classified as numeric.
            if any(keyword in col_name for keyword in ['id', 'response', 'uuid', 'key']):
                return "text"

            # Heuristic 4: Attempt numeric conversion.
            # Convert to numeric and check if a high percentage of values succeed.
            try:
                numeric_series = pd.to_numeric(str_values, errors='coerce')
                numeric_ratio = numeric_series.notna().sum() / len(str_values)
                if numeric_ratio > 0.9:  # Threshold for considering it numeric.
                    # A final check in case a numeric column is actually a 0/1 boolean.
                    unique_numeric = set(numeric_series.dropna().unique())
                    if len(unique_numeric) <= 2 and unique_numeric.issubset({0, 1, 0.0, 1.0}):
                        return "boolean"
                    return "numeric"
            except Exception:
                pass

            # Heuristic 5: Perform a more general datetime check if other checks failed.
            if self._could_be_datetime(str_values):
                return "datetime"

            # Heuristic 6: Differentiate between categorical and text.
            # Low cardinality (few unique values) suggests categorical data.
            unique_count = len(unique_str_lower)
            total_count = len(str_values)
            if unique_count <= min(50, total_count * 0.5):  # e.g., <= 50 unique values or < 50% unique
                return "categorical"

            # Default: If none of the above, classify as text.
            return "text"

        except Exception:
            return "text"  # Fallback type

    def _could_be_datetime(self, str_series) -> bool:
        """
        Helper function to check if a series of strings could be datetime objects.
        It samples the data rather than converting the whole series for performance.
        """
        try:
            # Take a small sample to test for datetime format.
            sample = str_series.head(10)
            success_count = 0
            for val in sample:
                val_str = str(val).strip()
                if len(val_str) < 8: continue  # Unlikely to be a date.

                # Quick check for common date/time separators.
                datetime_indicators = ['-', '/', ':', 'T', ' ']
                if any(indicator in val_str for indicator in datetime_indicators):
                    try:
                        # Attempt to parse the value.
                        if pd.notna(pd.to_datetime(val_str, errors='raise')):
                            success_count += 1
                    except Exception:
                        continue
            # If a high percentage of the sample parses correctly, assume it's datetime.
            return success_count >= len(sample) * 0.7
        except Exception:
            return False

    def _analyze_numeric_ultra_safe(self, series: pd.Series) -> dict:
        """Performs detailed, highly safe analysis for numeric columns."""
        result = {"analysis": "numeric"}
        try:
            # Use the safe converter to get numeric data.
            numeric_data = self.analyzer.safe_convert_numeric(series)
            clean_data = numeric_data.dropna()

            if len(clean_data) == 0:
                result["status"] = "no_valid_numbers"
                return result

            # Special case for columns with just one valid number.
            if len(clean_data) <= 1:
                return {"count": len(clean_data), "value": float(clean_data.iloc[0]) if len(clean_data) == 1 else None,
                        "status": "insufficient_data_for_analysis"}

            result["valid_numbers"] = len(clean_data)
            result["invalid_numbers"] = len(series) - len(clean_data)

            # Calculate standard statistics using the safe method.
            stats = {}
            for stat_name in ['min', 'max', 'mean', 'median', 'std']:
                stat_value = self.analyzer.safe_statistics(clean_data, stat_name)
                if stat_value is not None:
                    stats[stat_name] = round(stat_value, 4) if isinstance(stat_value, float) else stat_value
            result["statistics"] = stats

            # Calculate quartiles.
            quartiles = {}
            for q, name in [(0.25, "q25"), (0.75, "q75")]:
                q_val = self.analyzer.safe_statistics(clean_data, f"quantile_{q}")
                if q_val is not None: quartiles[name] = round(q_val, 4)
            result["quartiles"] = quartiles

            # Additional numeric-specific metrics.
            result["zero_count"] = self.analyzer.safe_execute(lambda: int((clean_data == 0).sum()), default=0)
            result["negative_count"] = self.analyzer.safe_execute(lambda: int((clean_data < 0).sum()), default=0)

            # Check for patterns like IDs (all unique) or categories (few unique).
            unique_count = self.analyzer.safe_execute(lambda: clean_data.nunique(), default=0)
            if unique_count > 0:
                if unique_count == len(clean_data):
                    result["pattern"] = "likely_id"
                elif unique_count <= 20:
                    result["pattern"] = "likely_categorical"
                    try:  # Safely get value counts for likely categorical data.
                        value_counts = clean_data.value_counts().head(10)
                        result["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
                    except Exception:
                        pass
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    def _analyze_boolean_safe(self, series: pd.Series) -> dict:
        """Performs detailed, safe analysis for boolean columns."""
        result = {"analysis": "boolean"}
        try:
            clean_data = series.dropna()
            if len(clean_data) == 0: return result

            # Convert to lowercased string to match against patterns.
            str_data = clean_data.astype(str).str.lower().str.strip()
            true_patterns = {'true', 'yes', 'y', '1', 'on', 'enabled', 'active', '1.0'}
            false_patterns = {'false', 'no', 'n', '0', 'off', 'disabled', 'inactive', '0.0'}

            true_count = sum(1 for val in str_data if val in true_patterns)
            false_count = sum(1 for val in str_data if val in false_patterns)

            result.update({
                "true_count": true_count,
                "false_count": false_count,
                "other_count": len(str_data) - true_count - false_count,
                "true_percentage": round(true_count / len(str_data) * 100, 1) if len(str_data) > 0 else 0
            })
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    def _analyze_datetime_safe(self, series: pd.Series) -> dict:
        """Performs detailed, safe analysis for datetime columns."""
        result = {"analysis": "datetime"}
        try:
            clean_data = series.dropna().astype(str)
            if len(clean_data) == 0: return result

            # Convert to datetime, coercing errors to NaT (Not a Time).
            parsed_dates = pd.to_datetime(clean_data, errors='coerce')
            valid_dates = parsed_dates.dropna()

            result.update({
                "valid_dates": len(valid_dates),
                "invalid_dates": len(clean_data) - len(valid_dates),
                "success_rate": round(len(valid_dates) / len(clean_data) * 100, 1) if len(clean_data) > 0 else 0
            })

            if len(valid_dates) > 0:
                result.update({
                    "date_range": [str(valid_dates.min()), str(valid_dates.max())],
                    "sample_format": str(valid_dates.iloc[0])  # Show an example of the parsed format.
                })
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    def _analyze_categorical_safe(self, series: pd.Series) -> dict:
        """Performs detailed, safe analysis for categorical columns."""
        result = {"analysis": "categorical"}
        try:
            clean_data = series.dropna()
            if len(clean_data) == 0: return result

            try:
                # Get the frequency of each category.
                value_counts = clean_data.value_counts()
                result["categories"] = len(value_counts)
                # Show the top 20 most frequent categories.
                result["values"] = {str(k): int(v) for k, v in value_counts.head(20).items()}
                if len(value_counts) > 0:
                    result["most_common_percentage"] = round(value_counts.iloc[0] / len(clean_data) * 100, 1)
            except Exception:
                # Fallback if value_counts fails (e.g., on unhashable types).
                result["categories"] = len(set(clean_data.astype(str)))
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    def _analyze_text_safe(self, series: pd.Series) -> dict:
        """Performs detailed, safe analysis for text columns."""
        result = {"analysis": "text"}
        try:
            clean_data = series.dropna().astype(str)
            if len(clean_data) == 0: return result

            # Calculate text length statistics.
            lengths = clean_data.str.len()
            result.update({
                "avg_length": round(lengths.mean(), 1) if len(lengths) > 0 else 0,
                "max_length": int(lengths.max()) if len(lengths) > 0 else 0,
                "min_length": int(lengths.min()) if len(lengths) > 0 else 0,
            })

            # Count common text patterns.
            result.update({
                "contains_numbers": int(clean_data.str.contains(r'\d', na=False).sum()),
                "contains_special_chars": int(clean_data.str.contains(r'[^a-zA-Z0-9\s]', na=False).sum())
            })
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    def _get_safe_samples(self, series: pd.Series, n=3) -> list:
        """
        Gets a small number of sample values from a series with high safety.
        Handles long strings and unprintable characters.
        """
        try:
            non_null = series.dropna()
            if len(non_null) == 0: return []

            # Select samples from the start, middle, and end of the series.
            samples = []
            if len(non_null) <= n:
                samples = non_null.tolist()
            else:
                indices = [0, len(non_null) // 2, len(non_null) - 1]
                for idx in indices[:n]: samples.append(non_null.iloc[idx])

            # Safely convert each sample to a string for JSON serialization.
            safe_samples = []
            for sample in samples:
                try:
                    str_sample = str(sample)
                    # Truncate very long samples for readability.
                    if len(str_sample) > 100:
                        str_sample = str_sample[:97] + "..."
                    safe_samples.append(str_sample)
                except Exception:
                    safe_samples.append("<unprintable>")
            return safe_samples[:n]
        except Exception:
            return []

    def _get_file_metadata_safe(self, file_path: str, df: pd.DataFrame) -> dict:
        """Gathers file system and DataFrame metadata safely."""
        metadata = {}
        # Get file system metadata.
        try:
            stat_info = os.stat(file_path)
            metadata.update({
                "file_size_bytes": stat_info.st_size,
                "file_size_mb": round(stat_info.st_size / (1024 * 1024), 2),
                "modified_time": stat_info.st_mtime
            })
        except Exception:
            pass  # Fail silently if file metadata is inaccessible.

        # Get DataFrame memory usage.
        try:
            metadata.update({
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                "total_cells": len(df) * len(df.columns)
            })
        except Exception:
            pass
        return metadata


class RobustProfilerGUI:
    """
    The Tkinter-based Graphical User Interface for the profiler. It provides
    an interactive way to select files, run the analysis, and view results.
    The analysis itself is run in a background thread to keep the GUI responsive.
    """

    def __init__(self, master):
        """Initializes the GUI application."""
        self.master = master
        self.profiler = UltraRobustCSVProfiler()
        # A thread-safe queue to pass results from the worker thread to the GUI thread.
        self.result_queue = queue.Queue()
        self.setup_ui()

    def setup_ui(self):
        """Sets up all the widgets and layouts for the GUI."""
        self.master.title("Robust CSV Profiler")
        self.master.geometry("800x500")
        self.master.resizable(True, True)

        main_frame = ttk.Frame(self.master, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)

        # --- File Selection Widgets ---
        ttk.Label(main_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(main_frame, textvariable=self.file_var, state='readonly')
        file_entry.grid(row=0, column=1, sticky=tk.EW, padx=(0, 10), pady=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_file).grid(row=0, column=2, pady=5)

        # --- Output Selection Widgets ---
        ttk.Label(main_frame, text="Save to:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.output_var = tk.StringVar()
        output_entry = ttk.Entry(main_frame, textvariable=self.output_var, state='readonly')
        output_entry.grid(row=1, column=1, sticky=tk.EW, padx=(0, 10), pady=5)
        self.save_btn = ttk.Button(main_frame, text="Save As...", command=self.save_as, state='disabled')
        self.save_btn.grid(row=1, column=2, pady=5)

        # --- Options Frame ---
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=10)
        self.include_samples_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include sample values", variable=self.include_samples_var).pack(
            side=tk.LEFT, padx=10)
        self.detailed_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Detailed statistical analysis", variable=self.detailed_analysis_var).pack(
            side=tk.LEFT, padx=10)

        # --- Analyze Button and Progress Bar ---
        self.analyze_btn = ttk.Button(main_frame, text="🔍 Analyze CSV", command=self.start_analysis, state='disabled')
        self.analyze_btn.grid(row=3, column=0, columnspan=3, pady=(15, 10))
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')

        # --- Status and Log Area ---
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=10)
        status_frame.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Select a CSV file to begin analysis")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky=tk.W)
        self.log_text = tk.Text(status_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=1, column=0, sticky=tk.EW, pady=(5, 0))
        scrollbar.grid(row=1, column=1, sticky=tk.NS, pady=(5, 0))

        # --- Bindings ---
        # Trace changes to the file/output paths to enable/disable buttons.
        self.file_var.trace('w', self.update_buttons)
        self.output_var.trace('w', self.update_buttons)

    def browse_file(self):
        """Opens a file dialog to select an input CSV file."""
        filename = filedialog.askopenfilename(
            title="Select CSV file (any format)",
            filetypes=[("CSV, TSV, Text", "*.csv *.tsv *.txt"), ("All files", "*.*")]
        )
        if filename:
            self.file_var.set(filename)
            # Automatically suggest a .json output file name.
            output_path = Path(filename).with_suffix('.json')
            self.output_var.set(str(output_path))
            self.status_var.set("File selected - ready to analyze")
            self.log_message(f"Selected: {os.path.basename(filename)}")

    def save_as(self):
        """Opens a file dialog to select the output JSON file path."""
        filename = filedialog.asksaveasfilename(
            title="Save profile as",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.output_var.set(filename)

    def update_buttons(self, *args):
        """Enables or disables buttons based on whether file paths are set."""
        if self.file_var.get():
            self.save_btn.config(state='normal')
            if self.output_var.get():
                self.analyze_btn.config(state='normal')
                self.status_var.set("Ready to analyze")
            else:
                self.analyze_btn.config(state='disabled')
                self.status_var.set("Choose output location")
        else:
            self.save_btn.config(state='disabled')
            self.analyze_btn.config(state='disabled')
            self.status_var.set("Select a CSV file to begin analysis")

    def log_message(self, message):
        """Adds a message to the GUI's log text area."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)  # Auto-scroll to the bottom.
        self.master.update_idletasks()

    def start_analysis(self):
        """
        Starts the CSV analysis in a background thread to prevent the GUI
        from freezing during a potentially long operation.
        """
        self.analyze_btn.config(state='disabled')
        self.log_text.delete(1.0, tk.END)  # Clear previous logs.
        self.progress.grid(row=5, column=0, columnspan=3, sticky=tk.EW, pady=(5, 0))
        self.progress.start(10)
        self.status_var.set("🔄 Analyzing CSV file...")
        self.log_message("Starting analysis...")

        # Create and start a daemon thread for the analysis.
        # Daemon threads automatically exit when the main program exits.
        thread = threading.Thread(
            target=self.run_analysis,
            args=(self.file_var.get(), self.output_var.get()),
            daemon=True
        )
        thread.start()
        # Start polling the result queue.
        self.master.after(100, self.check_results)

    def run_analysis(self, input_path, output_path):
        """
        This is the target function for the background thread. It runs the
        profiler, applies user options, and puts the result in the queue.
        """
        try:
            profile = self.profiler.profile(input_path)

            # Apply GUI options to the final profile.
            if not self.include_samples_var.get():
                for col in profile.get("columns", []): col.pop("samples", None)
            if not self.detailed_analysis_var.get():
                for col in profile.get("columns", []):
                    if "statistics" in col:
                        stats = col["statistics"]
                        col["statistics"] = {k: v for k, v in stats.items() if k in ["min", "max", "mean", "count"]}
                    col.pop("quartiles", None)

            # Save the result to a JSON file.
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)

            # Put the successful result into the queue for the GUI thread.
            self.result_queue.put(("SUCCESS", output_path, profile))

        except Exception as e:
            # Put any critical error into the queue.
            self.result_queue.put(("ERROR", f"Critical analysis failure: {str(e)}", traceback.format_exc()))

    def check_results(self):
        """
        Periodically checks the result queue. If a result is found, it's
        processed. If not, it schedules itself to check again later.
        """
        try:
            result = self.result_queue.get_nowait()  # Non-blocking get.
            self.handle_result(result)
        except queue.Empty:
            # If the queue is empty, check again in 100ms.
            self.master.after(100, self.check_results)

    def handle_result(self, result):
        """Processes the final result from the worker thread and updates the GUI."""
        self.progress.stop()
        self.progress.grid_remove()
        self.analyze_btn.config(state='normal')

        status, payload1, payload2 = result[0], result[1], result[2]

        if status == "SUCCESS":
            output_path, profile = payload1, payload2
            shape = profile.get("shape", {})
            warnings = profile.get("warnings", [])
            success_msg = (
                f"Analysis completed successfully!\n\n"
                f"File: {profile.get('file', 'N/A')}\n"
                f"Size: {shape.get('rows', 0):,} rows × {shape.get('columns', 0)} columns\n"
                f"Saved to: {os.path.basename(output_path)}\n"
                f"Warnings: {len(warnings)}"
            )

            self.status_var.set("Analysis completed successfully!")
            self.log_message(f"Analysis completed! Status: {profile.get('status', 'unknown')}")

            # Log any warnings encountered during the process.
            if warnings:
                self.log_message("\nWarnings encountered:")
                for warning in warnings: self.log_message(f"  • {warning}")

            # Log a summary of detected column types.
            columns = profile.get("columns", [])
            type_counts = {}
            error_count = sum(1 for col in columns if col.get("type") == "error")
            for col in columns:
                col_type = col.get("type", "unknown")
                if col_type != "error": type_counts[col_type] = type_counts.get(col_type, 0) + 1

            self.log_message(f"\nColumn types detected:")
            for col_type, count in sorted(type_counts.items()): self.log_message(f"  • {col_type}: {count}")
            if error_count > 0: self.log_message(f"  • errors: {error_count}")

            messagebox.showinfo("Analysis Complete", success_msg)

        elif status == "ERROR":
            error_msg, traceback_info = payload1, payload2
            self.status_var.set("Analysis failed")
            self.log_message(f"ERROR: {error_msg}")
            if traceback_info:
                self.log_message("Full error details:")
                self.log_message(traceback_info)
            messagebox.showerror("Analysis Failed", f"Analysis failed:\n\n{error_msg}\n\nCheck the log for details.")


def main():
    """
    Main entry point for the script. It determines whether to launch the GUI
    or run in command-line mode based on the presence of command-line arguments.
    """
    # If there are command-line arguments (other than the script name itself), run CLI mode.
    if len(sys.argv) > 1:
        # --- CLI Mode ---
        parser = argparse.ArgumentParser(
            description="A highly robust CSV profiler designed to handle a wide range of data formats and errors.",
            epilog="This tool is designed to process most CSV files gracefully, with extensive error handling to prevent crashes."
        )
        parser.add_argument('csv_file', help='Path to the CSV file to analyze.')
        parser.add_argument('-o', '--output', help='Path for the output JSON file (default: input_file.json).')
        parser.add_argument('--no-samples', action='store_true', help='Exclude sample values from the output.')
        parser.add_argument('--simple', action='store_true', help='Perform a simplified analysis (faster).')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Show detailed progress and summary information.')
        args = parser.parse_args()

        if not os.path.exists(args.csv_file):
            print(f"Error: File not found at '{args.csv_file}'")
            return 1

        if args.verbose:
            print(f"Analyzing: {args.csv_file}")
            print("Using robust profiler...")

        profiler = UltraRobustCSVProfiler()
        profile = profiler.profile(args.csv_file)

        if profile.get("status") == "error":
            print(f"Error during analysis: {profile.get('error', 'Unknown error')}")
            if args.verbose and "traceback" in profile:
                print("\nFull error details:")
                print(profile["traceback"])
            return 1

        # Apply CLI arguments to the profile.
        if args.no_samples:
            for col in profile.get("columns", []): col.pop("samples", None)
        if args.simple:
            for col in profile.get("columns", []):
                if "statistics" in col:
                    stats = col["statistics"]
                    col["statistics"] = {k: v for k, v in stats.items() if k in ["min", "max", "mean", "count"]}
                col.pop("quartiles", None)

        output_file = args.output or str(Path(args.csv_file).with_suffix('.json'))

        # Save the final JSON report.
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving to {output_file}: {str(e)}")
            return 1

        # Print summary to the console.
        shape = profile.get("shape", {})
        warnings = profile.get("warnings", [])
        print(f"\nAnalysis completed successfully!")
        print(f"Shape: {shape.get('rows', 0):,} rows × {shape.get('columns', 0)} columns")
        print(f"Profile saved to: {output_file}")

        if args.verbose:
            if warnings:
                print(f"Warnings ({len(warnings)}):")
                for warning in warnings[:5]: print(f"   • {warning}")
                if len(warnings) > 5: print(f"   ... and {len(warnings) - 5} more")

            columns = profile.get("columns", [])
            type_counts = {}
            error_count = sum(1 for col in columns if col.get("type") == "error")
            for col in columns:
                col_type = col.get("type", "unknown")
                if col_type != "error": type_counts[col_type] = type_counts.get(col_type, 0) + 1

            print("Column Types Detected:")
            for col_type, count in sorted(type_counts.items()): print(f"   • {col_type}: {count}")
            if error_count > 0: print(f"   • errors: {error_count}")

        return 0

    else:
        # --- GUI Mode ---
        try:
            root = tk.Tk()
            app = RobustProfilerGUI(root)
            # Center the window on the screen.
            root.update_idletasks()
            x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
            y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
            root.geometry(f"+{x}+{y}")

            root.title("Robust CSV Profiler - Resilient Data Analysis")

            root.mainloop()
            return 0
        except Exception as e:
            # Fallback message if the GUI fails to start.
            print(f"GUI failed to start: {str(e)}")
            print("Try using command-line mode instead:")
            print(f"python {sys.argv[0]} your_file.csv")
            return 1


# This standard construct ensures that main() is called only when the script
# is executed directly (not when imported as a module).
if __name__ == "__main__":
    sys.exit(main())