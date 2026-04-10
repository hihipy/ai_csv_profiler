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
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class DefensiveAnalyzer:
    """
    A utility class with static methods for executing operations with extensive
    error handling, aiming for fail-safe execution.
    """

    @staticmethod
    def safe_execute(func, *args, default=None, **kwargs):
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
        try:
            if series.dtype == 'object':
                cleaned = series.astype(str).str.strip()
                null_placeholders = [
                    '', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', '#N/A',
                    '#NULL!', '#DIV/0!', '#REF!', '#NAME?', '#NUM!', '#VALUE!',
                    'inf', '-inf', 'infinity', '-infinity'
                ]
                cleaned = cleaned.replace(null_placeholders, np.nan)
                result = pd.to_numeric(cleaned, errors='coerce')
            else:
                result = pd.to_numeric(series, errors='coerce')

            result = result.replace([np.inf, -np.inf], np.nan)

            if result.notna().sum() < len(result) * 0.1:
                return series.fillna(fill_value)

            return result.fillna(fill_value)

        except Exception:
            return series.fillna(fill_value)

    @staticmethod
    def safe_statistics(data, stat_name):
        try:
            if len(data) == 0:
                return None

            clean_data = data.dropna()
            if len(clean_data) == 0:
                return None

            if stat_name in ['mean', 'std', 'min', 'max']:
                q99 = clean_data.quantile(0.99)
                q01 = clean_data.quantile(0.01)
                if pd.notna(q99) and pd.notna(q01):
                    clean_data = clean_data[(clean_data >= q01) & (clean_data <= q99)]

            if len(clean_data) == 0:
                return None

            if stat_name == 'min':
                return float(clean_data.min())
            elif stat_name == 'max':
                return float(clean_data.max())
            elif stat_name == 'mean':
                return float(clean_data.mean())
            elif stat_name == 'median':
                return float(clean_data.median())
            elif stat_name == 'std':
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
    The core engine of the application. Orchestrates the entire profiling
    process from reading the CSV to compiling the final JSON report.
    """

    def __init__(self):
        self.analyzer = DefensiveAnalyzer()
        self.encoding_attempts = ['utf-8-sig', 'utf-8', 'cp1252', 'iso-8859-1',
                                  'latin1', 'ascii', 'utf-16', 'utf-32']
        self.separator_attempts = [',', ';', '\t', '|', ':', ' ']

    def profile(self, file_path: str) -> dict:
        base_profile = {
            "file": os.path.basename(file_path),
            "status": "unknown",
            "shape": {"rows": 0, "columns": 0},
            "columns": [],
            "warnings": [],
            "info": [],
            "metadata": {}
        }

        try:
            df, read_warnings, read_info = self._read_csv_ultra_safe(file_path)
            base_profile["warnings"].extend(read_warnings)
            base_profile["info"].extend(read_info)

            if df is None or df.empty:
                base_profile["status"] = "empty_or_unreadable"
                base_profile["error"] = "Could not read file or file is empty"
                return base_profile

            base_profile["shape"] = {"rows": len(df), "columns": len(df.columns)}
            base_profile["status"] = "processed"
            base_profile["columns"] = self._analyze_all_columns_safe(df)
            base_profile["metadata"] = self._get_file_metadata_safe(file_path, df)

            return base_profile

        except Exception as e:
            base_profile["status"] = "error"
            base_profile["error"] = f"Critical error: {str(e)}"
            base_profile["traceback"] = traceback.format_exc()
            return base_profile

    def _read_csv_ultra_safe(self, file_path: str) -> tuple:
        """
        Reads a CSV file using a multi-stage fallback strategy.

        Returns:
            tuple: A DataFrame (or None), a list of warnings, and a list of
                   informational messages (kept separate from true warnings).
        """
        warnings_list = []
        info_list = []

        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return None, ["File is empty"], info_list
            if file_size > 500 * 1024 * 1024:
                warnings_list.append(f"Large file ({file_size / 1024 / 1024:.1f}MB)")
        except Exception:
            warnings_list.append("Could not determine file size")

        # Strategy 1: Iterate through encodings and separators.
        for encoding in self.encoding_attempts:
            for separator in self.separator_attempts:
                try:
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        sep=separator,
                        on_bad_lines='skip',
                        low_memory=False,
                        dtype=str,
                        na_values=[''],
                        keep_default_na=False,
                        engine='python'
                    )
                    if len(df.columns) > 1 and len(df) > 0:
                        info_list.append(f"Read with encoding={encoding}, separator='{separator}'")
                        return df, warnings_list, info_list
                except Exception:
                    continue

        # Strategy 1.5: Try common encodings with auto-separator detection.
        # This catches BOM-encoded files and other cases where Strategy 1 falls through.
        for encoding in ['utf-8-sig', 'utf-8', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=None,
                    engine='python',
                    on_bad_lines='skip',
                    dtype=str,
                    na_values=[''],
                    keep_default_na=False
                )
                if len(df.columns) > 1 and len(df) > 0:
                    info_list.append(f"Read with encoding={encoding}, separator=auto-detected")
                    return df, warnings_list, info_list
            except Exception:
                continue

        # Strategy 2: Fallback using pandas' automatic separator detection, utf-8 only.
        try:
            df = pd.read_csv(file_path, encoding='utf-8', sep=None, engine='python',
                             on_bad_lines='skip', dtype=str)
            warnings_list.append("Used fallback reading method (auto-separator)")
            return df, warnings_list, info_list
        except Exception:
            pass

        # Strategy 3: Last resort - manual text parsing.
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:1000]

            if not lines:
                return None, ["File appears empty or unreadable"], info_list

            first_line = lines[0].strip()
            best_sep = max(self.separator_attempts, key=lambda sep: first_line.count(sep))

            data = []
            headers = first_line.split(best_sep)
            for line in lines[1:]:
                row = line.strip().split(best_sep)
                while len(row) < len(headers):
                    row.append('')
                row = row[:len(headers)]
                data.append(row)

            df = pd.DataFrame(data, columns=headers)
            warnings_list.append("Used manual text parsing as last resort")
            return df, warnings_list, info_list
        except Exception:
            return None, warnings_list + ["All reading methods failed"], info_list

    def _analyze_all_columns_safe(self, df: pd.DataFrame) -> list:
        columns = []
        for i, col_name in enumerate(df.columns):
            try:
                col_info = self._analyze_single_column_safe(df[col_name], str(col_name))
                col_info["position"] = i
                columns.append(col_info)
            except Exception as e:
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
        base_info = {
            "name": col_name,
            "type": "unknown",
            "status": "analyzing",
            "total_values": len(series),
            "samples": []
        }
        try:
            base_info["samples"] = self._get_safe_samples(series)
            base_info["missing"] = self.analyzer.safe_execute(lambda: int(series.isnull().sum()), default=0)
            base_info["empty_strings"] = self.analyzer.safe_execute(
                lambda: int((series.astype(str) == '').sum()), default=0)
            base_info["unique"] = self.analyzer.safe_execute(lambda: int(series.nunique()), default=0)

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
            else:
                base_info.update(self._analyze_text_safe(series))

            base_info["status"] = "completed"
            return base_info
        except Exception as e:
            base_info["status"] = "error"
            base_info["error"] = str(e)
            base_info["type"] = "error"
            return base_info

    def _detect_column_type_safe(self, series: pd.Series) -> str:
        try:
            clean_series = series.dropna()
            if len(clean_series) == 0:
                return "empty"

            str_values = clean_series.astype(str).str.strip()
            str_values = str_values[str_values != '']
            if len(str_values) == 0:
                return "empty"

            unique_str_lower = set(str_values.str.lower().unique())

            boolean_patterns = [{'true', 'false'}, {'yes', 'no'}, {'y', 'n'}, {'1', '0'},
                                 {'on', 'off'}, {'enabled', 'disabled'}, {'active', 'inactive'}]
            for pattern in boolean_patterns:
                if len(unique_str_lower) <= 2 and unique_str_lower.issubset(pattern):
                    return "boolean"

            col_name = str(series.name).lower() if series.name else ""
            if any(keyword in col_name for keyword in ['date', 'dt', 'time', 'start', 'end']):
                if self._could_be_datetime(str_values):
                    return "datetime"

            if any(keyword in col_name for keyword in ['id', 'response', 'uuid', 'key']):
                return "text"

            try:
                numeric_series = pd.to_numeric(str_values, errors='coerce')
                numeric_ratio = numeric_series.notna().sum() / len(str_values)
                if numeric_ratio > 0.9:
                    unique_numeric = set(numeric_series.dropna().unique())
                    if len(unique_numeric) <= 2 and unique_numeric.issubset({0, 1, 0.0, 1.0}):
                        return "boolean"
                    return "numeric"
            except Exception:
                pass

            if self._could_be_datetime(str_values):
                return "datetime"

            unique_count = len(unique_str_lower)
            total_count = len(str_values)
            if unique_count <= min(50, total_count * 0.5):
                return "categorical"

            return "text"

        except Exception:
            return "text"

    def _could_be_datetime(self, str_series) -> bool:
        try:
            sample = str_series.head(10)
            success_count = 0
            for val in sample:
                val_str = str(val).strip()
                if len(val_str) < 8:
                    continue
                datetime_indicators = ['-', '/', ':', 'T', ' ']
                if any(indicator in val_str for indicator in datetime_indicators):
                    try:
                        if pd.notna(pd.to_datetime(val_str, errors='raise')):
                            success_count += 1
                    except Exception:
                        continue
            return success_count >= len(sample) * 0.7
        except Exception:
            return False

    def _analyze_numeric_ultra_safe(self, series: pd.Series) -> dict:
        """
        Performs detailed numeric analysis. Reports clipped min/max (1%-99%)
        to avoid garbage outliers, and separately reports the true raw min/max
        so extreme values are not silently hidden.
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
                    "count": len(clean_data),
                    "value": float(clean_data.iloc[0]) if len(clean_data) == 1 else None,
                    "status": "insufficient_data_for_analysis"
                }

            result["valid_numbers"] = len(clean_data)
            result["invalid_numbers"] = len(series) - len(clean_data)

            stats = {}
            for stat_name in ['min', 'max', 'mean', 'median', 'std']:
                stat_value = self.analyzer.safe_statistics(clean_data, stat_name)
                if stat_value is not None:
                    stats[stat_name] = round(stat_value, 4) if isinstance(stat_value, float) else stat_value
            result["statistics"] = stats

            try:
                raw_min = float(clean_data.min())
                raw_max = float(clean_data.max())
                if raw_min != stats.get('min') or raw_max != stats.get('max'):
                    result["raw_statistics"] = {
                        "min": raw_min,
                        "max": raw_max,
                        "note": "Raw min/max before 1%-99% outlier clipping"
                    }
            except Exception:
                pass

            quartiles = {}
            for q, name in [(0.25, "q25"), (0.75, "q75")]:
                q_val = self.analyzer.safe_statistics(clean_data, f"quantile_{q}")
                if q_val is not None:
                    quartiles[name] = round(q_val, 4)
            result["quartiles"] = quartiles

            result["zero_count"] = self.analyzer.safe_execute(lambda: int((clean_data == 0).sum()), default=0)
            result["negative_count"] = self.analyzer.safe_execute(lambda: int((clean_data < 0).sum()), default=0)

            unique_count = self.analyzer.safe_execute(lambda: clean_data.nunique(), default=0)
            if unique_count > 0:
                if unique_count == len(clean_data):
                    result["pattern"] = "likely_id"
                elif unique_count <= 20:
                    result["pattern"] = "likely_categorical"
                    try:
                        value_counts = clean_data.value_counts().head(10)
                        result["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
                    except Exception:
                        pass
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    def _analyze_boolean_safe(self, series: pd.Series) -> dict:
        result = {"analysis": "boolean"}
        try:
            clean_data = series.dropna()
            if len(clean_data) == 0:
                return result

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
        result = {"analysis": "datetime"}
        try:
            clean_data = series.dropna().astype(str)
            if len(clean_data) == 0:
                return result

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
                    "sample_format": str(valid_dates.iloc[0])
                })
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    def _analyze_categorical_safe(self, series: pd.Series) -> dict:
        result = {"analysis": "categorical"}
        try:
            clean_data = series.dropna()
            if len(clean_data) == 0:
                return result

            try:
                value_counts = clean_data.value_counts()
                result["categories"] = len(value_counts)
                result["values"] = {str(k): int(v) for k, v in value_counts.head(20).items()}
                if len(value_counts) > 0:
                    result["most_common_percentage"] = round(value_counts.iloc[0] / len(clean_data) * 100, 1)
            except Exception:
                result["categories"] = len(set(clean_data.astype(str)))
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    def _analyze_text_safe(self, series: pd.Series) -> dict:
        result = {"analysis": "text"}
        try:
            clean_data = series.dropna().astype(str)
            if len(clean_data) == 0:
                return result

            lengths = clean_data.str.len()
            result.update({
                "avg_length": round(lengths.mean(), 1) if len(lengths) > 0 else 0,
                "max_length": int(lengths.max()) if len(lengths) > 0 else 0,
                "min_length": int(lengths.min()) if len(lengths) > 0 else 0,
            })

            result.update({
                "contains_numbers": int(clean_data.str.contains(r'\d', na=False).sum()),
                "contains_special_chars": int(clean_data.str.contains(r'[^a-zA-Z0-9\s]', na=False).sum())
            })
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    def _get_safe_samples(self, series: pd.Series, n=3) -> list:
        try:
            non_null = series.dropna()
            if len(non_null) == 0:
                return []

            samples = []
            if len(non_null) <= n:
                samples = non_null.tolist()
            else:
                indices = [0, len(non_null) // 2, len(non_null) - 1]
                for idx in indices[:n]:
                    samples.append(non_null.iloc[idx])

            safe_samples = []
            for sample in samples:
                try:
                    str_sample = str(sample)
                    if len(str_sample) > 100:
                        str_sample = str_sample[:97] + "..."
                    safe_samples.append(str_sample)
                except Exception:
                    safe_samples.append("<unprintable>")
            return safe_samples[:n]
        except Exception:
            return []

    def _get_file_metadata_safe(self, file_path: str, df: pd.DataFrame) -> dict:
        metadata = {}
        try:
            stat_info = os.stat(file_path)
            metadata.update({
                "file_size_bytes": stat_info.st_size,
                "file_size_mb": round(stat_info.st_size / (1024 * 1024), 2),
                "modified_time": stat_info.st_mtime
            })
        except Exception:
            pass

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
    The Tkinter-based GUI for the profiler. Analysis runs in a background
    thread to keep the interface responsive.
    """

    def __init__(self, master):
        self.master = master
        self.profiler = UltraRobustCSVProfiler()
        self.result_queue = queue.Queue()
        self.setup_ui()

    def setup_ui(self):
        self.master.title("Robust CSV Profiler")
        self.master.geometry("800x500")
        self.master.resizable(True, True)

        main_frame = ttk.Frame(self.master, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)

        ttk.Label(main_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(main_frame, textvariable=self.file_var, state='readonly')
        file_entry.grid(row=0, column=1, sticky=tk.EW, padx=(0, 10), pady=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_file).grid(row=0, column=2, pady=5)

        ttk.Label(main_frame, text="Save to:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.output_var = tk.StringVar()
        output_entry = ttk.Entry(main_frame, textvariable=self.output_var, state='readonly')
        output_entry.grid(row=1, column=1, sticky=tk.EW, padx=(0, 10), pady=5)
        self.save_btn = ttk.Button(main_frame, text="Save As...", command=self.save_as, state='disabled')
        self.save_btn.grid(row=1, column=2, pady=5)

        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=10)
        self.include_samples_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include sample values",
                        variable=self.include_samples_var).pack(side=tk.LEFT, padx=10)
        self.detailed_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Detailed statistical analysis",
                        variable=self.detailed_analysis_var).pack(side=tk.LEFT, padx=10)

        self.analyze_btn = ttk.Button(main_frame, text="🔍 Analyze CSV",
                                      command=self.start_analysis, state='disabled')
        self.analyze_btn.grid(row=3, column=0, columnspan=3, pady=(15, 10))
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')

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

        self.file_var.trace_add('write', self.update_buttons)
        self.output_var.trace_add('write', self.update_buttons)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select CSV file (any format)",
            filetypes=[("CSV, TSV, Text", "*.csv *.tsv *.txt"), ("All files", "*.*")]
        )
        if filename:
            self.file_var.set(filename)
            output_path = Path(filename).with_suffix('.json')
            self.output_var.set(str(output_path))
            self.status_var.set("File selected - ready to analyze")
            self.log_message(f"Selected: {os.path.basename(filename)}")

    def save_as(self):
        filename = filedialog.asksaveasfilename(
            title="Save profile as",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.output_var.set(filename)

    def update_buttons(self, *args):
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
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.master.update_idletasks()

    def start_analysis(self):
        self.analyze_btn.config(state='disabled')
        self.log_text.delete(1.0, tk.END)
        self.progress.grid(row=5, column=0, columnspan=3, sticky=tk.EW, pady=(5, 0))
        self.progress.start(10)
        self.status_var.set("🔄 Analyzing CSV file...")
        self.log_message("Starting analysis...")

        thread = threading.Thread(
            target=self.run_analysis,
            args=(self.file_var.get(), self.output_var.get()),
            daemon=True
        )
        thread.start()
        self.master.after(100, self.check_results)

    def run_analysis(self, input_path, output_path):
        try:
            profile = self.profiler.profile(input_path)

            if not self.include_samples_var.get():
                for col in profile.get("columns", []):
                    col.pop("samples", None)
            if not self.detailed_analysis_var.get():
                for col in profile.get("columns", []):
                    if "statistics" in col:
                        stats = col["statistics"]
                        col["statistics"] = {k: v for k, v in stats.items() if k in ["min", "max", "mean", "count"]}
                    col.pop("quartiles", None)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)

            self.result_queue.put(("SUCCESS", output_path, profile))

        except Exception as e:
            self.result_queue.put(("ERROR", f"Critical analysis failure: {str(e)}", traceback.format_exc()))

    def check_results(self):
        try:
            result = self.result_queue.get_nowait()
            self.handle_result(result)
        except queue.Empty:
            self.master.after(100, self.check_results)

    def handle_result(self, result):
        self.progress.stop()
        self.progress.grid_remove()
        self.analyze_btn.config(state='normal')

        status, payload1, payload2 = result[0], result[1], result[2]

        if status == "SUCCESS":
            output_path, profile = payload1, payload2
            shape = profile.get("shape", {})
            warnings = profile.get("warnings", [])
            info = profile.get("info", [])

            success_msg = (
                f"Analysis completed successfully!\n\n"
                f"File: {profile.get('file', 'N/A')}\n"
                f"Size: {shape.get('rows', 0):,} rows x {shape.get('columns', 0)} columns\n"
                f"Saved to: {os.path.basename(output_path)}\n"
                f"Warnings: {len(warnings)}"
            )

            self.status_var.set("Analysis completed successfully!")
            self.log_message(f"Analysis completed! Status: {profile.get('status', 'unknown')}")

            if info:
                self.log_message("\nInfo:")
                for msg in info:
                    self.log_message(f"  i {msg}")

            if warnings:
                self.log_message("\nWarnings:")
                for warning in warnings:
                    self.log_message(f"  ! {warning}")

            columns = profile.get("columns", [])
            type_counts = {}
            error_count = sum(1 for col in columns if col.get("type") == "error")
            for col in columns:
                col_type = col.get("type", "unknown")
                if col_type != "error":
                    type_counts[col_type] = type_counts.get(col_type, 0) + 1

            self.log_message(f"\nColumn types detected:")
            for col_type, count in sorted(type_counts.items()):
                self.log_message(f"  - {col_type}: {count}")
            if error_count > 0:
                self.log_message(f"  - errors: {error_count}")

            messagebox.showinfo("Analysis Complete", success_msg)

        elif status == "ERROR":
            error_msg, traceback_info = payload1, payload2
            self.status_var.set("Analysis failed")
            self.log_message(f"ERROR: {error_msg}")
            if traceback_info:
                self.log_message("Full error details:")
                self.log_message(traceback_info)
            messagebox.showerror("Analysis Failed",
                                 f"Analysis failed:\n\n{error_msg}\n\nCheck the log for details.")


def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="A highly robust CSV profiler designed to handle a wide range of data formats and errors.",
            epilog="Designed to process most CSV files gracefully, with extensive error handling to prevent crashes."
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

        profiler = UltraRobustCSVProfiler()
        profile = profiler.profile(args.csv_file)

        if profile.get("status") == "error":
            print(f"Error during analysis: {profile.get('error', 'Unknown error')}")
            if args.verbose and "traceback" in profile:
                print("\nFull error details:")
                print(profile["traceback"])
            return 1

        if args.no_samples:
            for col in profile.get("columns", []):
                col.pop("samples", None)
        if args.simple:
            for col in profile.get("columns", []):
                if "statistics" in col:
                    stats = col["statistics"]
                    col["statistics"] = {k: v for k, v in stats.items() if k in ["min", "max", "mean", "count"]}
                col.pop("quartiles", None)

        output_file = args.output or str(Path(args.csv_file).with_suffix('.json'))

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving to {output_file}: {str(e)}")
            return 1

        shape = profile.get("shape", {})
        warnings_list = profile.get("warnings", [])
        info_list = profile.get("info", [])
        print(f"\nAnalysis completed successfully!")
        print(f"Shape: {shape.get('rows', 0):,} rows x {shape.get('columns', 0)} columns")
        print(f"Profile saved to: {output_file}")

        if args.verbose:
            if info_list:
                print("Info:")
                for msg in info_list:
                    print(f"   i {msg}")
            if warnings_list:
                print(f"Warnings ({len(warnings_list)}):")
                for warning in warnings_list[:5]:
                    print(f"   ! {warning}")
                if len(warnings_list) > 5:
                    print(f"   ... and {len(warnings_list) - 5} more")

            columns = profile.get("columns", [])
            type_counts = {}
            error_count = sum(1 for col in columns if col.get("type") == "error")
            for col in columns:
                col_type = col.get("type", "unknown")
                if col_type != "error":
                    type_counts[col_type] = type_counts.get(col_type, 0) + 1

            print("Column Types Detected:")
            for col_type, count in sorted(type_counts.items()):
                print(f"   - {col_type}: {count}")
            if error_count > 0:
                print(f"   - errors: {error_count}")

        return 0

    else:
        try:
            root = tk.Tk()
            app = RobustProfilerGUI(root)
            root.update_idletasks()
            x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
            y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
            root.geometry(f"+{x}+{y}")
            root.title("Robust CSV Profiler - Resilient Data Analysis")
            root.mainloop()
            return 0
        except Exception as e:
            print(f"GUI failed to start: {str(e)}")
            print("Try using command-line mode instead:")
            print(f"python {sys.argv[0]} your_file.csv")
            return 1


if __name__ == "__main__":
    sys.exit(main())