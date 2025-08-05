#!/usr/bin/env python3
"""
AI CSV Profiler

Generates just the essential information AI assistants need to understand CSV data.
No fluff, just the facts.
"""

import argparse
import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk, messagebox
import pandas as pd


class SimpleCSVProfiler:
    """Dead simple CSV profiler for AI consumption."""

    def profile(self, file_path: str) -> dict:
        """Generate minimal profile of CSV file."""
        try:
            # Read CSV with encoding fallback
            df = self._read_csv(file_path)

            return {
                "file": os.path.basename(file_path),
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "columns": self._analyze_columns(df)
            }

        except Exception as e:
            return {"error": str(e)}

    def _read_csv(self, file_path: str) -> pd.DataFrame:
        """Read CSV with encoding fallback."""
        for encoding in ['utf-8', 'cp1252', 'iso-8859-1']:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not read CSV file")

    def _analyze_columns(self, df: pd.DataFrame) -> list:
        """Analyze each column with just essential info."""
        columns = []

        for col in df.columns:
            data = df[col]

            col_info = {
                "name": str(col),
                "type": self._get_simple_type(data),
                "missing": int(data.isnull().sum()),
                "unique": int(data.nunique()),
                "samples": self._get_samples(data)
            }

            # Add type-specific info
            if pd.api.types.is_numeric_dtype(data):
                col_info.update(self._analyze_numeric_column(data))

            elif data.nunique() <= 20:  # Categorical
                counts = data.value_counts()
                col_info["values"] = counts.to_dict()

            columns.append(col_info)

        return columns

    def _analyze_numeric_column(self, series: pd.Series) -> dict:
        """Enhanced numeric column analysis."""
        if series.dropna().empty:
            return {"stats": "No non-null values"}

        # Get basic statistics
        stats = series.describe()
        non_null_data = series.dropna()

        result = {
            "range": [float(stats['min']), float(stats['max'])],
            "mean": round(float(stats['mean']), 2),
            "median": round(float(stats['50%']), 2),
            "std": round(float(stats['std']), 2) if not pd.isna(stats['std']) else 0,
            "quartiles": {
                "q25": round(float(stats['25%']), 2),
                "q75": round(float(stats['75%']), 2)
            }
        }

        # Add additional insights
        result["zero_count"] = int((series == 0).sum())
        result["negative_count"] = int((series < 0).sum())

        # Detect if it's likely an ID or categorical numeric
        if series.nunique() == len(series) and series.min() > 0:
            result["likely_id"] = True
        elif series.nunique() <= 10 and all(val == int(val) for val in non_null_data.head(20) if not pd.isna(val)):
            result["likely_categorical"] = True
            # Show value counts for categorical-like numbers
            result["value_counts"] = series.value_counts().head(10).to_dict()

        # Detect potential currency/financial data
        if any(keyword in str(series.name).lower() for keyword in
               ['dollar', 'revenue', 'expense', 'cost', 'price', 'amount']):
            result["likely_currency"] = True
            # Add currency-specific stats
            non_zero = series[series != 0]
            if len(non_zero) > 0:
                result["non_zero_mean"] = round(float(non_zero.mean()), 2)
                result["non_zero_median"] = round(float(non_zero.median()), 2)

        # Check for outliers (basic IQR method)
        q1, q3 = stats['25%'], stats['75%']
        iqr = q3 - q1
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            result["outlier_count"] = len(outliers)
            if len(outliers) > 0:
                result["outlier_pct"] = round((len(outliers) / len(series)) * 100, 1)

        # Distribution insights
        if abs(result["mean"] - result["median"]) / max(abs(result["mean"]), abs(result["median"]), 1) > 0.2:
            if result["mean"] > result["median"]:
                result["distribution"] = "right_skewed"
            else:
                result["distribution"] = "left_skewed"
        else:
            result["distribution"] = "approximately_normal"

        return result

    def _get_simple_type(self, series: pd.Series) -> str:
        """Get simple, AI-friendly type name."""
        if pd.api.types.is_numeric_dtype(series):
            return "number"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "date"
        elif pd.api.types.is_bool_dtype(series):
            return "boolean"
        else:
            return "text"

    def _get_samples(self, series: pd.Series) -> list:
        """Get a few sample values (non-null)."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return []

        samples = non_null.head(3).tolist()
        return [str(val) for val in samples]


class ProfilerGUI:
    """Simple GUI for the CSV profiler."""

    def __init__(self, master):
        self.master = master
        self.profiler = SimpleCSVProfiler()
        self.result_queue = queue.Queue()

        self.setup_ui()

    def setup_ui(self):
        """Set up the GUI."""
        self.master.title("Minimal AI CSV Profiler")
        self.master.geometry("700x200")
        self.master.resizable(True, False)

        # Main frame
        main_frame = ttk.Frame(self.master, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)

        # File selection
        ttk.Label(main_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5)

        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(main_frame, textvariable=self.file_var, state='readonly')
        file_entry.grid(row=0, column=1, sticky=tk.EW, padx=(0, 10), pady=5)

        ttk.Button(main_frame, text="Browse...", command=self.browse_file).grid(row=0, column=2, pady=5)

        # Output selection
        ttk.Label(main_frame, text="Save to:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5)

        self.output_var = tk.StringVar()
        output_entry = ttk.Entry(main_frame, textvariable=self.output_var, state='readonly')
        output_entry.grid(row=1, column=1, sticky=tk.EW, padx=(0, 10), pady=5)

        ttk.Button(main_frame, text="Save As...", command=self.save_as, state='disabled').grid(row=1, column=2, pady=5)

        # Analyze button
        self.analyze_btn = ttk.Button(main_frame, text="🔍 Analyze CSV",
                                      command=self.start_analysis, state='disabled')
        self.analyze_btn.grid(row=2, column=0, columnspan=3, pady=(15, 10))

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')

        # Status
        self.status_var = tk.StringVar(value="Select a CSV file to begin")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                 relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_label.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(10, 0))

        # Bind file selection to enable buttons
        self.file_var.trace('w', self.update_buttons)
        self.output_var.trace('w', self.update_buttons)

    def browse_file(self):
        """Browse for CSV file."""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("TSV files", "*.tsv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            # Validate file
            validation_result = self.validate_file(filename)
            if validation_result["valid"]:
                self.file_var.set(filename)
                # Auto-generate output filename
                output_path = Path(filename).with_suffix('.json')
                self.output_var.set(str(output_path))
                self.status_var.set("File selected - choose output location")
            else:
                messagebox.showerror("Invalid File", validation_result["error"])

    def validate_file(self, file_path):
        """Validate the selected file."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {"valid": False, "error": "File does not exist"}

            # Check if it's a file (not directory)
            if not os.path.isfile(file_path):
                return {"valid": False, "error": "Selected path is not a file"}

            # Check file size (warn if > 100MB)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 100:
                response = messagebox.askyesno(
                    "Large File Warning",
                    f"File is {file_size_mb:.1f}MB. This may take a while to process.\n\nContinue anyway?"
                )
                if not response:
                    return {"valid": False, "error": "File too large - user cancelled"}

            # Check file extension (warn if not CSV/TSV/TXT)
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in ['.csv', '.tsv', '.txt']:
                response = messagebox.askyesno(
                    "File Type Warning",
                    f"File extension '{file_ext}' is not a typical CSV file.\n\nTry to analyze anyway?"
                )
                if not response:
                    return {"valid": False, "error": "Invalid file type - user cancelled"}

            # Quick peek at file content
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()

                # Check if file appears empty
                if not first_line:
                    return {"valid": False, "error": "File appears to be empty"}

                # Check for common separators
                if not any(sep in first_line for sep in [',', ';', '\t', '|']):
                    response = messagebox.askyesno(
                        "Format Warning",
                        "File doesn't appear to contain common CSV separators.\n\nTry to analyze anyway?"
                    )
                    if not response:
                        return {"valid": False, "error": "No CSV separators detected - user cancelled"}

            except Exception as e:
                return {"valid": False, "error": f"Cannot read file: {str(e)}"}

            return {"valid": True, "error": None}

        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}

    def save_as(self):
        """Choose output file."""
        filename = filedialog.asksaveasfilename(
            title="Save profile as",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            self.output_var.set(filename)

    def update_buttons(self, *args):
        """Update button states."""
        if self.file_var.get():
            # Enable save as button
            for widget in self.master.winfo_children():
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button) and "Save As" in child['text']:
                        child.config(state='normal')

            if self.output_var.get():
                self.analyze_btn.config(state='normal')
                self.status_var.set("Ready to analyze")
            else:
                self.analyze_btn.config(state='disabled')
                self.status_var.set("Choose output location")
        else:
            self.analyze_btn.config(state='disabled')
            self.status_var.set("Select a CSV file to begin")

    def start_analysis(self):
        """Start analysis in background thread."""
        self.analyze_btn.config(state='disabled')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=(5, 0))
        self.progress.start(10)
        self.status_var.set("🔄 Analyzing CSV...")

        # Start analysis thread
        thread = threading.Thread(
            target=self.run_analysis,
            args=(self.file_var.get(), self.output_var.get()),
            daemon=True
        )
        thread.start()

        # Check for results
        self.master.after(100, self.check_results)

    def run_analysis(self, input_path, output_path):
        """Run analysis in background."""
        try:
            profile = self.profiler.profile(input_path)

            if "error" in profile:
                self.result_queue.put(("ERROR", profile["error"]))
                return

            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2)

            self.result_queue.put(("SUCCESS", output_path, profile))

        except Exception as e:
            self.result_queue.put(("ERROR", str(e)))

    def check_results(self):
        """Check for analysis results."""
        try:
            result = self.result_queue.get_nowait()
            self.handle_result(result)
        except queue.Empty:
            self.master.after(100, self.check_results)

    def handle_result(self, result):
        """Handle analysis result."""
        self.progress.stop()
        self.progress.grid_remove()
        self.analyze_btn.config(state='normal')

        if result[0] == "SUCCESS":
            output_path, profile = result[1], result[2]

            shape = profile.get("shape", {})
            success_msg = (
                f"✅ Analysis complete!\n\n"
                f"📊 {profile.get('file', 'File')}: {shape.get('rows', 0):,} rows × {shape.get('columns', 0)} columns\n"
                f"📁 Saved to: {os.path.basename(output_path)}"
            )

            self.status_var.set("✅ Analysis complete!")
            messagebox.showinfo("Success", success_msg)

        elif result[0] == "ERROR":
            error_msg = result[1]
            self.status_var.set("❌ Analysis failed")
            messagebox.showerror("Error", f"Analysis failed:\n\n{error_msg}")


def main():
    import sys

    # Check if running in CLI mode
    if len(sys.argv) > 1:
        # CLI mode
        parser = argparse.ArgumentParser(description="Minimal CSV profiler for AI")
        parser.add_argument('csv_file', help='CSV file to analyze')
        parser.add_argument('-o', '--output', help='Output JSON file')

        args = parser.parse_args()

        if not os.path.exists(args.csv_file):
            print(f"Error: {args.csv_file} not found")
            return 1

        profiler = SimpleCSVProfiler()
        profile = profiler.profile(args.csv_file)

        if "error" in profile:
            print(f"Error: {profile['error']}")
            return 1

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(profile, f, indent=2)
            print(f"Profile saved to {args.output}")
        else:
            print(json.dumps(profile, indent=2))

        return 0

    else:
        # GUI mode
        root = tk.Tk()
        app = ProfilerGUI(root)

        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")

        root.mainloop()
        return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
