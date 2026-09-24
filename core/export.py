import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from io import BytesIO
import re


def make_portfolio_output_dict(panel_df, total_plus_summary, loadings_wide):
    return {
        "Portfolio Allocation": panel_df,
        "Performance Statistics": total_plus_summary,
        "Factor Loadings": loadings_wide,
    }


def export_output_dict_to_excel(output_dict, tickers, for_streamlit=False):
    if not isinstance(output_dict, dict):
        raise TypeError("output_dict must be a dict.")
    if not isinstance(tickers, (list, tuple)) or len(tickers) == 0:
        raise TypeError("tickers must be a non-empty list or tuple.")

    def _sanitize_filename_part(x):
        s = str(x).strip()
        s = re.sub(r'[<>:"/\\|?*]+', "", s)
        s = re.sub(r"\s+", "-", s)
        return s

    def _sanitize_sheet_name(x):
        s = str(x).strip()
        s = re.sub(r'[:\\/?*\[\]]+', "", s)
        return s[:31] if len(s) > 31 else s

    ticker_part = "-".join(_sanitize_filename_part(t) for t in tickers)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"SHARPPR_{ticker_part}_{timestamp}.xlsx"

    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, obj in output_dict.items():
            safe_sheet = _sanitize_sheet_name(sheet_name)

            if isinstance(obj, pd.Series):
                obj.to_frame().to_excel(writer, sheet_name=safe_sheet)
            elif isinstance(obj, pd.DataFrame):
                obj.to_excel(writer, sheet_name=safe_sheet)
            else:
                pd.DataFrame({"Value": [obj]}).to_excel(writer, sheet_name=safe_sheet, index=False)

    buffer.seek(0)

    if for_streamlit:
        return buffer.getvalue(), filename

    downloads_dir = Path.home() / "Downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    outpath = downloads_dir / filename

    with open(outpath, "wb") as f:
        f.write(buffer.getvalue())

    return str(outpath)
