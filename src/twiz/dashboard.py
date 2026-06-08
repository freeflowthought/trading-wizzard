import sys
from pathlib import Path

if __package__ in {None, ""}:  # Streamlit runs this file outside package context.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from twiz.config import load_watchlist_config
    from twiz.reporting import build_report_bundle, render_markdown
    from twiz.storage import ProjectPaths
else:
    from .config import load_watchlist_config
    from .reporting import build_report_bundle, render_markdown
    from .storage import ProjectPaths


def latest_report_date(paths: ProjectPaths) -> str:
    reports = sorted(paths.reports.glob("*-report.md"))
    if reports:
        return reports[-1].name.replace("-report.md", "")
    processed = sorted(paths.processed.glob("technical-*.json"))
    if processed:
        return processed[-1].name.replace("technical-", "").replace(".json", "")
    return ""


def main() -> None:
    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover - visible only in dashboard runtime
        raise RuntimeError("Streamlit is required for the dashboard. Run `uv sync`.") from exc

    paths = ProjectPaths()
    paths.ensure()
    default_date = latest_report_date(paths)
    st.set_page_config(page_title="Trading Wizzard", layout="wide")
    st.title("Trading Wizzard")
    report_date = st.text_input("Report date", value=default_date)
    if not report_date:
        st.info("Run `twiz report --date today` to generate the first report.")
        return

    bundle = build_report_bundle(paths, report_date, load_watchlist_config()["watchlist"])
    st.subheader("Watchlist")
    st.dataframe(bundle.watchlist_table, width="stretch")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Report")
        st.markdown(render_markdown(bundle))
    with right:
        st.subheader("Risk Flags")
        if bundle.risk_flags:
            for warning in bundle.risk_flags:
                st.warning(warning)
        else:
            st.success("No pipeline warnings.")

    st.subheader("Charts")
    for ticker, setup in bundle.technical_setup.items():
        chart_path = setup.get("chart_path")
        if not chart_path or not Path(chart_path).exists():
            continue
        st.caption(ticker)
        try:
            import json

            payload = json.loads(Path(chart_path).read_text(encoding="utf-8"))
            st.line_chart({ticker: [row["close"] for row in payload.get("rows", [])]})
        except Exception as exc:
            st.warning(f"Could not render chart for {ticker}: {exc}")


if __name__ == "__main__":
    main()
