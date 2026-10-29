from unittest.mock import patch

from twiz.ingest.public_macro import fetch_public_macro, parse_ecb, parse_fred_csv, parse_treasury


def test_treasury_preserves_native_series_and_bounds_history():
    data = b'''<feed xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
    xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"><m:properties>
    <d:NEW_DATE>2026-10-27T00:00:00</d:NEW_DATE><d:BC_10YEAR>4.99</d:BC_10YEAR>
    </m:properties><m:properties><d:NEW_DATE>2026-10-28T00:00:00</d:NEW_DATE>
    <d:BC_10YEAR>9</d:BC_10YEAR></m:properties></feed>'''
    rows = parse_treasury(data, "2026-10-27")
    assert rows[0]["series_id"] == "UST10Y"
    assert rows[0]["native_series_id"] == "BC_10YEAR"
    assert rows[0]["value"] == 4.99 and len(rows[0]["history"]) == 1


def test_csv_missing_values_and_future_rows_are_not_fabricated():
    data = (b'observation_date,BAMLH0A0HYM2,DCOILBRENTEU\n'
            b'2026-10-26,3.3,.\n2026-10-27,.,100\n2026-10-28,8,200\n')
    rows = parse_fred_csv(data, "2026-10-27")
    assert rows[0]["date"] == "2026-10-26" and rows[0]["value"] == 3.3
    assert rows[1]["date"] == "2026-10-27" and rows[1]["value"] == 100


def test_fx_uses_same_date_cross_and_keeps_reference_source():
    data = b'''<Envelope><Cube><Cube time="2026-10-27"><Cube currency="USD" rate="1.2"/>
    <Cube currency="JPY" rate="180"/><Cube currency="KRW" rate="1680"/></Cube>
    <Cube time="2026-10-28"><Cube currency="JPY" rate="1"/></Cube></Cube></Envelope>'''
    rows = parse_ecb(data, "2026-10-27")
    assert rows[0]["value"] == 150 and rows[1]["value"] == 1400
    assert "not live" in rows[0]["method"]


def test_feed_failure_isolated_and_offline_does_not_fetch():
    with patch("twiz.ingest.public_macro._get", side_effect=OSError("down")) as get:
        assert fetch_public_macro("2026-10-27", live=False) == ([], [])
        get.assert_not_called()
        rows, warnings = fetch_public_macro("2026-10-27", live=True)
    assert rows == [] and len(warnings) == 3
