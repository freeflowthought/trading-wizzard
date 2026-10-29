from twiz.reporting import _render_index_exposure, _render_regional_transmission


def test_regional_table_separates_unusable_probability_and_observed_transmission():
    scenario = {"regional_transmission": [
        {"region": "日本", "event_quality": "usable",
         "event_semantics": "维持利率为主情景（98.5%）",
         "fx": {"series_id": "USDJPY", "date": "2026-10-27", "change_5_pct": -2.1},
         "semiconductor": {"end_date": "2026-10-27", "vs_spy_5d_pp": -2.3},
         "action": "日元与半导体传导同时确认，降低首批上限"},
        {"region": "韩国", "event_quality": "unavailable",
         "event_semantics": "加息概率84.5%", "action": "低流动性，仅观察"},
    ]}
    text = "\n".join(_render_regional_transmission(scenario))
    assert "美国利率、信用与直接政策为核心" in text
    assert "98.5%" in text and "84.5%" not in text
    assert "-2.30个百分点" in text and "本报告日期无通过校验" in text
    assert "尚非回测校准的数值权重" in text


def test_absent_regional_context_does_not_invent_current_probabilities():
    assert _render_regional_transmission({}) == []


def test_branch_probability_labels_outcomes_and_jpy_strength_uses_inverse_fx():
    row = {"region": "Japan", "event_quality": "usable", "event_main_case": "分支",
           "event_semantics": "保留分支", "event_probability": .7,
           "event_yes_probability": .3, "event_no_probability": .7,
           "fx": {"series_id": "USDJPY", "change_5_pct": -1.9608,
                  "local_currency_strength_5_pct": 2, "date": "2026-10-27"}}
    text = "\n".join(_render_regional_transmission({"regional_transmission": [row]}))
    assert "Yes 30.0% / No 70.0%" in text
    assert "对应日元价值+2.00%" in text
    assert "保留分支（70.0%）" not in text


def test_index_exposure_maps_issuer_weights_without_substituting_smh_return():
    row = {"ticker": "QQQ", "weight_pct": 28, "holdings_date": "2026-10-26",
           "coverage": "complete", "source_url": "https://www.invesco.com/qqq-etf/en/about.html"}
    text = "\n".join(_render_index_exposure({"semiconductor_exposures": [row]}, "2026-10-27"))
    assert "28.00%" in text and "约-1.40个百分点" in text
    assert "其余成分不变" in text and "不直接用SMH涨跌代替" in text
    for override in ({"holdings_date": "2026-10-28"}, {"holdings_date": "2026-10-01"},
                     {"coverage": "partial"}, {"weight_pct": float("nan")}):
        assert _render_index_exposure({"semiconductor_exposures": [{**row, **override}]},
                                      "2026-10-27") == []
