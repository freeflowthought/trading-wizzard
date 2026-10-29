from twiz.learning import parse_timeline, study_corpus


def test_relative_dates_and_self_reported_wins_are_not_forecast_scores():
    text = ("canEagle [量化交易], — 2026-07-13, 2:53 PM\n"
            "到了20MV先止盈一半\n"
            "canEagle [量化交易], — Yesterday at 9:00 AM\n成功预测，赚了15%\n")
    rows = parse_timeline(text, "source")
    assert rows[0]["date"] == "2026-07-13"
    assert rows[1]["date"] is None
    assert rows[1]["score_eligible"] is False
    assert "self_reported_result" in rows[1]["tags"]
    assert "20MV" in rows[0]["text"]


def test_overlapping_exports_do_not_inflate_evidence(tmp_path):
    text = "canEagle [量化交易], — 2026-07-13, 2:53 PM\n先止盈一半\n"
    first, second = tmp_path/"one.txt", tmp_path/"two.txt"
    first.write_text(text)
    second.write_text(text+"Image\n")
    result = study_corpus([first, second])
    assert result["coverage"]["unique_posts"] == 1
    assert result["coverage"]["duplicates_removed"] == 1
    assert result["messages"][0]["also_in"]
