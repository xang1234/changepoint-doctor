import cpd
import changepoint_doctor as cpd_alias


def test_alias_import_matches_canonical_surface() -> None:
    assert cpd_alias.__version__ == cpd.__version__

    for name in [
        "Pelt",
        "Binseg",
        "Fpop",
        "Bocpd",
        "Cusum",
        "PageHinkley",
        "detect_offline",
        "OfflineChangePointResult",
    ]:
        assert getattr(cpd_alias, name) is getattr(cpd, name)


def test_alias_import_supports_minimal_detection_path() -> None:
    values = [0.0] * 20 + [4.0] * 20

    result = cpd_alias.Pelt(model="l2", min_segment_len=2).fit(values).predict(n_bkps=1)
    assert result.breakpoints == [20, 40]
