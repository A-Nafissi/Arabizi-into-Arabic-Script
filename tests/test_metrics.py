from evaluation.metrics import clean, score_corpus


def test_clean_collapses_whitespace():
    assert clean("  salam\n 3likom ") == "salam 3likom"


def test_exact_corpus_has_zero_edit_rates():
    result = score_corpus(["السلام عليكم"], ["السلام عليكم"])
    assert result["CER_percent"] == 0
    assert result["WER_percent"] == 0
    assert result["position_wise_exact_percent"] == 100
    assert result["sentence_exact_percent"] == 100


def test_position_metric_retains_legacy_zip_behavior():
    result = score_corpus(["ا ب ج"], ["ا ب"])
    assert result["position_wise_exact_percent"] == 100
    assert result["WER_percent"] > 0
