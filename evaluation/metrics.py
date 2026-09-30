"""Shared corpus metrics with the settings used in the paper."""

from sacrebleu.metrics import BLEU, CHRF

try:
    from rapidfuzz.distance import Levenshtein

    def levenshtein_distance(left, right):
        return Levenshtein.distance(left, right)
except ImportError:  # NLTK is slower but keeps the evaluation portable.
    from nltk.metrics.distance import edit_distance

    def levenshtein_distance(left, right):
        return edit_distance(left, right)


def clean(text):
    return " ".join(str(text).split())


def score_corpus(predictions, references):
    hypotheses = [clean(value) for value in predictions]
    references = [clean(value) for value in references]
    if len(hypotheses) != len(references) or not references:
        raise ValueError("Predictions and references must be non-empty and have equal lengths.")

    bleu = BLEU(tokenize="intl")
    chrf = CHRF(char_order=6, word_order=0, beta=2, whitespace=False)
    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])

    character_edits = 0
    word_edits = 0
    reference_characters = 0
    reference_words = 0
    position_exact = 0
    sentence_exact = 0
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        hyp_words = hypothesis.split()
        ref_words = reference.split()
        character_edits += levenshtein_distance(hypothesis, reference)
        word_edits += levenshtein_distance(hyp_words, ref_words)
        reference_characters += len(reference)
        reference_words += len(ref_words)
        position_exact += sum(a == b for a, b in zip(hyp_words, ref_words))
        sentence_exact += hypothesis == reference

    return {
        "rows": len(references),
        "BLEU": bleu_score.score,
        "chrF": chrf_score.score,
        "CER_percent": 100 * character_edits / reference_characters,
        "WER_percent": 100 * word_edits / reference_words,
        "position_wise_exact_percent": 100 * position_exact / reference_words,
        "sentence_exact_percent": 100 * sentence_exact / len(references),
        "character_edits": character_edits,
        "reference_characters": reference_characters,
        "word_edits": word_edits,
        "reference_words": reference_words,
        **{f"precision_{i + 1}": value for i, value in enumerate(bleu_score.precisions)},
        "BLEU_signature": str(bleu.get_signature()),
        "chrF_signature": str(chrf.get_signature()),
    }
