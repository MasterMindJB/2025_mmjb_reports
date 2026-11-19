from pathlib import Path

from lib.pdf_functions import _clean_llm_markdown
from lib.report_stats import extract_bigrams, top_bigram

PROMPT = Path('prompt_templates/personal_signature_phrase.txt')


def test_top_bigram_skips_pure_stopword_pairs():
    text = 'And I said and I went. My life, my life, my life is mine. And I'
    assert top_bigram(text) == ('my life', 3)


def test_top_bigram_keeps_discourse_markers():
    text = 'I think so. I think it works. You know, I think we agree. And I and I and I'
    assert top_bigram(text) == ('i think', 3)


def test_top_bigram_skips_pronoun_plus_a_light_verb():
    from lib.report_stats import display_bigram

    text = 'I got it. I got it. You set the table. The budget needs work. The budget needs work.'
    phrase, count = top_bigram(text)
    assert phrase not in {'i got', 'you set'}
    assert count >= 2
    assert display_bigram('i got') == 'I got'
    assert display_bigram('we should') == 'we should'


def test_top_bigram_falls_back_to_raw_top_and_handles_short_text():
    assert top_bigram('and i and i') == ('and i', 2)
    assert top_bigram('Hello') is None
    assert extract_bigrams("It's done.") == ['its done']


def test_prompt_template_is_verbatim_with_one_placeholder():
    text = PROMPT.read_text(encoding='utf-8').strip()
    assert text.startswith('Act as an expert sociolinguist with a specialization in group facilitation')
    assert '"{bigram}"' in text
    assert text.count('{') == 1
    assert text.endswith('"What?", "So, what?", "Now, what?".')


def test_markdown_cleanup_for_pdf():
    cleaned = _clean_llm_markdown('### Heading\n* item\n---\nA \u2192 B')
    assert cleaned == '**Heading**\n\u2022 item\n\nA -> B'


def test_off_diagonal_handles_non_square_crosstab():
    import pandas as pd

    from lib.report_stats import _off_diagonal

    # C never precedes anyone, so the name_previous crosstab has no C column.
    frame = pd.DataFrame({'name': ['A', 'B', 'A', 'C'], 'prev': [None, 'A', 'B', 'A']})
    table = pd.crosstab(frame['name'], frame['prev'], margins=True)
    assert sorted(_off_diagonal(table)) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_markdown_tables_are_flattened_for_pdf():
    cleaned = _clean_llm_markdown('| Feature | Meaning |\n| :--- | :--- |\n| **Tone** | Warm |')
    assert cleaned == 'Feature \u2014 Meaning\n**Tone** \u2014 Warm'
