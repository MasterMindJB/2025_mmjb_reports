import json

import pytest
from langchain_core.exceptions import OutputParserException

from lib.llm_functions import ConversationSummarizer
from lib.llm_validation import AMASE_KEYS, validate_llm_output
from tests.fakes import BrokenAMASE, FakeLLM


def test_partial_framework_is_rejected():
    with pytest.raises(OutputParserException):
        validate_llm_output(
            'personal_framework_amase',
            {'Assertive': {'value': 4, 'explanation': 'only this type'}},
        )


def test_valid_quotes_are_cleaned():
    cleaned = validate_llm_output(
        'personal_key_contributions',
        [{'tagline': '  Start  ', 'quote': 'Hello'}],
    )
    assert cleaned == [{'tagline': 'Start', 'quote': 'Hello'}]


def test_chain_retries_invalid_amase_output(tmp_path):
    from pathlib import Path

    model = BrokenAMASE()
    summarizer = ConversationSummarizer(tmp_path, Path('prompt_templates'), llm=model)
    with pytest.raises(Exception):
        summarizer.chains_with_retry['personal_framework_amase'].invoke(
            {
                'stats_markdown': 'none',
                'name': 'Alex Rivera',
                'conversation_script': 'Alex said hello.',
            }
        )
    assert len(model.calls) == ConversationSummarizer.stop_after_attempt


def test_chain_accepts_a_complete_framework(tmp_path):
    from pathlib import Path

    model = FakeLLM()
    summarizer = ConversationSummarizer(tmp_path, Path('prompt_templates'), llm=model)
    result = summarizer.chains_with_retry['personal_framework_amase'].invoke(
        {
            'stats_markdown': 'none',
            'name': 'Alex Rivera',
            'conversation_script': 'Return the scores for Alex Rivera.',
        }
    )
    assert set(result) == set(AMASE_KEYS)
    assert result['Assertive']['value'] == 3
    json.dumps(result)
