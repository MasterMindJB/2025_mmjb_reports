"""The simple personal bullets: four checks, one retry, no filler."""

import json

from lib.simple_bullets import bullet_problems, generate_simple_bullets, passing_sentences


SPEECH = (
    'The budget can wait until the client signs and the room has heard the point. '
    'I keep coming back to the work that has to be honest before it goes out.'
)


def _good():
    return {
        's1': 'You seem to care about honest work, and it seemed like you were trying to show the point.',
        's2': 'When you said "the budget can wait until the client signs", it felt like you wanted the work to be ready.',
        's3': 'As a point of curiosity, I wonder how your focus on the work affects how the group decides.',
    }


def test_the_four_checks_accept_a_clean_draft_and_reject_each_miss():
    clean = list(_good().values())
    assert bullet_problems(clean, SPEECH) == []

    bad_quote = dict(_good())
    bad_quote['s2'] = 'When you said "the weather stayed cold all morning", it felt like you wanted a pause.'
    assert any('verbatim' in reason for reason in bullet_problems(list(bad_quote.values()), SPEECH))

    bad_open = dict(_good())
    bad_open['s1'] = 'They seem to care about honest work, and it seemed like you were trying to show the point.'
    assert any('You seem to care about' in reason for reason in bullet_problems(list(bad_open.values()), SPEECH))

    twice = dict(_good())
    twice['s1'] = 'You seem to care about honest work, and I wonder if it seemed like you were trying to show it.'
    assert any('I wonder' in reason for reason in bullet_problems(list(twice.values()), SPEECH))

    advice = dict(_good())
    advice['s3'] = 'As a point of curiosity, I wonder how your focus on the work affects what you should do next time.'
    reasons = bullet_problems(list(advice.values()), SPEECH)
    assert any('should' in reason for reason in reasons)
    assert any('next time' in reason for reason in reasons)

    kept = passing_sentences(list(bad_quote.values()), SPEECH)
    assert len(kept) == 2
    assert all('weather' not in line for line in kept)


def test_a_failed_draft_is_retried_once_and_only_passing_sentences_ship():
    prompts = []

    class Client:
        model = 'gemini-3.1-flash-lite'

        def invoke(self, prompt, **kwargs):
            prompts.append((self.model, prompt, kwargs.get('generationConfig') or {}))
            if len(prompts) == 1:
                bad = _good()
                bad['s2'] = 'When you said "nobody said this sentence today", it felt like you wanted credit.'
                bad['s3'] = 'As a point of curiosity, I wonder how your focus on the work affects what you should try to do.'
                return type('Reply', (), {'content': json.dumps(bad)})()
            assert 'verbatim' in prompt
            assert 'should' in prompt
            return type('Reply', (), {'content': json.dumps(_good())})()

    bullets = generate_simple_bullets(Client(), {'utterances': []}, 'A', 'Alex', SPEECH)
    assert len(prompts) == 2
    assert '3.6' in prompts[0][0]
    assert prompts[0][2].get('maxOutputTokens') == 1024
    assert prompts[0][2].get('responseMimeType') == 'application/json'
    assert bullets == list(_good().values())
    assert 'SIMPLE_BULLETS' in prompts[0][1]


def test_a_second_failure_ships_the_passing_sentences_only():
    class Client:
        model = 'gemini-3.1-flash-lite'

        def invoke(self, prompt, **kwargs):
            bad = _good()
            bad['s2'] = 'When you said "nobody said this sentence today", it felt like you wanted a pause.'
            return type('Reply', (), {'content': json.dumps(bad)})()

    bullets = generate_simple_bullets(Client(), {'utterances': []}, 'A', 'Alex', SPEECH)
    assert len(bullets) == 2
    assert bullets[0].startswith('You seem to care about')
    assert bullets[1].startswith('As a point of curiosity, I wonder how')
    assert all('nobody said' not in line for line in bullets)
