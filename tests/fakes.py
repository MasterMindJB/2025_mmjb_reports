"""Offline stand-ins for AssemblyAI and Gemini."""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from lib.llm_validation import AMASE_KEYS, DOSHA_KEYS, ENNEAGRAM_KEYS, ERROR_KEYS


def _message_text(messages) -> str:
    parts = []
    for message in messages:
        content = message.content
        if isinstance(content, list):
            parts.append(' '.join(str(item) for item in content))
        else:
            parts.append(str(content))
    return '\n'.join(parts)


def _scores(keys, token: str) -> str:
    payload = {
        key: {'value': 3, 'explanation': f'{token} {key}'}
        for key in keys
    }
    return json.dumps(payload)


def _who(text: str) -> str:
    if 'onboarding plan' in text:
        return 'ALEX'
    if 'budget first' in text:
        return 'BLAKE'
    if 'draft the timeline' in text:
        return 'CASEY'
    return 'UNKNOWN'


class FakeLLM(BaseChatModel):
    calls: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return 'fake-hts'

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = _message_text(messages)
        self.calls.append(text)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._reply(text)))])

    def _reply(self, text: str) -> str:
        if 'BATCH:' in text and 'FACTS' in text:
            return _batch_reply(text)
        if 'Section:' in text and 'FACTS' in text:
            return _story_reply(text)
        who = _who(text)
        if 'Friction Factor' in text:
            return json.dumps({key: [] for key in ERROR_KEYS})
        if 'framework called AMASE' in text:
            return _scores(AMASE_KEYS, f'AMASE_TOKEN_{_subject(text)}')
        if 'Enneagram framework' in text:
            return _scores(ENNEAGRAM_KEYS, f'ENNEAGRAM_TOKEN_{_subject(text)}')
        if 'Doshas framework' in text:
            return _scores(DOSHA_KEYS, f'DOSHA_TOKEN_{_subject(text)}')
        if 'verbatim quotes' in text:
            return json.dumps([{'tagline': f'QUOTE_TAG_{who}', 'quote': 'A short synthetic quote.'}])
        if 'explicit and implicit' in text:
            return json.dumps([{'title': f'ACTION_TITLE_{who}', 'action': f'ACTION_BODY_{who}'}])
        if 'Summarize in one paragraph the following transcript' in text:
            return f'PERSONAL_SUMMARY_TOKEN_{who}'
        if 'super power' in text:
            return f'SUPERPOWER_TOKEN_{_subject(text)}'
        if 'expert sociolinguist' in text and 'verbal tic' in text:
            return json.dumps(
                {
                    'what': 'SIGNATURE_PHRASE_TOKEN You reach for a familiar phrase.',
                    'so_what': 'It buys you a second.',
                    'now_what': 'Ask before the next turn.',
                }
            )
        if 'give a title' in text:
            return 'GROUP_TITLE_TOKEN'
        if 'one paragraph summarizing the following conversation' in text:
            return 'GROUP_SHORT_TOKEN'
        if 'summarizing recount' in text:
            return 'GROUP_LONG_TOKEN'
        raise AssertionError('FakeLLM saw an unexpected prompt')


def _fact(text: str, key: str) -> str:
    import re

    match = re.search(rf'key={key} value=(\d+)', text)
    if not match:
        raise AssertionError(f'fact {key} missing')
    return match.group(1)


def _own_quotes(text: str):
    import re

    return re.findall(r'ts=(\d+:\d+) text=(.+)', text.split('NEXT SPEAKER')[0])


def _batch_reply(text: str) -> str:
    import json
    import re

    audience = 'group' if 'BATCH: group' in text else 'personal'
    sections = (
        ('character', 'goal', 'challenge', 'drama', 'resolution', 'one_thing')
    )
    payload = {}
    for section in sections:
        payload[section] = json.loads(_story_reply(f'Section: {section}\n{text}'))
    return json.dumps(payload)


def _story_reply(text: str) -> str:
    """Structured fields only. Digits stay in target_value, not in the prose."""
    import re

    section = re.search(r'Section: (\w+)', text).group(1)
    audience = 'group' if ('Audience: group' in text or 'BATCH: group' in text) else 'personal'
    if audience == 'group':
        target = int(_fact(text, 'target_interaction'))
        if section == 'character':
            payload = {
                'headline': 'A serial panel',
                'number_keys': ['participation', 'interaction', 'word_interaction'],
            }
        elif section == 'goal':
            payload = {'claim': 'the onboarding plan'}
        elif section == 'challenge':
            payload = {'claim': 'the center of the room', 'fact_key': 'interaction'}
        elif section == 'drama':
            payload = {'pattern_note': 'Who talks to whom, and who nobody answered.'}
        elif section == 'resolution':
            payload = {'claim': 'the onboarding plan'}
        else:
            payload = {
                'verb': 'ask',
                'verb_phrase': 'one question back',
                'trigger': 'before the next decision',
                'metric_key': 'interaction',
                'target_value': target,
            }
        return json.dumps(payload)

    quotes = _own_quotes(text)
    target = int(_fact(text, 'target_questions'))
    first_when, first_line = quotes[0]
    last_when, last_line = quotes[-1]
    claim = ' '.join(first_line.replace('?', ' ').replace('.', ' ').split()[:8])
    if section == 'character':
        payload = {
            'headline': claim,
            'number_keys': ['talk_share', 'questions', 'turns'],
        }
    elif section == 'goal':
        payload = {'claim': claim, 'quote': {'text': first_line, 'ts': first_when}}
    elif section == 'challenge':
        payload = {'claim': claim, 'fact_key': 'questions'}
    elif section == 'drama':
        payload = {
            'patterns': [
                {
                    'quotes': [{'text': first_line, 'ts': first_when}],
                    'pattern': 'You use this line as you open',
                    'twist': 'the next speaker took the next turn',
                    'cost': 'a question',
                    'gift': 'a start',
                }
            ],
            'exception': {
                'quote': last_line,
                'ts': last_when,
                'what_happened': ' '.join(last_line.replace('.', ' ').split()[-4:]),
            },
        }
    elif section == 'resolution':
        payload = {'claim': claim, 'quote': {'text': last_line, 'ts': last_when}}
    else:
        payload = {
            'verb': 'ask',
            'verb_phrase': 'one question',
            'trigger': 'before your second turn',
            'metric_key': 'questions',
            'target_value': target,
        }
    return json.dumps(payload)


class FlakyStoryLLM(FakeLLM):
    """First story call is bad JSON so the retry path has to ask again."""

    remaining_bad: int = 1

    def _reply(self, text: str) -> str:
        if self.remaining_bad and 'BATCH:' in text and 'FACTS' in text:
            self.remaining_bad -= 1
            return 'not-json'
        return super()._reply(text)


def _subject(text: str) -> str:
    for name in ('Alex Rivera', 'Blake Chen', 'Casey Quinn'):
        token = name.replace(' ', '_')
        if f'scores for {name}' in text or f'super power for **{name}**' in text:
            return token
    return 'UNKNOWN'


class BrokenAMASE(FakeLLM):
    def _reply(self, text: str) -> str:
        if 'framework called AMASE' in text:
            return json.dumps({'Assertive': {'value': 1, 'explanation': 'only one key'}})
        return super()._reply(text)


class FakeTranscription:
    def __init__(self, payload: dict):
        self.payload = payload
        self.deleted: list[str] = []

    def transcribe(self, audio_path, dest_dir, speakers_expected):
        dest_dir.mkdir(parents=True, exist_ok=True)
        return json.loads(json.dumps(self.payload))

    def delete_transcript(self, transcript_id: str) -> None:
        self.deleted.append(transcript_id)
