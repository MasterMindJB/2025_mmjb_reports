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
            return '**What?** SIGNATURE_PHRASE_TOKEN\n* So, what? A bullet.\n### Now, what?\nTry it.'
        if 'give a title' in text:
            return 'GROUP_TITLE_TOKEN'
        if 'one paragraph summarizing the following conversation' in text:
            return 'GROUP_SHORT_TOKEN'
        if 'summarizing recount' in text:
            return 'GROUP_LONG_TOKEN'
        raise AssertionError('FakeLLM saw an unexpected prompt')


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
