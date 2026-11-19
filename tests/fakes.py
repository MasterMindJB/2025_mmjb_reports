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
        if 'Section:' in text and 'TARGET:' in text:
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
            return '**What?** SIGNATURE_PHRASE_TOKEN\n* So, what? A bullet.\n### Now, what?\nTry it.'
        if 'give a title' in text:
            return 'GROUP_TITLE_TOKEN'
        if 'one paragraph summarizing the following conversation' in text:
            return 'GROUP_SHORT_TOKEN'
        if 'summarizing recount' in text:
            return 'GROUP_LONG_TOKEN'
        raise AssertionError('FakeLLM saw an unexpected prompt')


def _story_reply(text: str) -> str:
    """A craft-safe draft. Numbers and quotes are copied from the evidence packet."""
    import re

    section = re.search(r'Section: (\w+)', text).group(1)
    audience = 'group' if 'Audience: group' in text else 'personal'
    transitions = {
        'character': "That's who showed up. Here's what they came to do.",
        'goal': "It didn't go to plan, and not only because of you.",
        'challenge': 'Now the interesting part.',
        'drama': 'So how did it end?',
        'resolution': 'If you change one thing next time, make it this.',
    }
    quotes = re.findall(r'QUOTE: (\d+:\d+) "([^"]+)"', text)
    nxt = re.findall(r'NEXT: (\d+:\d+) "([^"]+)"', text)
    if audience == 'group':
        interaction = re.search(r'INTERACTION: (\d+)', text).group(1)
        participation = re.search(r'PARTICIPATION: (\d+)', text).group(1)
        target = re.search(r'TARGET: (\d+)', text).group(1)
        if section == 'character':
            payload = {
                'headline': f'A serial panel at {interaction}% interaction.',
                'body': f'{participation}% participation, against the rest of the exchange.',
                'transition': transitions['character'],
            }
        elif section == 'one_thing':
            payload = {
                'one_thing': (
                    f'Ask one question back before the next decision. Target: {target}% interaction.'
                ),
                'why': f'Interaction was {interaction}%, so {target}% is the halfway step.',
            }
        elif section == 'challenge':
            payload = {
                'body': f'Interaction was {interaction}%.',
                'transition': transitions['challenge'],
            }
        elif section == 'drama':
            payload = {
                'body': 'Who talks to whom, and who nobody answered.',
                'transition': transitions['drama'],
            }
        elif section == 'goal':
            payload = {
                'body': 'Inferred from the session, the room came to settle a plan.',
                'transition': transitions['goal'],
            }
        else:
            payload = {
                'body': 'The room landed on the next step.',
                'transition': transitions['resolution'],
            }
        return json.dumps(payload)

    share = re.search(r'TALK_SHARE: (\d+)', text).group(1)
    questions = re.search(r'QUESTION_COUNT: (\d+)', text).group(1)
    target = re.search(r'TARGET: (\d+)', text).group(1)
    before = re.search(r'BEFORE_AFTER: (.+)', text)
    if section == 'character':
        payload = {
            'headline': 'Opened on the plan',
            'body': f'{share}% talk time: you held the floor most of the way, more than the rest of the room.',
            'transition': transitions['character'],
        }
    elif section == 'one_thing':
        payload = {
            'one_thing': f'Ask one question before you offer the plan. Target: {target} questions.',
            'why': f'You asked {questions} questions, so {target} is the halfway step.',
        }
    elif section == 'goal' and quotes:
        when, line = quotes[0]
        payload = {
            'body': f'Inferred from what you said: at {when} you said "{line}".',
            'transition': transitions['goal'],
        }
    elif section == 'challenge':
        payload = {
            'body': before.group(1).strip() if before else f'You asked {questions} questions.',
            'transition': transitions['challenge'],
        }
    elif section == 'drama' and quotes:
        when, line = quotes[0]
        follow = f' The next speaker said "{nxt[0][1]}".' if nxt else ''
        hedge = ''
        if 'might' not in line.lower() and (not nxt or 'might' not in nxt[0][1].lower()):
            hedge = ' You might already know this about yourself: this phrase often reads as a start.'
        close = ''
        if len(quotes) > 1:
            close = f' The one moment it didn\'t happen was {quotes[-1][0]}, and that\'s when you got specific.'
        payload = {
            'body': f'At {when} you said "{line}".{hedge}{follow} It buys a start. It also costs the question.{close}',
            'transition': transitions['drama'],
        }
    elif section == 'resolution' and quotes:
        when, line = quotes[-1]
        payload = {
            'body': f'At {when} you said "{line}".',
            'transition': transitions['resolution'],
        }
    else:
        payload = {'body': 'You came in on the plan.', 'transition': transitions.get(section, transitions['goal'])}
    return json.dumps(payload)


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
