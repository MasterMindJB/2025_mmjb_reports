"""Synthetic session used by offline tests. No real participant data."""

UTTERANCES = [
    {
        'speaker': 'A',
        'text': 'We should try the new onboarding plan.',
        'start': 0,
        'end': 4000,
        'confidence': 0.94,
    },
    {
        'speaker': 'B',
        'text': 'Could we check the budget first?',
        'start': 4000,
        'end': 8000,
        'confidence': 0.93,
    },
    {
        'speaker': 'C',
        'text': 'I will draft the timeline.',
        'start': 8000,
        'end': 11000,
        'confidence': 0.91,
    },
    {
        'speaker': 'A',
        'text': 'That might work if we start Monday.',
        'start': 11000,
        'end': 15000,
        'confidence': 0.92,
    },
    {
        'speaker': 'B',
        'text': 'I can own the numbers from here.',
        'start': 15000,
        'end': 18000,
        'confidence': 0.95,
    },
]


def synthetic_transcript() -> dict:
    sentiment = []
    for utterance in UTTERANCES:
        sentiment.append(
            {
                'speaker': utterance['speaker'],
                'text': utterance['text'],
                'start': utterance['start'],
                'end': utterance['end'],
                'sentiment': 'POSITIVE' if utterance['speaker'] != 'B' else 'NEUTRAL',
                'confidence': 0.8,
            }
        )
    return {
        'id': 'synthetic-transcript-001',
        'status': 'completed',
        'speakers_expected': 3,
        'utterances': [dict(item) for item in UTTERANCES],
        'sentiment_analysis_results': sentiment,
    }
