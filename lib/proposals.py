"""One session call that marks which turns put forward a new suggestion, option, or plan."""

from __future__ import annotations

import json
import logging

PROMPT_MARK = 'PROPOSAL_TURNS'


def in_window_turns(transcript: dict | None) -> list[dict]:
    """Numbered turns already inside the session window. The number is the list index."""
    turns = []
    for row in (transcript or {}).get('utterances') or []:
        text = str(row.get('text') or '').strip()
        if not text:
            continue
        turns.append({
            'speaker': row.get('speaker'),
            'name': row.get('name') or row.get('speaker') or 'Speaker',
            'text': text,
        })
    return turns


def proposal_prompt(turns: list[dict]) -> str:
    lines = [f'{index}. {turn["speaker"]}: {turn["text"]}' for index, turn in enumerate(turns)]
    body = '\n'.join(lines)
    return (
        f'{PROMPT_MARK}\n'
        'One proposal is one speaker turn that puts forward a new suggestion, option, or plan. '
        'Count at most one per turn. Restating someone else\'s earlier proposal does not count.\n'
        'Count these:\n'
        '- "We should try the new plan."\n'
        '- "What if we wait until Monday?"\n'
        '- "You should ask the client first."\n'
        'Do not count these:\n'
        '- "What do you all think?" A question that only asks for opinions is not a proposal.\n'
        '- "I agree with that plan." That restates someone else.\n'
        '- "Could you say more?" That does not offer an option.\n'
        'Return JSON only, with the numbers from the list: {"turns": []}\n'
        f'{body}\n'
    )


def parse_proposal_indices(payload, turn_count: int) -> list[int] | None:
    """None means the model reply cannot be used. Do not guess."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get('turns')
    if not isinstance(raw, list):
        return None
    chosen: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float, str)):
            return None
        try:
            index = int(item)
        except (TypeError, ValueError):
            return None
        if index < 0 or index >= turn_count:
            return None
        if index not in chosen:
            chosen.append(index)
    return chosen


def counts_from_indices(turns: list[dict], indices: list[int]) -> list[tuple[str, int]]:
    """At most one proposal per listed turn. People with none stay at zero."""
    totals: dict[str, int] = {}
    for turn in turns:
        totals.setdefault(str(turn.get('name') or turn.get('speaker') or 'Speaker'), 0)
    for index in indices:
        turn = turns[index]
        name = str(turn.get('name') or turn.get('speaker') or 'Speaker')
        totals[name] = totals.get(name, 0) + 1
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))


def _parse_json(text: str):
    cleaned = str(text or '').strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`')
        cleaned = cleaned.removeprefix('json').strip()
    data = json.loads(cleaned)
    return data


def classify_proposals(llm, transcript: dict | None, progress_dir=None) -> list[tuple[str, int]] | None:
    """One call. On any failure, return None so the chart is left out."""
    turns = in_window_turns(transcript)
    if not turns:
        return []
    try:
        raw = llm.invoke(proposal_prompt(turns))
        content = raw.content if hasattr(raw, 'content') else raw
        if isinstance(content, list):
            content = ' '.join(str(part) for part in content)
        indices = parse_proposal_indices(_parse_json(content), len(turns))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
        logging.warning('proposal call was not usable: %s', exc)
        return None
    except Exception as exc:
        logging.warning('proposal call failed: %s', exc)
        return None
    if indices is None:
        return None
    _audit(progress_dir, indices)
    return counts_from_indices(turns, indices)


def _audit(progress_dir, indices: list[int]) -> None:
    """The raw index list stays in checkpoints. It is not written into the report."""
    if not progress_dir:
        return
    from pathlib import Path

    folder = Path(progress_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'checkpoints.json'
    existing = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            existing = []
    if not isinstance(existing, list):
        existing = []
    existing.append({'section': 'proposals', 'turns': list(indices)})
    path.write_text(json.dumps(existing, indent=2), encoding='utf-8')
