"""Load the storytelling rules and build one prompt per report section."""

from __future__ import annotations

from pathlib import Path

RULES_PATH = Path(__file__).resolve().parents[1] / 'prompt_templates' / 'story_craft_rules.md'
PROMPT_DIR = RULES_PATH.parent

# Which numbered blocks from story_craft_rules.md each section receives.
SECTION_BLOCKS = {
    'character': ('0', '1', '3', '5', '6'),
    'goal': ('0', '3', '5'),
    'challenge': ('0', '3', '5', '6'),
    'drama': ('0', '2', '3', '5'),
    'resolution': ('0', '3', '5'),
    'one_thing': ('0', '4', '5', '6'),
    'group_drama': ('0', '2', '3', '5', '6'),
}


def rules_text() -> str:
    return RULES_PATH.read_text(encoding='utf-8').strip()


def rule_blocks() -> dict[str, str]:
    """Split the rules file on its numbered headings. The heading stays with the block."""
    text = rules_text()
    chunks = []
    current: list[str] = []
    number = None
    for line in text.splitlines(keepends=True):
        if line.startswith('## '):
            if number is not None:
                chunks.append((number, ''.join(current).strip()))
            number = line[3:].split('.', 1)[0].strip()
            current = [line]
        else:
            current.append(line)
    if number is not None:
        chunks.append((number, ''.join(current).strip()))
    return dict(chunks)


def craft_for(section: str) -> str:
    """The verbatim blocks this section's prompt is allowed to see."""
    key = section
    blocks = rule_blocks()
    numbers = SECTION_BLOCKS[key]
    return '\n\n'.join(blocks[number] for number in numbers)


def render_section_prompt(*, audience: str, section: str, evidence: str) -> str:
    """Fill one section template with its craft blocks and the computed evidence."""
    craft_key = 'group_drama' if audience == 'group' and section == 'drama' else section
    name = f'story_{audience}_{section}.txt'
    template = (PROMPT_DIR / name).read_text(encoding='utf-8')
    craft = craft_for(craft_key)
    filled = (
        template.replace('{craft}', craft)
        .replace('{evidence}', evidence)
        .replace('{section}', section)
    )
    if audience == 'personal':
        filled += (
            '\n\nVoice: write to the reader as you. '
            'Do not say the speaker, the student, or the participant. '
            'Do not use he, she, or they for the reader, and do not capitalize He, She, or They mid-sentence. '
            'A quote must copy the turn exactly, including its original capitalization. '
            'Banned words include inflections such as leveraging, leveraged, and empowering.\n'
        )
    return filled
