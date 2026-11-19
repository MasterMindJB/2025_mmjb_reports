"""Reject LLM payloads that do not match the prompt contracts.

Invalid payloads raise OutputParserException so the existing retry wrapper can
ask again. A partial personality object is not a successful analysis.
"""

from langchain_core.exceptions import OutputParserException

AMASE_KEYS = ('Assertive', 'Mirror', 'Audience', 'Sharpshooter', 'Expert')
ENNEAGRAM_KEYS = (
    'Type 1: The Reformer',
    'Type 2: The Helper',
    'Type 3: The Achiever',
    'Type 4: The Individualist',
    'Type 5: The Investigator',
    'Type 6: The Loyalist',
    'Type 7: The Enthusiast',
    'Type 8: The Challenger',
    'Type 9: The Peacemaker',
)
DOSHA_KEYS = ('Vata', 'Pitta', 'Kapha')
ERROR_KEYS = (
    'personal attacks',
    'contempt',
    'meanness',
    'lack of empathy',
    'dismissal',
    'rudeness',
    'interruptions',
    'criticism',
    'unwilling to compromise',
    'not paying attention',
    'unconnected advice',
    'absolutism',
    'assuming clarity',
    'not good questions',
    'generic responses',
    'talking too much',
    'universal statements',
    'obvious statements',
)
TEXT_OUTPUTS = (
    'personal_summary',
    'personal_signature_phrase',
    'personal_super_power',
    'conversation_summary_long',
    'conversation_summary_short',
    'conversation_title',
)


def _fail(message: str) -> None:
    raise OutputParserException(message)


def _score_map(data, keys: tuple[str, ...], label: str) -> dict:
    if not isinstance(data, dict):
        _fail(f'{label} must be a JSON object')
    missing = [key for key in keys if key not in data]
    extra = [key for key in data if key not in keys]
    if missing or extra:
        _fail(f'{label} keys mismatch; missing={missing}; unexpected={extra}')
    cleaned = {}
    for key in keys:
        item = data[key]
        if not isinstance(item, dict) or 'value' not in item or 'explanation' not in item:
            _fail(f'{label}.{key} must contain value and explanation')
        value = item['value']
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(f'{label}.{key}.value must be a number from 0 to 5')
        if isinstance(value, float) and not value.is_integer():
            _fail(f'{label}.{key}.value must be a whole number from 0 to 5')
        whole = int(value)
        if whole < 0 or whole > 5:
            _fail(f'{label}.{key}.value must be from 0 to 5')
        explanation = item['explanation']
        if not isinstance(explanation, str):
            _fail(f'{label}.{key}.explanation must be a string')
        cleaned[key] = {'value': whole, 'explanation': explanation}
    return cleaned


def _object_list(data, label: str, fields: tuple[str, ...]) -> list:
    if not isinstance(data, list):
        _fail(f'{label} must be a JSON array')
    cleaned = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            _fail(f'{label}[{index}] must be an object')
        cleaned_item = {}
        for field in fields:
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                _fail(f'{label}[{index}].{field} must be a non-empty string')
            cleaned_item[field] = value.strip()
        cleaned.append(cleaned_item)
    return cleaned


def _errors(data) -> dict:
    label = 'personal_errors_and_corrections'
    if not isinstance(data, dict):
        _fail(f'{label} must be a JSON object')
    missing = [key for key in ERROR_KEYS if key not in data]
    extra = [key for key in data if key not in ERROR_KEYS]
    if missing or extra:
        _fail(f'{label} keys mismatch; missing={missing}; unexpected={extra}')
    cleaned = {}
    for key in ERROR_KEYS:
        items = data[key]
        if not isinstance(items, list):
            _fail(f'{label}.{key} must be an array')
        cleaned_items = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                _fail(f'{label}.{key}[{index}] must be an object')
            factor = item.get('friction_factor')
            observation = item.get('observation')
            advice = item.get('advice', '')
            if isinstance(factor, bool) or not isinstance(factor, (int, float)):
                _fail(f'{label}.{key}[{index}].friction_factor must be 0 to 5')
            if isinstance(factor, float) and not factor.is_integer():
                _fail(f'{label}.{key}[{index}].friction_factor must be a whole number')
            whole = int(factor)
            if whole < 0 or whole > 5:
                _fail(f'{label}.{key}[{index}].friction_factor must be 0 to 5')
            if not isinstance(observation, str) or not observation.strip():
                _fail(f'{label}.{key}[{index}].observation must be a non-empty string')
            if not isinstance(advice, str):
                _fail(f'{label}.{key}[{index}].advice must be a string')
            cleaned_items.append(
                {
                    'friction_factor': whole,
                    'observation': observation.strip(),
                    'advice': advice.strip(),
                }
            )
        cleaned[key] = cleaned_items
    return cleaned


def validate_llm_output(name: str, data):
    """Return cleaned output, or raise OutputParserException."""
    if name == 'personal_key_contributions':
        return _object_list(data, name, ('tagline', 'quote'))
    if name == 'personal_action_items':
        return _object_list(data, name, ('title', 'action'))
    if name == 'personal_framework_amase':
        return _score_map(data, AMASE_KEYS, name)
    if name == 'personal_framework_enneagram':
        return _score_map(data, ENNEAGRAM_KEYS, name)
    if name == 'personal_framework_doshas':
        return _score_map(data, DOSHA_KEYS, name)
    if name == 'personal_errors_and_corrections':
        return _errors(data)
    if name in TEXT_OUTPUTS:
        if not isinstance(data, str) or not data.strip():
            _fail(f'{name} must be a non-empty string')
        return data.strip()
    _fail(f'No validator is registered for {name}')
