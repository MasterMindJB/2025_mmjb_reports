"""Session statistics and charts.

The formulas are the ones in `report_generation.ipynb` (speaker totals, type-token
ratio, modal-verb count, turn transitions, and 1 - Gini). They live here so the
web app can run them without opening the notebook. Metric definitions are unchanged.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import colorsys

import matplotlib

matplotlib.use('Agg', force=False)

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import PIL.Image
import seaborn as sns

from lib.chord_diagram import directed_follows, save_chord_diagram, visible_chord_pairs
from lib.helper_functions import barplot, figure_to_image, gini

NICE_STATS_COL_NAME = {
    'action_items_count': 'Action Items',
    'idea_count': 'Idea Generation',
    'question_count': 'Questions Asked',
    'talk_time_pct': 'Speaking Time (%)',
    'interaction_count': 'Total Interactions',
    'unique_words_pct': 'Unique Words (%) (Idea Density)',
    'word_count': 'Word Count',
    'words_per_min': 'Words Per Minute (WPM)',
}

NICE_STATS_COL_APPLY_OP = {
    'action_items_count': lambda x: int(x),
    'idea_count': lambda x: int(x),
    'question_count': lambda x: int(x),
    'talk_time_pct': lambda x: round(x, 1),
    'interaction_count': lambda x: int(x),
    'unique_words_pct': lambda x: round(x, 1),
    'word_count': lambda x: int(x),
    'words_per_min': lambda x: round(x, 1),
}


@dataclass
class SessionTables:
    df_utter: pd.DataFrame
    df_speaker_stats: pd.DataFrame
    df_nice_stats_data: pd.DataFrame
    conversation_script: str
    gini_reverse_interventions: float
    gini_reverse_interactions: float
    gini_reverse_interactions_both_wrd_cnt: float
    interactions_prev: pd.DataFrame


@dataclass
class ChartImages:
    barplots: list[PIL.Image.Image]
    convo_sentiment: PIL.Image.Image
    speaker_sentiment: PIL.Image.Image
    convo_wpm: PIL.Image.Image
    speaker_wpm: PIL.Image.Image
    action_items: PIL.Image.Image
    chord_path: Path
    matrix_path: Path
    chord_pairs: list


def idea_count(text: str) -> int:
    tokens = nltk.word_tokenize(text)
    return sum(1 for _word, tag in nltk.pos_tag(tokens) if tag == 'MD')


def _unique_words(row: pd.Series) -> pd.Series:
    words = str(row['text']).split()
    unique = set(words)
    row['unique_words_count'] = len(unique)
    row['unique_words_pct'] = (100 * len(unique) / len(words)) if words else 0
    return row


def _one_minus_gini(values) -> float:
    cleaned = [float(value) for value in values if pd.notna(value)]
    if not cleaned or sum(cleaned) == 0:
        return 0.0
    score = 1 - gini(cleaned)
    if math.isnan(score):
        return 0.0
    return float(score)


def _off_diagonal(frame_with_margins: pd.DataFrame) -> list:
    if frame_with_margins.shape[0] < 2 or frame_with_margins.shape[1] < 2:
        return []
    core = frame_with_margins.iloc[:-1, :-1]
    # The crosstab is not square when a speaker never follows anyone, or is never
    # followed. Align rows and columns on the union of speakers. A square table is unchanged.
    labels = list(dict.fromkeys(list(core.index) + list(core.columns)))
    core = core.reindex(index=labels, columns=labels, fill_value=0).astype('float')
    for index in range(core.shape[0]):
        core.iat[index, index] = float('nan')
    return core.melt()['value'].dropna().tolist()


def _blank_figure(title: str, figsize=(10, 2)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.axis('off')
    fig.tight_layout()
    return fig


def _as_image(fig) -> PIL.Image.Image:
    image = figure_to_image(fig)
    plt.close(fig)
    return image


def compute_session_tables(transcript: dict, speaker_map: dict) -> SessionTables:
    utterances = [
        item for item in (transcript.get('utterances') or []) if item.get('speaker') not in (None, 'UNK')
    ]
    if not utterances:
        raise ValueError('The transcript has no speaker utterances.')
    df_utter = pd.DataFrame.from_dict(utterances)
    unknown = sorted(set(df_utter['speaker']) - set(speaker_map))
    if unknown:
        raise ValueError(f"Speakers are missing names: {', '.join(unknown)}")
    df_utter['word_count'] = df_utter['text'].apply(lambda text: len(str(text).split(' ')))
    df_utter['target'] = df_utter['speaker'].shift(-1)
    df_utter['previous'] = df_utter['speaker'].shift(1)
    duration = (df_utter['end'] - df_utter['start']) / 1000
    df_utter['talk_time_sec'] = duration.where(duration > 0, 0)
    df_utter['words_per_min'] = 0.0
    speaking = df_utter['talk_time_sec'] > 0
    df_utter.loc[speaking, 'words_per_min'] = (
        df_utter.loc[speaking, 'word_count'] / df_utter.loc[speaking, 'talk_time_sec'] * 60
    )

    sentiment_rows = [
        item
        for item in (transcript.get('sentiment_analysis_results') or [])
        if item.get('speaker') not in (None, 'UNK')
    ]
    df_sentiment = pd.DataFrame.from_dict(sentiment_rows) if sentiment_rows else pd.DataFrame()
    if not df_sentiment.empty and 'sentiment' in df_sentiment.columns:
        df_sentiment['sentiment_value'] = df_sentiment['sentiment'].map(
            {'NEGATIVE': -1, 'NEUTRAL': 0, 'POSITIVE': 1}
        )

    df_speaker_stats = df_utter.groupby(['speaker']).agg(
        text=('text', lambda values: ' '.join(values)),
        word_count=('word_count', 'sum'),
        talk_time_sec=('talk_time_sec', 'sum'),
        words_per_min=('words_per_min', 'mean'),
    )
    df_speaker_stats['word_count_pct'] = (
        100 * df_speaker_stats['word_count'] / df_speaker_stats['word_count'].sum()
    )
    talk_total = df_speaker_stats['talk_time_sec'].sum()
    df_speaker_stats['talk_time_pct'] = (
        100 * df_speaker_stats['talk_time_sec'] / talk_total if talk_total else 0
    )
    df_speaker_stats['question_count'] = df_speaker_stats['text'].apply(lambda text: str(text).count('?'))
    df_speaker_stats['idea_count'] = df_speaker_stats['text'].apply(idea_count)
    df_speaker_stats = df_speaker_stats.apply(_unique_words, axis=1)
    interaction_count = df_utter['speaker'].value_counts().rename('interaction_count')
    df_speaker_stats = pd.merge(
        df_speaker_stats, interaction_count, left_index=True, right_index=True
    )
    action_counts = {
        speaker: len((speaker_map.get(speaker) or {}).get('personal_action_items') or [])
        for speaker in df_speaker_stats.index
    }
    df_speaker_stats = pd.merge(
        df_speaker_stats,
        pd.Series(action_counts, name='action_items_count'),
        left_index=True,
        right_index=True,
    )
    df_speaker_stats['speaker'] = df_speaker_stats.index
    df_speaker_stats.index.name = 'name'
    df_speaker_stats = df_speaker_stats[
        [
            'speaker',
            'text',
            'word_count',
            'word_count_pct',
            'words_per_min',
            'talk_time_sec',
            'talk_time_pct',
            'question_count',
            'idea_count',
            'unique_words_count',
            'unique_words_pct',
            'interaction_count',
            'action_items_count',
        ]
    ]

    names = {key: value['name'] for key, value in speaker_map.items()}
    if not df_sentiment.empty and 'speaker' in df_sentiment.columns:
        df_sentiment['name'] = df_sentiment['speaker'].replace(names)
    df_utter['name'] = df_utter['speaker'].replace(names)
    df_utter['name_target'] = df_utter['target'].replace(names)
    df_utter['name_previous'] = df_utter['previous'].replace(names)
    df_speaker_stats = df_speaker_stats.reset_index(drop=True)
    df_speaker_stats['name'] = df_speaker_stats['speaker'].replace(names)
    df_speaker_stats = df_speaker_stats.set_index('name')

    df_nice = (
        df_speaker_stats[list(NICE_STATS_COL_NAME.keys())]
        .rename(columns=NICE_STATS_COL_NAME)
        .sort_index(axis=0)
    )
    df_nice.loc['Average'] = df_nice.mean(numeric_only=True)
    for old_name, new_name in NICE_STATS_COL_NAME.items():
        df_nice[new_name] = df_nice[new_name].apply(NICE_STATS_COL_APPLY_OP[old_name])

    df_interactions_next = pd.crosstab(df_utter['name'], df_utter['name_target'], margins=True)
    df_interactions_prev = pd.crosstab(df_utter['name'], df_utter['name_previous'], margins=True)
    df_interactions_next_words = (
        pd.crosstab(
            index=df_utter['name'],
            columns=df_utter['name_target'],
            values=df_utter['word_count'],
            aggfunc='sum',
            margins=True,
        )
        .fillna(0)
        .astype('int')
    )
    df_interactions_prev_words = (
        pd.crosstab(
            index=df_utter['name'],
            columns=df_utter['name_previous'],
            values=df_utter['word_count'],
            aggfunc='sum',
            margins=True,
        )
        .fillna(0)
        .astype('int')
    )
    lines = [f'* {row["name"]}: "{row["text"]}"' for _, row in df_utter.iterrows()]
    conversation_script = ('\n'.join(lines) + '\n') if lines else ''
    interactions_prev = df_interactions_prev.iloc[:-1, :-1].copy()
    return SessionTables(
        df_utter=df_utter,
        df_speaker_stats=df_speaker_stats,
        df_nice_stats_data=df_nice,
        conversation_script=conversation_script,
        gini_reverse_interventions=_one_minus_gini(df_interactions_next['All'].iloc[:-1].tolist()),
        gini_reverse_interactions=_one_minus_gini(_off_diagonal(df_interactions_next)),
        gini_reverse_interactions_both_wrd_cnt=_one_minus_gini(
            _off_diagonal(df_interactions_prev_words) + _off_diagonal(df_interactions_next_words)
        ),
        interactions_prev=interactions_prev,
    )


def _sentiment_frame(transcript: dict, names: dict[str, str]) -> pd.DataFrame:
    rows = [
        item
        for item in (transcript.get('sentiment_analysis_results') or [])
        if item.get('speaker') not in (None, 'UNK')
    ]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame.from_dict(rows)
    if 'sentiment' not in frame.columns:
        return pd.DataFrame()
    frame['sentiment_value'] = frame['sentiment'].map({'NEGATIVE': -1, 'NEUTRAL': 0, 'POSITIVE': 1})
    frame['name'] = frame['speaker'].replace(names)
    return frame


def render_charts(
    transcript: dict,
    tables: SessionTables,
    dest_dir: Path,
    roles: dict[str, str] | None = None,
) -> ChartImages:
    dest_dir.mkdir(parents=True, exist_ok=True)
    names = tables.df_nice_stats_data.index[:-1]
    palette = sns.color_palette('hsv', max(len(names), 1))
    adjusted = []
    for color in palette:
        hue, light, saturation = colorsys.rgb_to_hls(*color)
        adjusted.append(colorsys.hls_to_rgb(hue, light * 1.6, saturation * 0.75))
    color_by_name = {name: adjusted[index] for index, name in enumerate(names)}

    bar_images = []
    for column in tables.df_nice_stats_data.columns:
        ordered = (
            tables.df_nice_stats_data[[column]].iloc[:-1].sort_values(by=column, ascending=False)
        )
        figure = barplot(
            data=ordered.transpose(),
            title=column,
            colors=[color_by_name[name] for name in ordered.index],
        )
        bar_images.append(_as_image(figure))

    names_map = {
        speaker: name
        for name, speaker in tables.df_speaker_stats['speaker'].items()
    }
    sentiment = _sentiment_frame(transcript, names_map)
    convo_sentiment = _as_image(_conversation_sentiment(sentiment))
    speaker_sentiment = _as_image(_speaker_sentiment(sentiment, list(tables.df_speaker_stats.index)))
    convo_wpm = _as_image(_conversation_wpm(tables.df_utter))
    speaker_wpm = _as_image(_speaker_wpm(tables.df_utter, list(tables.df_speaker_stats.index)))
    action_items = _as_image(_action_pie(tables.df_speaker_stats['action_items_count']))
    ordered = tables.df_utter.sort_values('start') if 'start' in tables.df_utter.columns else tables.df_utter
    sequence = [str(name) for name in ordered['name'].tolist()]
    roster = [str(name) for name in tables.df_speaker_stats.index]
    chord_path = dest_dir / 'who_talks_with_whom.png'
    save_chord_diagram(sequence, chord_path, names=roster, roles=roles or {})
    _pair_names, directed = directed_follows(sequence, roster)
    chord_pairs = visible_chord_pairs(_pair_names, directed)
    matrix_path = dest_dir / 'transition_matrix.png'
    _save_transition_heatmap(tables.interactions_prev, matrix_path)
    return ChartImages(
        barplots=bar_images,
        convo_sentiment=convo_sentiment,
        speaker_sentiment=speaker_sentiment,
        convo_wpm=convo_wpm,
        speaker_wpm=speaker_wpm,
        action_items=action_items,
        chord_path=chord_path,
        matrix_path=matrix_path,
        chord_pairs=chord_pairs,
    )


def _conversation_sentiment(df_sentiment: pd.DataFrame):
    if df_sentiment.empty or 'sentiment_value' not in df_sentiment.columns:
        return _blank_figure('Sentiment Evolution')
    series = df_sentiment.set_index('start')['sentiment_value'].dropna()
    if series.empty:
        return _blank_figure('Sentiment Evolution')
    window = max(1, int(series.shape[0] * 0.1))
    rolling = series.rolling(window=window, center=True, min_periods=1).mean()
    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(111)
    ax = rolling.plot(ax=ax, alpha=0)
    ax.set_ylim(-1, 1)
    ax.fill_between(rolling.index, rolling, 0, where=(rolling > 0), color='lime', alpha=1)
    ax.fill_between(rolling.index, rolling, 0, where=(rolling < 0), color='deeppink', alpha=1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['-', '0', '+'])
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_title('Sentiment Evolution')
    fig.tight_layout()
    return fig


def _speaker_sentiment(df_sentiment: pd.DataFrame, names: list):
    ncols = 2
    nrows = max(1, math.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, nrows * 2), squeeze=False)
    flat = axes.ravel()
    for extra in range(len(names), len(flat)):
        flat[extra].axis('off')
    if df_sentiment.empty:
        fig.tight_layout()
        return fig
    timeline = df_sentiment.set_index('start')
    for index, name in enumerate(sorted(names)):
        person = timeline.loc[timeline['name'] == name, 'sentiment_value']
        person = person.reindex(timeline.index, fill_value=0)
        window = max(1, int(person.shape[0] * 0.10))
        rolling = person.rolling(window=window, center=True, min_periods=1).mean()
        ax = flat[index]
        rolling.plot(ax=ax, alpha=0)
        ax.fill_between(rolling.index, rolling, 0, where=(rolling > 0), color='lime', alpha=1)
        ax.fill_between(rolling.index, rolling, 0, where=(rolling < 0), color='deeppink', alpha=1)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['-', '0', '+'])
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_title(f'Sentiment Evolution ({name})')
        ax.autoscale(tight=True, axis='x')
    fig.tight_layout()
    return fig


def _conversation_wpm(df_utter: pd.DataFrame):
    series = df_utter.set_index('start')['words_per_min']
    window = max(1, int(series.shape[0] * 0.1))
    rolling = series.rolling(window=window, min_periods=1).mean()
    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(111)
    rolling.plot(ax=ax, label='')
    ax.set_title('Words Per Minute Evolution')
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_ylim(0, 300)
    ax.autoscale(tight=True, axis='x')
    ax.grid(axis='y', alpha=0.3)
    mean = series.mean()
    ax.axhline(y=mean, color='lime', zorder=-1, label=f'Average = {round(mean, 1)}')
    ax.legend()
    fig.tight_layout()
    return fig


def _speaker_wpm(df_utter: pd.DataFrame, names: list):
    ncols = 2
    nrows = max(1, math.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, nrows * 2), squeeze=False)
    flat = axes.ravel()
    for extra in range(len(names), len(flat)):
        flat[extra].axis('off')
    timeline = df_utter.set_index('start')
    for index, name in enumerate(sorted(names)):
        person = timeline.loc[timeline['name'] == name, 'words_per_min'].reindex(timeline.index)
        window = max(1, int(person.shape[0] * 0.10))
        rolling = person.rolling(window=window, center=True, min_periods=1).mean()
        ax = flat[index]
        rolling.plot(ax=ax, label='')
        ax.set_title(f'Words per minute Evolution ({name})')
        ax.set_ylim(0, 400)
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.autoscale(tight=True, axis='x')
        ax.grid(axis='y', alpha=0.3)
        mean = person.mean()
        ax.axhline(y=mean, color='lime', zorder=-1, label=f'Average = {round(mean, 1)}')
        ax.legend()
    fig.tight_layout()
    return fig


def _action_pie(counts: pd.Series):
    ordered = counts.sort_index()
    fig, ax = plt.subplots()
    if ordered.sum() <= 0:
        ax.set_title('Action Items')
        ax.text(0.5, 0.5, 'No action items', ha='center', va='center')
        ax.axis('off')
        return fig
    colors = plt.get_cmap('Set3').colors
    ax.pie(
        ordered.to_list(),
        labels=ordered.index.to_list(),
        autopct=lambda value: round(value * ordered.sum() / 100),
        pctdistance=0.8,
        colors=colors,
        counterclock=False,
        startangle=90,
    )
    return fig


def _save_transition_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if matrix.empty:
        ax.set_title('Who spoke after whom')
        ax.text(0.5, 0.5, 'No transitions', ha='center')
        ax.axis('off')
    else:
        sns.heatmap(
            matrix.fillna(0),
            annot=True,
            fmt='g',
            cmap='Blues',
            ax=ax,
            cbar=False,
            linewidths=0.4,
            linecolor='white',
        )
        ax.set_title('Who spoke after whom')
        ax.set_xlabel('Previous speaker')
        ax.set_ylabel('Speaker')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def framework_frames(speaker_analysis: dict, framework_key: str):
    values = pd.DataFrame()
    explanations = pd.DataFrame()
    for speaker in speaker_analysis:
        name = speaker_analysis[speaker]['name']
        frame = pd.DataFrame.from_dict(speaker_analysis[speaker][framework_key], orient='columns')
        values = pd.concat(
            [values, frame.loc[['value']].rename(index={'value': name}).astype(int)]
        ).sort_index()
        explanations = pd.concat(
            [explanations, frame.loc[['explanation']].rename(index={'explanation': name})]
        ).sort_index()
    return values, explanations


# Most frequent bigram. A pair is skipped only when both words are stopwords,
# unless it is a known hedge ("i think", "you know"). Ties keep the phrase that
# appeared first. If every pair is skipped, the raw top bigram is used.

import string as _string
from collections import Counter as _Counter

BIGRAM_STOPWORDS = frozenset(
    """i me my myself we our ours ourselves you your yours yourself yourselves he him
    his himself she her hers herself it its itself they them their theirs themselves
    what which who whom this that these those am is are was were be been being have has
    had having do does did doing a an the and but if or because as until while of at by
    for with about against between into through during before after above below to from
    up down in out on off over under again further then once here there when where why
    how all any both each few more most other some such no nor not only own same so than
    too very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn
    hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn
    dont im its thats youre theyre were ive id ill cant wont didnt doesnt isnt
    gonna wanna yeah okay oh um uh""".split()
)

DISCOURSE_MARKERS = frozenset(
    {
        'i think',
        'i mean',
        'i guess',
        'i feel',
        'i believe',
        'i know',
        'i suppose',
        'you know',
        'you see',
        'sort of',
        'kind of',
        'a lot',
        'and so',
        'so that',
        'i would',
        'we should',
        'we need',
        'you have',
        'have to',
        'at least',
        'of course',
    }
)


def extract_bigrams(text: str) -> list[str]:
    cleaned = str(text or '').translate(str.maketrans('', '', _string.punctuation)).lower()
    tokens = nltk.word_tokenize(cleaned)
    return [f'{first} {second}' for first, second in zip(tokens, tokens[1:])]


def top_bigram(text: str) -> tuple[str, int] | None:
    """Return (bigram, count) for the most frequent meaningful bigram, or None."""
    counts = _Counter(extract_bigrams(text))
    if not counts:
        return None
    for phrase, count in counts.most_common():
        first, second = phrase.split(' ', 1)
        if phrase in DISCOURSE_MARKERS or not (
            first in BIGRAM_STOPWORDS and second in BIGRAM_STOPWORDS
        ):
            return phrase, count
    return counts.most_common(1)[0]
