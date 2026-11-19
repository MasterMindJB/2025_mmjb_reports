"""Turn an analysis bundle into one group PDF and one individual PDF per participant."""

from __future__ import annotations

import threading
from pathlib import Path

from app.interfaces import AnalysisBundle, RenderedReport
from lib.pdf_functions import create_report_group, create_report_personal, pdf_filename_for
from lib.report_stats import framework_frames, render_charts

ASSETS = Path(__file__).resolve().parents[1] / 'report_assets'
_PLOT_LOCK = threading.Lock()


class PdfReportRenderer:
    def __init__(self, assets: Path = ASSETS):
        self.assets = assets

    def render(
        self,
        *,
        transcript: dict,
        bundle: AnalysisBundle,
        reports_dir: Path,
        participant_labels: set[str],
        feedback_url: str,
        roles_by_label: dict[str, str] | None = None,
    ) -> list[RenderedReport]:
        reports_dir.mkdir(parents=True, exist_ok=True)
        participant_map = {
            label: bundle.speaker_map[label]
            for label in bundle.speaker_map
            if label in participant_labels
        }
        participant_analysis = {
            label: bundle.speaker_analysis[label]
            for label in participant_map
            if label in bundle.speaker_analysis
        }
        missing = [label for label in participant_map if label not in participant_analysis]
        if missing:
            raise RuntimeError(f'Missing analysis for {", ".join(missing)}.')

        roles_by_name = {}
        for label, info in bundle.speaker_map.items():
            role = (roles_by_label or {}).get(label) or info.get('role') or 'participant'
            roles_by_name[info['name']] = role

        with _PLOT_LOCK:
            charts = render_charts(transcript, bundle.tables, reports_dir, roles=roles_by_name)
            group_path = reports_dir / 'group.pdf'
            stories = getattr(bundle, 'stories', None) or {}
            create_report_group(
                images_barplot=charts.barplots,
                image_convo_sentiment_evo=charts.convo_sentiment,
                image_speaker_sentiment_evo=charts.speaker_sentiment,
                image_convo_wpm=charts.convo_wpm,
                image_speaker_wpm=charts.speaker_wpm,
                image_action_items=charts.action_items,
                path_image_chord=charts.chord_path,
                path_image_matrix=charts.matrix_path,
                chord_pairs=charts.chord_pairs,
                path_report_assets=self.assets,
                path_file=group_path,
                summaries=bundle.summaries,
                gini_reverse_interventions=bundle.tables.gini_reverse_interventions,
                gini_reverse_interactions=bundle.tables.gini_reverse_interactions,
                gini_reverse_interactions_both_wrd_cnt=bundle.tables.gini_reverse_interactions_both_wrd_cnt,
                df_nice_stats_data=bundle.tables.df_nice_stats_data.transpose(),
                speaker_map=bundle.speaker_map,
                link_feedback=feedback_url or None,
                include_individual_sections=False,
                story_draft=stories.get('group'),
                transcript=transcript,
            )
            amase_values, amase_explanations = framework_frames(
                participant_analysis, 'personal_framework_amase'
            )
            enneagram_values, enneagram_explanations = framework_frames(
                participant_analysis, 'personal_framework_enneagram'
            )
            dosha_values, dosha_explanations = framework_frames(
                participant_analysis, 'personal_framework_doshas'
            )
            create_report_personal(
                transcript=transcript,
                story_drafts=stories.get('personal'),
                path_report_assets=self.assets,
                path_reports_dir=reports_dir,
                df_nice_stats_data=bundle.tables.df_nice_stats_data,
                speaker_map=participant_map,
                link_feedback=feedback_url or None,
                speaker_analysis=participant_analysis,
                df_framework_amase_values=amase_values,
                df_framework_amase_explanation=amase_explanations,
                df_framework_enneagram_values=enneagram_values,
                df_framework_enneagram_explanation=enneagram_explanations,
                df_framework_doshas_values=dosha_values,
                df_framework_doshas_explanation=dosha_explanations,
            )

        rendered = [
            RenderedReport('group', group_path, group_path.name, None),
        ]
        for label, info in participant_map.items():
            filename = pdf_filename_for(info['name'])
            path = reports_dir / filename
            if not path.is_file():
                raise RuntimeError(f'Individual report was not written for {info["name"]}.')
            rendered.append(RenderedReport('individual', path, filename, label))
        return rendered
