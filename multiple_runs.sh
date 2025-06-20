declare -A file_speakers_count=(
    # ['file_name']=speakers_expected

    # Testing recordings
    ['filename1.mp4']=4 # Example1
    ['filename2.mp4']=4 # Example2
)

cat /dev/null > multiple_runs_log.txt

for file_name in ${(k)file_speakers_count}; do
    count=$file_speakers_count[$file_name]
    
    # IMPORTANT: `speaker_map_skip_rename` must be set to True
    # report_generation.ipynb uses `input()` and papermill does not support so no renaming
    # can happen

    command="papermill report_generation.ipynb - -p file_name '$file_name' -p speakers_expected $count \
-p speaker_map_force_regen False \
-p speaker_map_force_regen_personal_summary False \
-p speaker_map_force_regen_personal_key_contributions False \
-p speaker_map_force_regen_personal_action_items False \
-p speaker_map_force_rename False \
-p speaker_map_skip_rename True \
-p conversation_summaries_force_regen False \
-p conversation_summaries_force_regen_conversation_title False \
-p conversation_summaries_force_regen_conversation_summary_short False \
-p conversation_summaries_force_regen_conversation_summary_long False \
-p speaker_analysis_force_regen False \
-p speaker_analysis_force_regen_personal_super_power False \
-p speaker_analysis_force_regen_personal_framework_amase False \
-p speaker_analysis_force_regen_personal_framework_enneagram False \
-p speaker_analysis_force_regen_personal_framework_doshas False \
-p speaker_analysis_force_regen_personal_errors_and_corrections False"

    printf '\n\n********************\n' | tee -a multiple_runs_log.txt
    date | tee -a multiple_runs_log.txt
    printf '\n********************\n\n' | tee -a multiple_runs_log.txt
    echo $command | tee -a multiple_runs_log.txt
    eval $command >> multiple_runs_log.txt
    printf '\n\n**** end of run ****\n' | tee -a multiple_runs_log.txt
    date | tee -a multiple_runs_log.txt
    printf '\n\n**** end of run ****\n' | tee -a multiple_runs_log.txt
done;