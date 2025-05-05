#!/usr/bin/env bash

# first run
python difficulty_measure/parent_model_difficulty_measure.py --base_dir difficulty_measure/data --end_index 20000 &

# second run
python difficulty_measure/parent_model_difficulty_measure.py --base_dir difficulty_measure/data --start_index 20000 &

# wait for both to finish
wait
echo "Both jobs are done."
