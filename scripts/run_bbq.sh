# task: bbq-Age or -Religion or -Sexual or -Nationality or -Disability or -SES or -Physical
python main.py \
    --task bbq-Age \
    --seed -1 \
    --data_dir ./data/BIG-bench/bigbench/benchmark_tasks/bbq_lite/resources/ \
    --log_path ./logs/bbq-age.txt \
    --rule_path ./rules/bbq-age.json