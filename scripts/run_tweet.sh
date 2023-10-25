# task: tweet-offensive or tweet-irony
python main.py \
    --task tweet-offensive \
    --seed -1 \
    --data_dir ./data/tweeteval/datasets/ \
    --log_path ./logs/tweet-offensive.txt \
    --rule_path ./rules/tweet-offensive.json