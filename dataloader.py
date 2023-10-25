import json

def load_data_bbq(path, task):
    data = []
    data_path = path + task.replace('bbq-', '') + '.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['examples']

def load_data_tweet(path, task, split='test'):

    task = task.replace('tweet-', '')
    
    data = []
    sentences, idxs = [], []
    with open(path + f'tweeteval/datasets/{task}/{split}_text.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sent = line.strip()
            sentences.append(sent)
    with open(path + f'tweeteval/datasets/{task}/{split}_labels.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            idx = eval(line.strip())
            idxs.append(idx)
    for idx, sent in zip(idxs, sentences):
        data.append({'sentence': sent, 'label': idx})
    
    return data

def load_data_bbh(path, task):

    task = task.replace('bbh-', '')
    file_name = {
        'dyck': 'dyck_languages.json'
    }
    
    data = []
    with open(path + file_name[task], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['examples']