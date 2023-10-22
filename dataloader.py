import json

def load_data_bbq(path, task):
    data = []
    data_path = path + task.replace('bbq-', '') + '.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['examples']