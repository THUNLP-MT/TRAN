import json
import openai
import time
from tqdm import tqdm
import os
import sys
import logging
import argparse

import requests
from requests.auth import HTTPBasicAuth
from rank_bm25 import BM25Okapi

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--task", type=str, default="bbq-Age",)
    parser.add_argument("--data_dir", type=str, default="./data/BIG-bench/bigbench/benchmark_tasks/bbq_lite/resources/",)
    parser.add_argument("--log_path", type=str, default="./logs/log.txt")
    parser.add_argument("--rule_path", type=str, default="./rules/rule-book.json")
    parser.add_argument("--num_rule_limit", type=int, default=100)
    parser.add_argument("--num_rule_per_sample", type=int, default=3)
    args = parser.parse_args()

    return args

compare_contradictory_prompt = "\
I will give you two rules. \
Please help me classify whether the contents of these two rules are contradictory. \
You are only allowed to give me the answer, selecting from \"contradictory\" and \"not contradictory\".\n\n\
"

compare_identical_prompt = "\
I will give you two rules. \
Please help me classify whether the contents of these two rules are exactly identical. \
You are only allowed to give me the answer, selecting from \"identical\" and \"not identical\".\n\n\
"

def post_message(messages, tokens, logger):
    # 调用接口
    stime = 1

    response = openai.Completion.create(
        model="gpt-3.5-turbo-0301",
        messages=messages,
        temperature=0,
        stream=false,
    )  

    tokens += response['usage']['total_tokens']
    messages.append({'role': response['choices'][0]['message']['role'], 'content': response['choices'][0]['message']['content']})
    time.sleep(stime)

    return messages, response, tokens

def reasoning_rules_bbq(messages, tokens, logger, ans_index):

    messages.append({'role': 'user', 'content': f'You are wrong. This correct answer is Answer {ans_index}.'})
    messages, _, tokens = post_message(messages, tokens, logger)
    
    messages.append({'role': 'user', 'content': f'Please give me the reasons for Answer {ans_index} as the correct answer. List by points.'})
    messages, _, tokens = post_message(messages, tokens, logger)

    messages.append({'role': 'user', 'content': 'Be more general and concise.'})
    messages, _, tokens = post_message(messages, tokens, logger)
    
    messages.append({'role': 'user', 'content': 'Please rewrite these reasons into rules for making judgments, using the format of "if..., then...". Give it in sections. Each is an independent rule. Directly give the content of the rule. Do not answer anything else:'})
    messages, response, tokens = post_message(messages, tokens, logger)

    return messages, response, tokens

def get_valid_rules(raw_rules):

    valid_rules = []
    for rule in raw_rules:
        lower_rule = rule.lower()
        if len(rule) >= 10 and 'If' in rule and 'then' in lower_rule:
            secs = rule.split('If')
            ls = len(secs[0])
            valid_rules.append(rule[ls:])
        
    return valid_rules

def compare_rules(rule1, rule2, compare_prompt):

    prompt = compare_prompt + '1. ' + rule1 + '\n2. ' + rule2 + '\nAnswer: '
    return prompt

class RuleBook():

    def __init__(self, logger):
        self.rules = []
        self.valid_rules = []
        self.rule_id = {}
        self.rule_input_idx = {}
        self.rule_use = {}
        self.rule_keep = {}
        self.samples = []
        self.sample_id = {}
        self.valid_samples = []
        self.sample_line_data = {}
        self.sample_rule = {}
        self.rule_sample = {}

        self.fail_sample_line_data = {}
        self.fail_samples = []

        self.logger = logger
    
    def register_sentence_bert(self):
        pass

    def update_rules(self, sample, line_data, rules, index):
                
        if sample not in self.samples:
            self.samples.append(sample)
            self.sample_line_data[sample] = line_data
            self.sample_rule[sample] = []
            sid = len(self.samples) - 1
            self.sample_id[sample] = sid
            self.logger.info('Sample Already in the Rule Book')
        else:
            sid = self.sample_id[sample]

        already_exist_rules = []

        for rule in rules:
            
            self.rules.append(rule)
            rid = len(self.rules) - 1
            
            if rule in self.valid_rules:
                self.logger.info('Rule Already in the Rule Book')
                self.valid_rules.remove(rule)
                self._replace_rule_idx(self.rule_id[rule], rid)
                already_exist_rules.append(rule)
            else:
                self.rule_use[rule] = 0

            self.valid_rules.append(rule)
            self.rule_id[rule] = rid
            self.rule_input_idx[rule] = index
            self.rule_keep[rule] = True
            self.sample_rule[sample].append(rid)
            self.rule_sample[rule] = sid

        self.valid_samples = [sample for sample in self.samples if len(self.sample_rule[sample]) > 0]
        tokenized_samples = [sample.split() for sample in self.valid_samples]
        self.bm25 = BM25Okapi(tokenized_samples)
        
        self.quick_log()
        return already_exist_rules
    
    def update_samples_rules(self, samples, rules, sample_line_data, sample_rules, index):
            
        for rule in rules:
            self.rules.append(rule)
            rid = len(self.rules) - 1
            self.valid_rules.append(rule)
            self.rule_use[rule] = 0
            self.rule_id[rule] = rid
            self.rule_input_idx[rule] = index
            self.rule_keep[rule] = True
        
        for sample in samples:
            self.samples.append(sample)
            self.sample_line_data[sample] = sample_line_data[sample]
            self.sample_rule[sample] = []
            sid = len(self.samples) - 1
            self.sample_id[sample] = sid
            for rule in sample_rules[sample]:
                rid = self.rule_id[rule]
                self.sample_rule[sample].append(rid)
                self.rule_sample[rule] = sid
        
        self.valid_samples = [sample for sample in self.samples if len(self.sample_rule[sample]) > 0]
        if len(self.valid_samples) > 0:
            tokenized_samples = [sample.split() for sample in self.valid_samples]
            self.bm25 = BM25Okapi(tokenized_samples)
        else: self.bm25 = None

        self.quick_log()

    def retrieval_rules_bm25(self, query, line_data, n_sample=10, n_rule=3):
        
        tokenized_query = query.split()
        top_samples = self.bm25.get_top_n(tokenized_query, self.valid_samples, n=n_sample)
        
        select_rules = []
        self.logger.info('Selected Examples: ')
        for sample in top_samples: 
            self.logger.info(sample.replace('\n', '<n>'))
            for rule_idx in self.sample_rule[sample]:
                rule = self.rules[rule_idx]
                if self.rule_keep[rule] and rule not in select_rules: select_rules.append(rule)
        
        tokenized_rules = [rule.split() for rule in select_rules]
        bm25_rules = BM25Okapi(tokenized_rules)
        top_rules = bm25_rules.get_top_n(tokenized_query, select_rules, n=n_rule)

        self._update_rule_use(top_rules)
        
        return top_rules 

    def check_contradictory_identical(self, new_rules):
        
        valid_rules = [vr for vr in self.valid_rules if vr not in new_rules]
        tokenized_rules = [vr.split() for vr in valid_rules]
        bm25_rules = BM25Okapi(tokenized_rules)

        similar_rules = []
        for new_rule in new_rules:
            sim_rules = bm25_rules.get_top_n(new_rule.split(), valid_rules, n=1)
            similar_rules.append(sim_rules[0])

        tokens = 0
        self.logger.info('Checking Rules ... ')
        for new_rule, sim_rule in zip(new_rules, similar_rules):
            
            if sim_rule in self.valid_rules:
            
                self.logger.info('Checking Incoming Rule: ' + new_rule)
                self.logger.info('Most Similar Rule: ' + sim_rule)

                keep = True
                
                if keep:
                    self.logger.info('Check Conflict')
                    rel = 'contradictory'
                    messages = [{'role': 'user', 'content': compare_rules(new_rule, sim_rule, compare_contradictory_prompt)}]
                    messages, response, tokens = post_message(messages, tokens, self.logger) 

                    answer = response['choices'][0]['message']['content'].replace('assistant', '').replace(':', '').strip().lower()
                    if 'contradictory' in answer and 'not' not in answer: keep = False
                
                if keep:
                    self.logger.info('Check Identical')
                    rel = 'identical'
                    messages = [{'role': 'user', 'content': compare_rules(new_rule, sim_rule, compare_identical_prompt)}]
                    messages, response, tokens = post_message(messages, tokens, self.logger)

                    answer = response['choices'][0]['message']['content'].replace('assistant', '').replace(':', '').strip().lower()
                    if 'identical' in answer and 'not' not in answer: keep = False
                
                if not keep:
                    self.logger.info('*****Rule Out*****')
                    self.logger.info('Incoming Rule: ' + new_rule)
                    self.logger.info('Out Rule: ' + sim_rule)
                    self.logger.info('Out by ' + rel)
                    
                    out_idx = self.rule_id[sim_rule]
                    in_idx = self.rule_id[new_rule]

                    replace = True
                    if 'identical' in rel:
                        sid = self.rule_sample[sim_rule]
                        line_sample = self.sample_line_data[self.samples[sid]]
                        success_rules, tokens = self.check_rules_example([new_rule], line_sample, tokens, self.logger, self.convert_prompt, self.task_descrip_prompt, self.check_true_or_false)
                        if len(success_rules) == 0: replace = False
                    
                    if replace:
                        self.rule_keep[sim_rule] = False
                        self.valid_rules.remove(sim_rule)
                        self._replace_rule_idx(out_idx, in_idx)
                    else: 
                        self.logger.info('Fail to correct the old sample...')
                        self.logger.info('Do not replace the rule')

                    self.logger.info('******************')
            else:
                self.logger.info('Checking Incoming Rule: ' + new_rule)
                self.logger.info('Aleardy Out: Most Similar Rule: ' + sim_rule)

        self.logger.info('Check Tokens: ' + str(tokens))

        return tokens
    
    def get_sim_samples(self, sample, line_data, n_sample=2):
        
        tokenized_samples = [fs.split() for fs in self.fail_samples]
        bm25_samples = BM25Okapi(tokenized_samples)
        similar_samples = bm25_samples.get_top_n(sample.split(), self.fail_samples, n=2)
        for ss in similar_samples: self.logger.info('Summary Similar Sample: ' + ss.replace('\n', '<n>'))

        return similar_samples
        
    def get_summary_rules(self, similar_samples, line_data):
        
        line_datas = [line_data] + [self.fail_sample_line_data[sample] for sample in similar_samples]
        prompt = self.construct_summary_prompt(line_datas, self.summary_prompt)

        messages = [{'role': 'user', 'content': prompt}]
        messages, response, tokens = post_message(messages, 0, self.logger)

        raw_rules = response['choices'][0]['message']['content'].split('\n')
        valid_rules = get_valid_rules(raw_rules)
        for vr in valid_rules: self.logger.info('Summary Valid Rule: ' + vr)

        return valid_rules, tokens

    def summary_and_update(self, sample, line_data, n_sample=2):
        
        self.logger.info('====================================')
        self.logger.info('========== Summary Rules ===========')
        self.logger.info('====================================')

        tokens = 0
        similar_samples = self.get_sim_samples(sample, line_data, n_sample=n_sample)
        valid_rules, new_tokens = self.get_summary_rules(similar_samples, line_data)
        tokens += new_tokens

        self.logger.info('====== Checking Summary Rules ======')

        new_valid_rules = [vr for vr in valid_rules if vr not in self.valid_rules]
        if len(new_valid_rules) == 0: return tokens
        elif len(new_valid_rules) >= 5: new_valid_rules = new_valid_rules[:4]

        summary_samples = similar_samples + [sample]
        summary_sample_line_datas, success_sample_rules = {}, {}
        summary_sample_line_datas[sample] = line_data
        for ss in similar_samples: summary_sample_line_datas[ss] = self.fail_sample_line_data[ss]

        success_rules, success_samples = [], []
        for summary_sample in summary_samples:
            line_sample = summary_sample_line_datas[summary_sample]
            self.logger.info('Checking Summary Rules for Fail Sample: ' + summary_sample.replace('\n', '<n>'))
            success_new_rules, tokens = self.check_rules_example(new_valid_rules, line_sample, tokens, self.logger, self.convert_prompt, self.task_descrip_prompt, self.check_true_or_false)
            if len(success_new_rules) > 0:
                self.logger.info('Get New Effective Summary Rules!')
                success_sample_rules[summary_sample] = success_new_rules
                success_samples.append(summary_sample)
                success_rules += success_new_rules
            else: self.logger.info('No New Effective Summary Rules!')
        
        success_rules = list(set(success_rules))

        self.update_samples_rules(success_samples, success_rules, summary_sample_line_datas, success_sample_rules, line_data['index'])
        
        for ss in similar_samples:
            if ss in success_samples: self.fail_samples.remove(ss)
        self.fail_sample_line_data[sample] = line_data
        if sample not in success_samples: self.fail_samples.append(sample)

        return tokens

    def compress_lru(self, threshold=100):

        if len(self.valid_rules) > threshold:

            self.logger.info('Out Rules by LRU...')
            
            tp = len(self.valid_rules) - threshold
            for out_rule in self.valid_rules[:tp]:
                self.logger.info('LRU Out Rule: ' + out_rule)
                self.rule_keep[out_rule] = False
                
                out_idx = self.rule_id[out_rule]
                for sample in self.samples:
                    rule_idxs = self.sample_rule[sample]
                    if out_idx in rule_idxs:
                        rule_idxs.remove(out_idx)
                        self.sample_rule[sample] = rule_idxs
            
            self.valid_rules = self.valid_rules[tp:]

    def save(self, path):
        self.logger.info('Saving Rule Book ...')
        save_data = {'rules': self.rules,
                     'valid_rules': self.valid_rules,
                     'rule_id': self.rule_id,
                     'rule_input_idx': self.rule_input_idx,
                     'rule_use': self.rule_use,
                     'rule_keep': self.rule_keep,
                     'samples': self.samples,
                     'sample_id': self.sample_id,
                     'valid_samples': self.valid_samples,
                     'sample_line_data': self.sample_line_data,
                     'sample_rule': self.sample_rule,
                     'rule_sample': self.rule_sample,
                     'fail_samples': self.fail_samples,
                     'fail_sample_line_data': self.fail_sample_line_data}
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f)

    def load(self, path):
        self.logger.info('Loading Rule Book ...')
        with open(path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)

        self.rules = save_data['rules']
        self.valid_rules = save_data['valid_rules']
        self.rule_id = save_data['rule_id']
        self.rule_input_idx = save_data['rule_input_idx']
        self.rule_use = save_data['rule_use']
        self.rule_keep = save_data['rule_keep']
        self.samples = save_data['samples']
        self.sample_id = save_data['sample_id']
        self.valid_samples = save_data['valid_samples']
        self.sample_line_data = save_data['sample_line_data']
        self.sample_rule = save_data['sample_rule']
        self.rule_sample = save_data['rule_sample']

        self.logger.info('Total ' + str(len(save_data['samples'])) + ' Examples')
        self.logger.info('Available ' + str(len(save_data['valid_samples'])) + ' Examples')
        self.logger.info('Total ' + str(len(save_data['rules'])) + ' Rules')
        self.logger.info('Available ' + str(len(save_data['valid_rules'])) + ' Rules')

        if 'fail_samples' in save_data:
            self.fail_samples = save_data['fail_samples']
            self.fail_sample_line_data = save_data['fail_sample_line_data']

        tokenized_samples = [sample.split() for sample in self.valid_samples]
        self.bm25 = BM25Okapi(tokenized_samples)

    def quick_log(self):
        self.logger.info('Total ' + str(len(self.samples)) + ' Examples')
        self.logger.info('Available ' + str(len(self.valid_samples)) + ' Examples')
        self.logger.info('Total ' + str(len(self.rules)) + ' Rules')
        self.logger.info('Available ' + str(len(self.valid_rules)) + ' Rules')
    
    def log_rules(self):
        self.logger.info('Rules: ')
        zrls = 0
        for sample in self.samples:
            self.logger.info(sample.replace('\n', '<n>'))
            for rule_idx in self.sample_rule[sample]:
                rule = self.rules[rule_idx]
                use_count = self.rule_use[rule]
                kep = self.rule_keep[rule]
                self.logger.info(f'\tKeep: ' + str(kep) + f'\tUse Count: {use_count}\t' + rule)
                if use_count == 0: zrls += 1
        self.logger.info('Never Use Rules: ' + str(zrls))
        self.quick_log

    def load_check_rule_example(self, check_rules_example, convert_prompt, task_descrip_prompt, check_true_or_false):
        self.check_rules_example = check_rules_example
        self.convert_prompt = convert_prompt
        self.task_descrip_prompt = task_descrip_prompt
        self.check_true_or_false = check_true_or_false

    def load_construct_summary_prompt(self, construct_summary_prompt, summary_prompt):
        self.construct_summary_prompt = construct_summary_prompt
        self.summary_prompt = summary_prompt

    def _update_rule_use(self, rules):

        for rule in rules:
            self.rule_use[rule] += 1
            self.valid_rules.remove(rule)
            self.valid_rules.append(rule)

    def _replace_rule_idx(self, old_idx, new_idx):
        
        for sample in self.samples:
            rule_idxs = self.sample_rule[sample]
            if old_idx in rule_idxs:
                self.logger.info('Update Sample: ' + sample)
                rule_idxs.remove(old_idx)
                rule_idxs.append(new_idx)
                self.sample_rule[sample] = rule_idxs

def set_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return