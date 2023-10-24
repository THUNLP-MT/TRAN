import json
import openai
import time
from tqdm import tqdm
import os
import sys
import logging

import requests
from requests.auth import HTTPBasicAuth
from rank_bm25 import BM25Okapi

from utils import post_message

def formulate_rule_prompt(rules):

    rule_prompt = 'Given following rules: \n'
    for i, rule in enumerate(rules):
        rule_prompt += f'\"' + rule.strip() + '\"'
        rule_prompt += '\n'

    return rule_prompt + '\n'

def check_rules_example(rules, line_data, tokens, logger, convert_prompt, task_descrip_prompt, check_true_or_false, task):
    # Check rules
    success_rules = []
    for rule in rules:

        messages = [{'role': 'user', 'content': formulate_rule_prompt([rule]) + convert_prompt(line_data, task_prompt=task_descrip_prompt)}]
        messages, response, tokens = post_message(messages, tokens, logger)        
        
        answer = messages[1]['content'].replace('assistant', '').replace(':', '').strip()
        correct = check_true_or_false(answer, line_data, task)
        if correct:
            logger.info('The Answer is Correct!')
            success_rules.append(rule)
        else:
            logger.info('The Answer is Still Wrong!')
    
    return success_rules, tokens

task_descrip_prompt_bbq = "\
Help me perform a multiple-choice question answering task. \
Given the context, I will give you a question and three possible answers to choose from. \
You need to find the best answer. \
You are only allowed to respond the answer index, selecting from 1, 2, and 3. \
"

summary_prompt_bbq = "\
I am doing a multiple-choice question answering task. \
Given the context and question, I need to choose the best answer from three possible answers. \
Here I will give you several examples. \
Please help me summarize the rules to choose the answer, using the format of \"if..., then...\". \
Be general and concise. Give it in sections. Each is an independent rule. Directly give the content of the rule. \
Do not answer anything else. \
"

def line_data_to_key_bbq(line_data):
    return 'Context: \"' + line_data['context'] + '\"\nQuestion: \"' + line_data['question'] + '\"' 

def check_true_or_false_bbq(answer, line_data, task=None):
    
    pred = answer.lower()
    label = str(int(line_data['label']) + 1)
    right_answer = line_data['ans' + str(int(line_data['label']))].lower()

    if label not in pred and right_answer not in pred: return False
    else: return True

def convert_prompt_bbq(line_data, task_prompt):
    question_prompt = 'Context: \"' + line_data['context'] + '\"\nQuestion: \"' + \
                        line_data['question'] + '\"\nAnswer 1: \"' + line_data['ans0'] + \
                        '\"\nAnswer 2: \"' + line_data['ans1'] + '\"\nAnswer 3: \"' + \
                        line_data['ans2'] + '\"\nCorrect Answer: '
    prompt = task_prompt + '\n\n' + question_prompt

    return prompt

def construct_summary_prompt_bbq(line_datas, summary_prompt, task):
    
    prompt = summary_prompt + '\n\nExamples:\n\n'
    for line_data in line_datas:
        prompt += 'Context: \"' + line_data['context'] + '\"\nQuestion: \"' + \
                        line_data['question'] + '\"\nAnswer 1: \"' + line_data['ans0'] + \
                        '\"\nAnswer 2: \"' + line_data['ans1'] + '\"\nAnswer 3: \"' + \
                        line_data['ans2'] + '\"\nCorrect Answer: Answer '
        prompt += str(int(line_data['label']) + 1)
        prompt += '\n\n'
    
    return prompt + 'Rules: '

task_descrip_prompt_tweet_offensive = "\
Help me perform a classification task. \
I will give you a review and you should help me by figuring whether this review is semantically offensive. \
You are only allowed to give me the answer, selecting from \"offensive\" and \"not offensive\". \
"

task_descrip_prompt_tweet_irony = "\
Help me perform a classification task. \
I will give you a review and you should help me by figuring whether this review is semantically irony. \
You are only allowed to give me the answer, selecting from \"irony\" and \"not irony\". \
"

task_descrip_prompt_tweet = {
    'tweet-offensive': task_descrip_prompt_tweet_offensive,
    'tweet-irony': task_descrip_prompt_tweet_irony
}

def line_data_to_key_tweet(line_data):
    return line_data['sentence']

summary_prompt_tweet_offensive = "\
I am doing a classification task. \
Given a review, I need to figure out whether this review is semantically offensive. \
Here I will give you several examples. \
Please help me summarize the rules to classify these reviews, using the format of \"if..., then...\". \
Be precise and concise. Give it in sections. Each is an independent rule. Directly give the content of the rule. \
Do not answer anything else. \
"

summary_prompt_tweet_irony = "\
I am doing a classification task. \
Given a review, I need to figure out whether this review is semantically irony. \
Here I will give you several examples. \
Please help me summarize the rules to classify these reviews, using the format of \"if..., then...\". \
Be precise and concise. Give it in sections. Each is an independent rule. Directly give the content of the rule. \
Do not answer anything else. \
"

summary_prompt_tweet = {
    'tweet-offensive': summary_prompt_tweet_offensive,
    'tweet-irony': summary_prompt_tweet_irony
}

def check_true_or_false_tweet(answer, line_data, task=None):
    
    task = task.replace('tweet-', '')
    pred = answer.lower()
    if task in pred:
        if 'not' in pred: label = 0
        else: label = 1
    else: label = -1

    if int(label) == int(line_data['label']): return True
    else: return False

def convert_prompt_tweet(line_data, task_prompt):

    prompt = task_prompt + '\n\nReview: \"' + line_data['sentence'] + '\"\nSentiment: '

    return prompt

def construct_summary_prompt_tweet(line_datas, summary_prompt, task):

    task = task.replace('tweet-', '')
    cats = [f'Not {task}', task]
    prompt = summary_prompt + '\n\nExamples:\n\n'
    for line_data in line_datas:
        prompt += 'Review: \"' + line_data['sentence'] + '\"\nSentiment: '
        prompt += cats[line_data['label']]
        prompt += '\n\n'

    return prompt + 'Rules: '

summary_prompt_bbh_word_sorting = "\
I am doing a word sorting task. \
Given a list of words, I need to sort them lexicographically. \
Here I will give you several examples. \
Please help me summarize the rules to sort the words, using the format of \"if..., then...\". \
Be general and concise. Give it in sections. Each is an independent rule. Directly give the content of the rule. \
Do not answer anything else. \
"

summary_prompt_bbh_dyck_languages = "\
I am doing a sequence completion task. \
I need to predict the sequence of the closing parentheses of a Dyck-4 word without its last few closing parentheses. \
Here I will give you several examples. \
Please help me summarize the rules to complete the sequence, using the format of \"if..., then...\". \
Be general and concise. Give it in sections. Each is an independent rule. Directly give the content of the rule. \
Do not answer anything else. \
"

summary_prompt_bbh_object_counting = "\
I am doing an object counting task. \
Given a list of objects, I need to count the number. \
Here I will give you several examples. \
Please help me summarize the rules to count the objects, using the format of \"if..., then...\". \
Be general and concise. Give it in sections. Each is an independent rule. Directly give the content of the rule. \
Do not answer anything else. \
"

task_descrip_prompt_dbpedia = "\
Help me perform a text classification task. \
I will give you a pair of title and content. \
Classify the text into one of the following 14 categories of \
\"Company\", \"Educational Institution\", \"Artist\", \
\"Athlete\", \"Office Holder\", \"Mean Of Transportation\", \
\"Building\", \"Natural Place\", \"Village\", \"Animal\", \
\"Plant\", \"Album\", \"Film\", \"Written Work\". \
You are only allowed to answer one category from these 14 categories. \
"

summary_prompt_dbpedia = "\
I am doing a text classification task. \
Given a pair of title and content, I need to classify it into \
one of the following 14 categories: \
\"Company\", \"Educational Institution\", \"Artist\", \
\"Athlete\", \"Office Holder\", \"Mean Of Transportation\", \
\"Building\", \"Natural Place\", \"Village\", \"Animal\", \
\"Plant\", \"Album\", \"Film\", \"Written Work\". \
Here I will give you several examples. \
Please help me summarize the rules to classify the given text, using the format of \"if..., then...\". \
Be general and concise. Give it in sections. Each is an independent rule. Directly give the content of the rule. \
Do not answer anything else. \
"

task_descrip_prompt_agnews = "\
Please help me perform a news classification task. \
I will give you a news title and the corresponding description. \
You should classify the news into the categories of \"World\", \"Sports\", \"Business\", and \"Technology\". \
You are only allowed to give me a word, selecting from these four categories.\
"

summary_prompt_agnews = "\
I am doing a text classification task. \
Given a news title and the corresponding description, I need to classify it into \
one of the categories of \"World\", \"Sports\", \"Business\", and \"Technology\". \
Here I will give you several examples. \
Please help me summarize the rules to classify the given news, using the format of \"if..., then...\". \
Be general and concise. Give it in sections. Each is an independent rule. Directly give the content of the rule. \
Do not answer anything else. \
"

def line_data_to_key_bbh_word_sorting(line_data):
    return line_data['target']

def line_data_to_key_bbh_dyck_languages(line_data):
    return 'Question: ' + line_data['input']

def line_data_to_key_bbh_object_counting(line_data):
    return 'Question: ' + line_data['input']

def line_data_to_key_dbpedia(line_data):
    return 'Title: \"' + line_data['title'] + '\"\nContent: \"' + line_data['content'] + '\"'

def line_data_to_key_agnews(line_data):
    return 'News: \"' + line_data['title'] + '\"\nDescription: \"' + line_data['description'] + '\"'

def check_true_or_false_bbh_word_sorting(answer, line_data):
    
    gt_answer = line_data['target']
    for i in range(1,10):
        answer = answer.replace(str(i), '')
        gt_answer = gt_answer.replace(str(i), '')
    answer = answer.replace('.', '').split(':')[-1].strip()
    answer = answer.replace(',', ' ').strip()

    gt_answer_words = gt_answer.split()
    answer_words = answer.split()
    
    if len(gt_answer_words) != len(answer_words): return False

    for i in range(len(gt_answer_words)):
        gt_word = gt_answer_words[i].lower()
        word = answer_words[i].lower()
        if gt_word not in word: return False

    return True

def check_true_or_false_bbh_dyck_languages(answer, line_data):
    
    answer = answer.strip().split('\n')[-1]

    gt_answer = line_data['target']
    squeeze_gt_answer = gt_answer.replace(' ', '')
    squeeze_answer = answer.replace(' ', '').replace(' ', '').replace(' ', '')
    if squeeze_gt_answer not in squeeze_answer: return False

    pre_sent = line_data['input'].split('Input: ')[-1].strip()
    squeeze_pre_sent = pre_sent.replace(' ', '')
    squeeze_answer = squeeze_answer.replace(squeeze_pre_sent, '').strip()

    init_len = len(squeeze_answer)
    squeeze_answer = squeeze_answer.replace(squeeze_gt_answer, '')
    clean_len = len(squeeze_answer)

    if init_len - clean_len != len(squeeze_gt_answer): return False
    for token in ['[', ']', '<', '>', '{', '}']:
        if token in squeeze_answer: return False
    
    return True

def check_true_or_false_bbh_object_counting(answer, line_data):
    
    gt_answer = str(line_data['target'])
    answer = answer.lower()
    
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']
    str_number = numbers[int(line_data['target'])]

    if gt_answer not in answer and str_number not in answer: return False

    return True

def check_true_or_false_dbpedia(answer, line_data):
    
    pred = answer.lower()
    label = line_data['label'].lower()

    if label not in pred: return False
    else: return True

def check_true_or_false_agnews(answer, line_data):
    
    pred = answer.lower()
    label = line_data['label'].lower()

    if label not in pred: return False
    else: return True