# TRAN
This repo contains the codes for our work “Failures Pave the Way: Enhancing Large Language Models through Tuning-free Rule Accumulation” (EMNLP 2023).

## Setup

The required package can be installed by running the following command.

```
pip install -r requirements.txt
```

For all experiments, please download the data from their repos ([Big-Bench](https://github.com/google/BIG-bench) and [TweetEval](https://github.com/cardiffnlp/tweeteval)). 
The downloaded data are placed in the `data` folder. Here is the example of downloading Big-Bench.

```
cd data
git clone https://github.com/google/BIG-bench
```

## Experiments

Please modify the openai key in the `main.py`.
The `scripts` folder contains code to reproduce our experiments.
For example, to run experiments on `BBQ-Lite`, run the following code:

```
bash scripts/run_bbq.sh # run experiments on bbq-lite
```

## Contacts

Thank you for your interest in our work!

Please feel free to ask about any questions about the algorithms, codes, as well as problems encountered in running them so that we can make it clearer and better. You can either create an issue in the github repo or contact us at YangZeyuan2020@gmail.com.

## Citation

```
@misc{yang2023failures,
      title={Failures Pave the Way: Enhancing Large Language Models through Tuning-free Rule Accumulation}, 
      author={Zeyuan Yang and Peng Li and Yang Liu},
      year={2023},
      eprint={2310.15746},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Licence

`TRAN` is licensed under the terms of the MIT license. See LICENSE for more details.