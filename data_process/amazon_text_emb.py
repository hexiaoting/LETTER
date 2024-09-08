import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
from utils import set_device, load_json, load_plm, clean_text
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoTokenizer, AutoModel
import ipdb


def load_data(args):

    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)

    return item2feature

#features=['title','price','brand','description','categories']
def load_item_data(args,features=['title','brand','description','categories']):
    results = {}
    item_path = os.path.join(args.root, 'meta_all.jsonl')
    with open(item_path, 'r') as f:
        readin = f.readlines()
        for line in readin:
            result = {}
            item = json.loads(line)
            for feature in features:
                if feature not in item.keys():
                    continue
                result[feature] = item[feature]
            results[item['item_id']] = result
    # sorted_items = sorted(results.items(), key=lambda item: item[0])
    return results

def generate_item2feature(args):
    meta_path = os.path.join(args.root, f'{args.dataset}/meta_all.json')

def text_process(example):
    all_text = []
    if "title" in example:
        all_text.append(f'title: {example["title"]}')
    if "description" in example:
        all_text.append(f'description: {example["description"]}')
    if "brand" in example:
        all_text.append(f'brand: {example["brand"]}')
    if "price" in example:
        all_text.append(f'price: {example["price"]}')
    if "categories" in example and len(example["categories"][0]) > 0:
        all_text.append(f'category: {", ".join(example["categories"][0])}')
    return '; '.join(all_text)
def generate_text(item2feature, features):
    item_text_list = []

    for item in item2feature:
        data = item2feature[item]
        text = text_process(data)
        item_text_list.append(text)
        # item_text_list.append([int(item), text])

    return item_text_list

def preprocess_text(args):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)

    #item2feature = load_data(args)
    item2feature = load_item_data(args)
    # load item text and clean
    item_text_list = generate_text(item2feature, ['title', 'brand','description','categories'])
    # item_text_list = generate_text(item2feature, ['title'])
    # return: list of (item_ID, cleaned_item_text)
    print('Saving data...')
    with open(args.output_dir+"meta_text.txt", 'w+') as fout:
        for d in tqdm(item_text_list):
            fout.write(json.dumps({'text': d}) + '\n')

    return item_text_list

def generate_item_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding: ')
    print(' Dataset: ', args.dataset)

    # items, texts = zip(*item_text_list)
    # order_texts = [[0]] * len(items)
    # for item, text in zip(items, texts):
    #     order_texts[item-1] = text
    # for text in order_texts:
    #     assert text != [0]

    embeddings = []
    start, batch_size = 0, 16
    while start < len(item_text_list):
        if (start+1)%100==0:
            print("==>",start+1)
        sentences = item_text_list[start: start + batch_size]
        # print(field_texts)
        # field_texts = zip(*field_texts)

        field_embeddings = []
        # for sentences in field_texts:
            # sentences = list(sentences)
            # print(sentences)
            # if word_drop_ratio > 0:
            #     print(f'Word drop with p={word_drop_ratio}')
            #     new_sentences = []
            #     for sent in sentences:
            #         new_sent = []
            #         sent = sent.split(' ')
            #         for wd in sent:
            #             rd = random.random()
            #             if rd > word_drop_ratio:
            #                 new_sent.append(wd)
            #         new_sent = ' '.join(new_sent)
            #         new_sentences.append(new_sent)
            #     sentences = new_sentences
        encoded_sentences = tokenizer(sentences, max_length=args.max_sent_len,
                                          truncation=True, return_tensors='pt',padding="longest").to(args.device)
        with torch.no_grad():
            outputs = model(input_ids=encoded_sentences.input_ids,
                            attention_mask=encoded_sentences.attention_mask)

        if args.plm_name in ['bert-base-uncased']:
            field_embeddings.append(outputs.last_hidden_state[:, 0])

        elif args.plm_name in ['t5-base']: #TODO
            decoder_input_ids = torch.zeros((batch_data["input_ids"].shape[0], 1), dtype=torch.long).to(batch_data["input_ids"].device)
            hiddens = model(**batch_data, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
            reps = hiddens.decoder_hidden_states[-1][:, 0, :]
        else:
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            field_embeddings.append(mean_output)

        field_mean_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0)
        embeddings.append(field_mean_embedding)
        start += batch_size

    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    print('Embeddings shape: ', embeddings.shape)

    file = os.path.join(args.output_dir, args.dataset + '.emb-' + args.plm_name + "-td" + ".npy")
    np.save(file, embeddings)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty', help='Instruments / Arts / Games')
    parser.add_argument('--root', type=str, default="../../../datasets/amazon-review/review_core_2014/preprocess/")
    parser.add_argument('--output_dir', type=str, default="/home/hewenting/dataprocess/")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='bert-base-uncased')
    parser.add_argument('--plm_checkpoint', type=str,
                        default='/mnt/disk5/hewenting_nfs_serverdir/models/google-bert:bert-base-uncased')
    parser.add_argument('--max_sent_len', type=int, default=512)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.root = os.path.join(args.root, args.dataset)

    device = set_device(args.gpu_id)
    args.device = device


    item_text_list = preprocess_text(args)

    plm_tokenizer, plm_name = load_plm(args.plm_checkpoint)

    if plm_tokenizer.pad_token_id is None:
        plm_tokenizer.pad_token_id = 0
    plm_name = plm_name.to(device)
    # ipdb.set_trace()
    generate_item_embedding(args, item_text_list,plm_tokenizer,
                            plm_name, word_drop_ratio=args.word_drop_ratio)
