
import json
from tqdm import tqdm
from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm
import torch
from sentence_transformers import  util
import os
torch.cuda.is_available()



def get_3top_k(keywords,contrast_list,sentence_to_embedding,retrieval_model,topk):
    """得到最有可能的前3*tok的句子"""
    sentence_to_similitary = {}
    for sample in tqdm(contrast_list):
        embedding1 = sentence_to_embedding[sample]
        embedding2 = retrieval_model.encode(keywords, convert_to_tensor=True)
        sentence_similitay = abs(float(round(util.pytorch_cos_sim(embedding1, embedding2).item(),8)))
        sentence_to_similitary[sample] = sentence_similitay
        sort_dict = sorted(sentence_to_similitary.items(),key = lambda item:item[1],reverse=True)
    index = 0
    choose_en_list = []
    for sample in sort_dict:
        index +=1
        if index>int(topk)*3:
            break
        else:
            choose_en_list.append(sample[0])
    return choose_en_list


def use_zero_classification(en_list,user_input,classifier):
    all_results = {}
    all_results[user_input] = {}
    for sample in tqdm(en_list):
        results = classifier(sample, user_input)
        all_results[results['labels'][0]][results['sequence']] = results['scores'][0] 
    
    tmp_sample = all_results[user_input]
    tmp_sort_sample = sorted(tmp_sample.items(),key = lambda item:item[1],reverse=True)
    all_results[user_input] = tmp_sort_sample
    
    return all_results



def sort_get_sentence(keywords,classifier,choose_en_list):
    folder_path = '/home/huhao/nlp_web_test/templates/json/result/'
    result_path = os.path.join(folder_path,keywords+'.json')
    json_path = '/home/huhao/nlp_web_test/templates/json/yiya_review.json'
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    reviews_list = json_data['reviews']
    en_to_ch = {}
    reviews_list = json_data['reviews']
    for product in reviews_list:
        reviews = product['reviews']
        for tmp_sample in reviews:
            en_title = tmp_sample['en_title']
            en_content = tmp_sample['en_content']   
            en_sentence = en_title +'     '+ en_content
            ch_sentence = tmp_sample['ch_title'] + tmp_sample['ch_content']
            en_to_ch[en_sentence] = ch_sentence
    
    tmp_results = use_zero_classification(choose_en_list,keywords,classifier)
    end_results = {}
    for sample in tmp_results[keywords]:
        end_results[sample[0]] = {}
        end_results[sample[0]]['score'] = sample[1]
        end_results[sample[0]]['ch'] = en_to_ch[sample[0]]
    out_file = open(result_path, "w")
    json.dump(end_results, out_file, indent=6,ensure_ascii=False)
    


# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device = 1)
# end_results = sort_get_sentence('quality',classifier,10)
# print(len(end_results))