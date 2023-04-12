
import time
start=time.time()
import json
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
    
end=time.time()
print('Running time: %s Seconds'%(end-start))