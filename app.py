"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/8/30 2:33 PM
"""
import time
from cmath import log
import json
from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import os
torch.cuda.is_available()
import datetime
from zero_shot import sort_get_sentence,get_3top_k
from transformers import AutoModelForQuestionAnswering, BertTokenizer
from transformers import pipeline
# 初始化模型
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device = 1)
retrieval_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device=2)


# 初始化qa模型


qa_model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
qa_tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
qa_model = pipeline("question-answering",model=qa_model, tokenizer=qa_tokenizer)

import openai

#初始化openai模型
# Set up the OpenAI API client
openai.api_key = "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"

# Set up the model and prompt
model_engine = "text-davinci-003"



#创建Flask对象app并初始化
app = Flask(__name__)
#通过python装饰器的方法定义路由地址
@app.route("/")
#定义方法 用jinjia2引擎来渲染页面，并返回一个index.html页面
def root():
    return render_template("index.html")

@app.route("/zero_shot_page")
def zero_shot_page():
    return render_template("zero_shot.html")

@app.route("/qa_submit")
def qa():
    return render_template("qa.html")

@app.route("/chatgpt_page")
def chatgpt_page():
    return render_template("chatgpt.html")


def zero_shot_show_result(keywords,topk):
    folder_path = '/home/huhao/nlp_web_test/templates/json/result/'
    json_path = os.path.join(folder_path,keywords+'.json')
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    end_results = []
    end_results.append("<h2 class='h2-heading'>您输入的关键词为{}<br>输入的topk为{}</h2><br>".format(keywords,topk))
    end_results.append("<h3 class='h2-heading'>输出结果如下：<br></h3>")
    tmp_results = []
    for sample in json_data:
        end_results.append("{}".format(sample+' 置信度：'+str(json_data[sample]['score'])+'<br>'))
        end_results.append("{}".format( json_data[sample]['ch']+'<br>'))
        end_results.append("***********************************************<br>")
        tmp_results.append("{}".format(sample+' 置信度：'+str(json_data[sample]['score'])))
        tmp_results.append("{}".format( json_data[sample]['ch']))
        if len(end_results)>int(topk)*3:
            break
    return end_results,tmp_results

def zero_shot_log(results,keywords,topk):
    now_time = datetime.datetime.now()
    log_dict = {}
    log_path = '/home/huhao/nlp_web_test/templates/json/log/zero_shot_log.json'
    with open(log_path, 'r') as f:
        json_data = json.load(f)
    index = len(json_data['log'])
    index +=1
    log_dict['time'] = str(now_time)
    log_dict['number'] = '#'+str(index)
    log_dict['keywords'] = keywords
    log_dict['topk'] = topk
    log_dict['results'] = results
    json_data['log'].append(log_dict)
    out_file = open(log_path, "w")
    json.dump(json_data, out_file, indent=6,ensure_ascii=False)
 
def get_list():
    json_path = '/home/huhao/nlp_web_test/templates/json/yiya_review.json'
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    reviews_list = json_data['reviews']
    end_results = []
    for product in tqdm(reviews_list):
        reviews = product['reviews']
        for tmp_sample in reviews:
            en_title = tmp_sample['en_title']
            en_content = tmp_sample['en_content']   
            en_sentence = en_title +'     '+ en_content
            end_results.append(en_sentence)
    return end_results
def get_sentence_similitay(contrast_list):
    """得到指定句子之间的相似度"""
    #Compute embedding for both lists
    sentence_to_embedding = {}
    for sample in contrast_list:
        embedding= retrieval_model.encode(sample, convert_to_tensor=True)
        sentence_to_embedding[sample] = embedding
    return sentence_to_embedding
contrast_list = get_list()
sentence_to_embedding = get_sentence_similitay(contrast_list) # 利用embedding方法做一个简单的排序


def whether_right(keywords,topk):
    folder_path = '/home/huhao/nlp_web_test/templates/json/result/'
    result_path = os.path.join(folder_path,keywords+'.json')
    if not os.path.exists(result_path):
        return False
    else:
        with open(result_path, 'r') as f:
            json_data = json.load(f)
        if len(json_data)>=3*int(topk):
            return True
        else:
            return False
#app的路由地址"/submit"即为ajax中定义的url地址，采用POST、GET方法均可提交
@app.route("/zero_shot_submit",methods=["GET", "POST"])
def zero_shot_submit():
    #从这里定义具体的函数 返回值均为json格式
    #由于POST、GET获取数据的方式不同，需要使用if语句进行判断
    if request.method == "POST":
        keywords = request.form.get("keywords")
        topk = request.form.get("topk")
    if request.method == "GET":
        keywords = request.args.get("keywords")
        topk = request.args.get("topk")
    #如果获取的数据为空
    if len(keywords) == 0 :
        return {'message':"您未输入关键词或句子"}
    if len(topk) == 0:
        topk = 10
    if whether_right(keywords,topk) == True:
        end_results ,tmp_results= zero_shot_show_result(keywords,topk)
    else:
        
        choose_en_list = get_3top_k(keywords,contrast_list,sentence_to_embedding,retrieval_model,topk)
        print(len(choose_en_list))
        start=time.time()
        sort_get_sentence(keywords,classifier,choose_en_list)
        end=time.time()
        print('Running time: %s Seconds'%(end-start))
        end_results ,tmp_results = zero_shot_show_result(keywords,topk)
        zero_shot_log(tmp_results,keywords,topk)

    return {'message':end_results}


@app.route("/qa_submit",methods=["GET", "POST"])
def qa_submit():
    #从这里定义具体的函数 返回值均为json格式
    #由于POST、GET获取数据的方式不同，需要使用if语句进行判断
    if request.method == "POST":
        question = request.form.get("keywords")
        context = request.form.get("topk")
    if request.method == "GET":
        question = request.args.get("keywords")
        context = request.args.get("topk")
    #如果获取的数据为空
    if len(question) == 0 :
        return {'message':"您未输入关键词或句子"}
    end_results = qa_model(question = question, context = context)['answer']
    return {'message':end_results}


@app.route("/chatgpt_submit",methods=["GET", "POST"])
def chatgpt_submit():
    #从这里定义具体的函数 返回值均为json格式
    #由于POST、GET获取数据的方式不同，需要使用if语句进行判断
    if request.method == "POST":
        question = request.form.get("keywords")
    if request.method == "GET":
        question = request.args.get("keywords")
    #如果获取的数据为空
    if len(question) == 0 :
        return {'message':"您未输入问题"}

    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=question,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    end_results = completion.choices[0].text
    return {'message':end_results}


# #定义app在8080端口运行
app.run(host="0.0.0.0",port=8080)
