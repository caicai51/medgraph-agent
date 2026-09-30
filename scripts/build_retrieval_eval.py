"""Build a reviewable weak-label retrieval set aligned to real Milvus ids."""
import argparse
import json
import os
import random
from pymilvus import Collection, connections

CASES = [
    ("q001", "感冒一般会有哪些表现？", "感冒", ("symptom", "symptoms", "症状"), "synonym", "easy"),
    ("q002", "得了感冒通常如何处理？", "感冒", ("treat", "treatment", "治疗", "overview", "简介"), "paraphrase", "easy"),
    ("q003", "高血压病人平时吃东西要注意什么？", "高血压", ("diet", "food", "宜吃", "忌吃", "饮食"), "colloquial", "medium"),
    ("q004", "糖尿病通常要做哪些化验？", "糖尿病", ("check", "exam", "检查"), "paraphrase", "medium"),
    ("q005", "流行性感冒怎样预防？", "流行性感冒", ("prevent", "预防"), "direct", "easy"),
    ("q006", "胃溃疡通常由什么因素引起？", "胃溃疡", ("cause", "病因"), "synonym", "medium"),
    ("q007", "冠心病常用哪些药物？", "冠心病", ("drug", "drugs", "药"), "professional", "medium"),
    ("q008", "痛风患者哪些食物最好别吃？", "痛风", ("not_eat", "忌吃", "diet", "food"), "colloquial", "hard"),
]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--collection',default='medical_qa_vectors'); p.add_argument('--output',default='evaluation/retrieval_weak.jsonl'); p.add_argument('--seed',type=int,default=20260910); a=p.parse_args()
    connections.connect(host=os.getenv('MILVUS_HOST','localhost'),port=os.getenv('MILVUS_PORT','19530'))
    c=Collection(a.collection); rows=[]
    for qid,query,disease,markers,category,difficulty in CASES:
        docs=c.query(expr=f'document_id == "{disease}"',output_fields=['id','document_id','content','metadata'],limit=256)
        relevant=[str(d['id']) for d in docs if any(m.lower() in str(d['id']).lower() or m in str(d.get('metadata',{})) for m in markers)]
        rows.append({'query_id':qid,'query':query,'relevant_doc_ids':sorted(set(relevant)),'relevant_answers':[],'disease_filter':disease,'category':category,'difficulty':difficulty,'source':'weak:disease_and_type','split':'validation' if int(qid[-1])%3 else 'test'})
    random.Random(a.seed).shuffle(rows); os.makedirs(os.path.dirname(a.output),exist_ok=True)
    with open(a.output,'w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'collection':a.collection,'cases':len(rows),'labeled_cases':sum(bool(x['relevant_doc_ids']) for x in rows),'labels':sum(len(x['relevant_doc_ids']) for x in rows),'seed':a.seed},ensure_ascii=False))
if __name__=='__main__': main()
