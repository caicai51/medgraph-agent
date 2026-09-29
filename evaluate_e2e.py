"""Run deterministic end-to-end retrieval/generation regression checks."""
import argparse, json, os, re, statistics, time
from datetime import datetime, timezone
import requests

REFUSAL = "未检索到能够支持该问题的可靠医疗证据"

def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values)-1, int((len(values)-1)*fraction))] if values else 0

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='evaluation/e2e_regression.jsonl')
    parser.add_argument('--output',default='outputs/e2e_regression.json')
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--user',default='admin')
    args=parser.parse_args()
    cases=[json.loads(line) for line in open(args.dataset,encoding='utf-8') if line.strip()]
    rows=[]
    for case in cases:
        started=time.perf_counter()
        response=requests.post(args.base_url+'/stream',headers={'X-User-Id':args.user},json={'query':case['query'],'use_local':True,'local_model':'qwen2.5:7b'},timeout=90)
        response.raise_for_status()
        matches=re.findall(r'event: done\r?\ndata: (.+)',response.text)
        done=json.loads(matches[-1]) if matches else {}
        answer=done.get('content','')
        refused=REFUSAL in answer
        expected_ok=(not case['expected_any']) or any(x in answer for x in case['expected_any'])
        forbidden_hits=[x for x in case['forbidden'] if x in answer]
        passed=expected_ok and not forbidden_hits and refused==case['must_refuse']
        rows.append({'query_id':case['query_id'],'query':case['query'],'category':case['category'],'passed':passed,'refused':refused,'expected_ok':expected_ok,'forbidden_hits':forbidden_hits,'retrieval_ms':done.get('performance',{}).get('retrieval_ms'),'total_ms':done.get('performance',{}).get('total_ms'),'trace_steps':done.get('trace',{}).get('steps',[])})
    latencies=[r['total_ms'] for r in rows if isinstance(r['total_ms'],(int,float))]
    output={'run_at':datetime.now(timezone.utc).isoformat(),'dataset':args.dataset,'passed':sum(r['passed'] for r in rows),'total':len(rows),'pass_rate':sum(r['passed'] for r in rows)/len(rows),'mean_total_ms':statistics.mean(latencies),'p50_total_ms':percentile(latencies,.5),'p95_total_ms':percentile(latencies,.95),'results':rows}
    os.makedirs(os.path.dirname(args.output) or '.',exist_ok=True)
    with open(args.output,'w',encoding='utf-8') as file: json.dump(output,file,ensure_ascii=False,indent=2)
    print(json.dumps(output,ensure_ascii=False,indent=2))
    raise SystemExit(0 if output['passed']==output['total'] else 1)

if __name__=='__main__': main()
