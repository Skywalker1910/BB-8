"""Frozen 2x2 evaluation: Instruct/LoRA crossed with no evidence/BM25 evidence.

python -m grounded.evaluate [--adapter checkpoints/bb8-grounded-v005-pilot]
"""

import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time
import uuid

import torch

from grounded.core import Retriever, assess, format_prompt, retrieval_query
from grounded.training import read_jsonl
from experiments.tracking import git_commit, git_is_dirty, runtime_info, sha256_file


def verify_data(directory):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        if sha256_file(str(directory/name)) != digest:
            raise ValueError(f'Frozen dataset changed: {name}; create a new version')
    return manifest


def summarize(rows):
    profiles = {}
    for profile in dict.fromkeys(r['profile'] for r in rows):
        selected = [r for r in rows if r['profile'] == profile]
        answerable = [r for r in selected if r['answerable']]
        unknown = [r for r in selected if not r['answerable']]
        citation_count = sum(len(r['checks']['citations']) for r in selected)
        invalid = sum(len(r['checks']['invalid_citations']) for r in selected)
        categories = {}
        for category in dict.fromkeys(r['category'] for r in selected):
            group = [r for r in selected if r['category'] == category]
            categories[category] = {'passed':sum(r['checks']['pass'] for r in group),'total':len(group)}
        profiles[profile] = {
            'cases':len(selected), 'answerable_passes':sum(r['checks']['pass'] for r in answerable),
            'answerable_total':len(answerable),
            'answerable_keyword_matches':sum(r['checks']['keywords_ok'] and not r['checks']['abstained'] for r in answerable),
            'answerable_pass_rate':sum(r['checks']['pass'] for r in answerable)/len(answerable),
            'unanswerable_abstentions':sum(r['checks']['abstained'] for r in unknown),
            'unanswerable_total':len(unknown),
            'unanswerable_abstention_rate':sum(r['checks']['abstained'] for r in unknown)/len(unknown),
            'invalid_citations':invalid, 'citation_count':citation_count,
            'citation_id_precision':(citation_count-invalid)/citation_count if citation_count else None,
            'retrieval_recall_at_3':sum(r['retrieval_hit'] for r in answerable)/len(answerable) if profile in {'base_rag','lora_rag'} else None,
            'median_latency_ms':statistics.median(r['latency_ms'] for r in selected),
            'categories':categories,
        }
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter')
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--data',default='data/grounded_v1')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    directory = Path(args.data)
    data_manifest = verify_data(directory)
    corpus = read_jsonl(directory/'corpus.jsonl')
    cases = read_jsonl(directory/'evaluation.jsonl')
    retriever = Retriever(corpus)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    output = Path('outputs/grounded')/run_id
    output.mkdir(parents=True)
    print(f'RUN_DIR={output}',flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    revision = '7ae557604adf67be50417f59c2c2f167def9a775'
    base = 'Qwen/Qwen2.5-0.5B-Instruct'
    options = dict(revision=revision,cache_dir='.cache/huggingface',local_files_only=True,trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(base,**options)
    model = AutoModelForCausalLM.from_pretrained(base,**options,
        dtype=torch.bfloat16 if args.device=='cuda' else torch.float32).to(args.device)
    if args.adapter:
        from peft import PeftModel
        metadata = json.loads((Path(args.adapter)/'bb8_model_config.json').read_text())
        if metadata['base_model']['revision'] != revision:
            raise ValueError('Adapter does not match the evaluation base revision')
        model = PeftModel.from_pretrained(model,args.adapter)
    model.eval()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    profiles = ['base_no_rag','base_rag'] + (['lora_no_rag','lora_rag'] if args.adapter else [])
    manifest = {'run_id':run_id,'base_model':base,'base_revision':revision,'profiles':profiles,
                'dataset_manifest':data_manifest,'git_commit':git_commit(),'git_dirty':git_is_dirty(),
                'runtime':runtime_info(torch.device(args.device)),
                'decoding':{'strategy':'greedy','repetition_penalty':1.,'max_new_tokens':96,'input_budget':768},
                'adapter':args.adapter,'adapter_sha256':sha256_file(str(Path(args.adapter)/'adapter_model.safetensors')) if args.adapter else None,
                'limitations':'Agent-authored pilot. Keyword/citation-ID checks are not entailment or factual accuracy; human review required.',
                'source_hashes':{p:sha256_file(p) for p in ['grounded/core.py','grounded/data.py','grounded/evaluate.py','grounded/training.py']}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for name in manifest['source_hashes']:
        destination = output/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(name,destination)
    rows = []
    with (output/'results.jsonl').open('x',encoding='utf-8') as handle, torch.inference_mode():
        for profile in profiles:
            use_rag = profile in {'base_rag','lora_rag'}
            with model.disable_adapter() if args.adapter and profile.startswith('base_') else nullcontext():
                for index,case in enumerate(cases):
                    query = retrieval_query(case['question'],case['history'])
                    retrieved = retriever.search(query) if use_rag else []
                    if case.get('inject'):
                        retrieved = [{**d,'text':d['text']+'\n'+case['inject']} for d in retrieved]
                    prompt, visible, history, count = format_prompt(tokenizer,case['question'],retrieved,case['history'])
                    inputs = tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(args.device)
                    if args.device=='cuda': torch.cuda.synchronize()
                    started = time.perf_counter()
                    generated = model.generate(**inputs,max_new_tokens=96,do_sample=False,repetition_penalty=1.,
                        pad_token_id=tokenizer.pad_token_id,eos_token_id=model.generation_config.eos_token_id)
                    if args.device=='cuda': torch.cuda.synchronize()
                    elapsed = round((time.perf_counter()-started)*1000,2)
                    ids = generated[0,inputs.input_ids.shape[1]:].tolist()
                    answer = tokenizer.decode(ids,skip_special_tokens=True).strip()
                    row = {'profile':profile,'case_id':case['id'],'category':case['category'],
                           'answerable':case['answerable'],'question':case['question'],'answer':answer,
                           'retrieval_query':query,'retrieved_ids':[d['id'] for d in retrieved],
                           'visible_ids':[d['id'] for d in visible],'formatted_prompt':prompt,
                           'input_tokens':count,'generated_ids':ids,'generated_tokens':len(ids),
                           'latency_ms':elapsed,'retrieval_hit':bool(set(case['gold_ids']) & {d['id'] for d in visible}),
                           'checks':assess(case,answer,visible)}
                    rows.append(row)
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n'); handle.flush()
                    print(f"{profile} {index+1}/{len(cases)} {case['id']} pass={row['checks']['pass']}",flush=True)
    stats = summarize(rows)
    gates = data_manifest['promotion_gates']
    candidate = stats.get('lora_rag')
    numeric_gates = bool(candidate and candidate['answerable_pass_rate'] >= gates['answerable_pass_rate_min']
        and candidate['unanswerable_abstention_rate'] >= gates['unanswerable_abstention_min']
        and candidate['invalid_citations'] <= gates['invalid_citations_max']
        and candidate['answerable_pass_rate'] >= stats['base_rag']['answerable_pass_rate']
        and candidate['unanswerable_abstention_rate'] >= stats['base_rag']['unanswerable_abstention_rate'])
    summary = {'run_id':run_id,'profiles':stats,'numeric_gates_passed':numeric_gates,
               'promoted':False,'human_review':'pending','output_dir':str(output)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines = ['# Grounded BB8 pilot results','','Keyword/citation checks are not general factual accuracy. Human claim review is pending.','',
             '| Profile | Answerable checks | Unknown abstentions | Invalid citation IDs | Retrieval recall@3 |',
             '|---|---:|---:|---:|---:|']
    for name,s in stats.items():
        lines.append(f"| {name} | {s['answerable_passes']}/{s['answerable_total']} | {s['unanswerable_abstentions']}/{s['unanswerable_total']} | {s['invalid_citations']} | {s['retrieval_recall_at_3']} |")
    lines += ['',f'Numeric promotion gates passed: {numeric_gates}. No automatic deployment or promotion.','',
              '## Raw responses','']
    for row in rows:
        lines += [f"### {row['profile']} / {row['case_id']}",'',row['question'],'','```text',row['answer'],'```','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    with Path('experiments/grounded_registry.jsonl').open('a',encoding='utf-8') as registry:
        registry.write(json.dumps(summary)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
