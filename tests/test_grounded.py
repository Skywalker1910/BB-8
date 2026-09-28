"""Leakage checks, retrieval contracts, safe formatting and full-example labels."""

from pathlib import Path
import pytest

from grounded.core import ABSTAIN, Retriever, assess, format_prompt, retrieval_query
from grounded.data import behavior_examples
from grounded.evaluate import verify_data
from grounded.training import ChatDataset, read_jsonl
from fine_tune import InstructionDataset


def test_dataset_frozen_and_groups_disjoint():
    root = Path('data/grounded_v1')
    verify_data(root)
    train = read_jsonl(root/'train.jsonl')
    validation = read_jsonl(root/'validation.jsonl')
    evaluation = read_jsonl(root/'evaluation.jsonl')
    assert len(train)==192 and len(validation)==48 and len(evaluation)==88
    assert not {r['group'] for r in train} & {r['group'] for r in validation}
    assert not {r['group'] for r in evaluation} & {r['group'] for r in train}
    assert all(r['synthetic'] for r in train)


def test_retriever_deterministic_and_no_match():
    retriever = Retriever([{'id':'a','title':'Python','text':'A uses Python.'},
                           {'id':'b','title':'Ruby','text':'B uses Ruby.'}])
    assert retriever.search('Python')[0]['id']=='a'
    assert retriever.search('zzzzzz')==[]
    assert retriever.search('Python')==retriever.search('Python')
    with pytest.raises(ValueError): retriever.search('Python',0)


def test_followup_query_ignores_model_claims():
    query = retrieval_query('Where is it hosted?', [{'role':'user','content':'Tech-Portfolio'},
                                                   {'role':'assistant','content':'Invented claim'}])
    assert 'Tech-Portfolio' in query and 'Invented' not in query


def test_checks_reject_unknown_citations_and_require_support():
    case = {'answerable':True,'gold_ids':['a'],'keywords':['Python']}
    assert assess(case,'Python [a]',[{'id':'a'}])['pass']
    assert not assess(case,'Python [a] [invented]',[{'id':'a'}])['pass']
    assert not assess(case,'Python',[])['pass']
    assert assess({'answerable':False},ABSTAIN,[])['pass']


class ChatTokenizer:
    eos_token_id = 1
    def encode(self,text,**kwargs): return list(text.encode())
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        text=''.join(f"{m['role']}: {m['content']}\n" for m in messages)
        if add_generation_prompt: text+='assistant: '
        if not tokenize: return text
        result=list(text.encode())
        if not add_generation_prompt: result.append(1)
        return result


def test_full_chat_examples_preserve_eos_mask_and_skip_oversize():
    records=behavior_examples('unit',1)
    tokenizer=ChatTokenizer()
    dataset=ChatDataset(records,tokenizer,3000)
    first=dataset[0]
    prefix=tokenizer.apply_chat_template(records[0]['messages'],tokenize=True,add_generation_prompt=True)
    assert first['input_ids'][:len(prefix)]==prefix
    assert first['labels'][:len(prefix)]==[-100]*len(prefix)
    assert first['labels'][-1]==1
    with pytest.raises(ValueError,match='No complete'): ChatDataset(records,tokenizer,32)


def test_prompt_budget_never_slices_latest_question():
    tokenizer=ChatTokenizer()
    with pytest.raises(ValueError,match='Latest question'):
        format_prompt(tokenizer,'large'*100,[],max_tokens=32)
    prompt,evidence,_,count=format_prompt(tokenizer,'Question',
        [{'id':'a','title':'Long','text':'X'*1000}],max_tokens=600)
    assert 'Question' in prompt and evidence==[] and count<=600


def test_corrected_dolly_preprocessing_never_loses_response_marker():
    class Tokenizer:
        eos_token_id=1
        def encode(self,text,**kwargs): return list(text.encode())
    record={'instruction':'x'*40,'context':'','response':'ok'}
    corrected=InstructionDataset([record],Tokenizer(),128)
    legacy=InstructionDataset([record],Tokenizer(),128,preprocessing='legacy_v1')
    assert len(corrected[0]['input_ids']) > len(legacy[0]['input_ids'])
    assert corrected[0]['labels'][-1]==1
    skipped=InstructionDataset([record],Tokenizer(),32)
    assert len(skipped)==0 and skipped.skipped_ids==[0]


def test_serving_citation_warnings_do_not_fabricate_sources():
    from grounded.service import citation_status
    result = citation_status('Python [invented]', [{'id':'real'}])
    assert result['invalid_ids']==['invented'] and result['warnings']
    assert citation_status('Python',[])['warnings']
    assert not citation_status(ABSTAIN,[])['warnings']


def test_grounded_api_rejects_non_boolean_switch(monkeypatch):
    import json
    from types import SimpleNamespace
    from api import handler
    monkeypatch.setenv('BB8_API_KEY','')
    monkeypatch.setattr(handler,'chat_bundle',lambda model:SimpleNamespace(
        generator=SimpleNamespace(),max_context_tokens=768,prompt_style='chat_template'))
    event={'rawPath':'/chat','httpMethod':'POST','body':json.dumps({
        'messages':[{'role':'user','content':'hi'}],'grounded':'false'})}
    assert handler.lambda_handler(event,None)['statusCode']==400


def test_grounded_api_returns_supplied_sources_and_warnings(monkeypatch):
    import json
    from types import SimpleNamespace
    from api import handler
    from grounded import service
    class Generator:
        def generate(self, **kwargs): return 'Python [missing]'
        def generate_with_details(self, **kwargs):
            return {'text':'Python [missing]','generated_tokens':4,'stop_reason':'eos'}
    bundle=SimpleNamespace(name='test',generator=Generator(),max_context_tokens=768,
                           prompt_style='chat_template',backend='hf_lora')
    monkeypatch.setenv('BB8_API_KEY','')
    monkeypatch.setattr(handler,'chat_bundle',lambda model:bundle)
    monkeypatch.setattr(service,'prepare',lambda model,messages:('prompt',10,[{'id':'real','text':'Python'}]))
    event={'rawPath':'/chat','httpMethod':'POST','body':json.dumps({
        'messages':[{'role':'user','content':'language?'}],'grounded':True})}
    response=handler.lambda_handler(event,None)
    assert response['statusCode']==200
    body=json.loads(response['body'])
    assert body['reply']=='Python [missing]'  # Never silently replace with a canned answer.
    assert body['sources'][0]['id']=='real'
    assert body['citation_checks']['invalid_ids']==['missing']
    assert body['generated_tokens']==4
