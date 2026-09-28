"""Evidence retrieval shared with evaluation, with an auditable response contract."""

from pathlib import Path
import json
import re

from grounded.core import ABSTAIN, Retriever, format_prompt, retrieval_query
from grounded.evaluate import verify_data
from grounded.training import read_jsonl


def prepare(bundle, messages):
    if bundle.prompt_style != 'chat_template':
        raise ValueError('Grounded mode requires a native-chat model; select a grounded baseline or adapter')
    directory = Path(__file__).resolve().parents[1] / 'data/grounded_v1'
    verify_data(directory)
    corpus = read_jsonl(directory/'corpus.jsonl')
    question, history = messages[-1]['content'], messages[:-1]
    retrieved = Retriever(corpus).search(retrieval_query(question,history))
    prompt, visible, _, count = format_prompt(bundle.generator.tokenizer,question,retrieved,history,
                                             max_tokens=bundle.max_context_tokens)
    return prompt, count, visible


def citation_status(answer, documents):
    ids = re.findall(r'\[([A-Za-z0-9_-]+)\]',answer)
    valid = {d['id'] for d in documents}
    invalid = [ident for ident in ids if ident not in valid]
    warnings = []
    if invalid: warnings.append('The model cited an ID that was not supplied as evidence.')
    if ABSTAIN.lower() not in answer.lower() and not ids:
        warnings.append('The model answered without a source citation; verify the answer.')
    return {'cited_ids':ids,'invalid_ids':invalid,'warnings':warnings,
            'notice':'A valid citation ID does not prove that the cited text supports every claim.'}
