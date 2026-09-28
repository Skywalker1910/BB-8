"""Complete native-chat SFT records: skip oversize records instead of corrupting them."""

import json
from pathlib import Path
from torch.utils.data import Dataset


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


class ChatDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.examples = []
        self.skipped_ids = []
        self.truncated_examples = 0
        for record in records:
            prefix = tokenizer.apply_chat_template(record['messages'], tokenize=True, add_generation_prompt=True)
            complete = tokenizer.apply_chat_template(
                [*record['messages'], {'role':'assistant','content':record['response']}],
                tokenize=True, add_generation_prompt=False)
            if complete[:len(prefix)] != prefix:
                raise ValueError('Chat template prefix mismatch; cannot safely mask assistant-only labels')
            if len(complete) > max_length:
                self.skipped_ids.append(record['id'])
                continue
            if len(complete) <= len(prefix) or tokenizer.eos_token_id not in complete[len(prefix):]:
                raise ValueError('Assistant response must include an end-of-turn token')
            self.examples.append({'input_ids':complete, 'attention_mask':[1]*len(complete),
                                  'labels':[-100]*len(prefix)+complete[len(prefix):]})
        if not self.examples:
            raise ValueError('No complete training examples fit the configured context')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
