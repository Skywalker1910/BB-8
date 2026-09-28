"""Interactive terminal chat for a locally trained BB8 checkpoint."""

import argparse

import torch

from inference.conversation import build_chat_prompt, extract_assistant_reply
from inference.model_loader import load_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a trained BB8 model")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--strategy", choices=["greedy", "temperature", "top_k", "top_p"], default="top_p")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    bundle = load_model_bundle(args.model_dir, args.checkpoint, args.device)
    generator = bundle.generator
    max_context = bundle.max_context_tokens or generator.model.max_seq_len
    messages: list[dict[str, str]] = []

    print(
        f"\nBB8 terminal chat | model={bundle.name} | "
        f"backend={bundle.backend} | device={bundle.device}"
    )
    print(f"Context window: {max_context} tokens")
    print("Commands: /reset clears history, /help shows commands, /exit closes chat")
    print(
        "Educational checkpoint: answers may be incorrect, invented, or repetitive.\n"
    )

    while True:
        try:
            message = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat closed.")
            break

        if not message:
            continue
        if message.lower() in {"/exit", "/quit"}:
            print("Chat closed.")
            break
        if message.lower() == "/reset":
            messages.clear()
            print("Conversation cleared.\n")
            continue
        if message.lower() == "/help":
            print("/reset  clear history\n/exit   close chat\n")
            continue

        request_messages = [*messages, {"role": "user", "content": message}]
        prompt, context_tokens = build_chat_prompt(
            request_messages,
            generator.tokenizer,
            max_context,
            bundle.prompt_style,
        )
        generated = generator.generate(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            strategy=args.strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            return_full_text=False,
        )
        reply = extract_assistant_reply(prompt, generated) or "(no text generated)"
        messages.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
        )
        print(f"BB8 > {reply}")
        print(f"      [{context_tokens}/{max_context} context tokens]\n")


if __name__ == "__main__":
    main()
