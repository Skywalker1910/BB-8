"""Generate text from a trained BB8 checkpoint."""

import argparse

from inference.model_loader import load_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a trained BB8 model")
    parser.add_argument("--model-dir", required=True, help="Directory containing model artifacts")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument(
        "--strategy",
        choices=["greedy", "temperature", "top_k", "top_p"],
        default="top_p",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    bundle = load_model_bundle(
        args.model_dir,
        checkpoint_name=args.checkpoint,
        device=args.device,
    )
    text = bundle.generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"\nModel: {bundle.name} | device: {bundle.device}\n")
    print(text)


if __name__ == "__main__":
    main()

