"""End-to-end Unsloth QLoRA finetune: Thadou-Kuki <-> English translator.

Steps: load 4-bit model -> LoRA -> SFT on out/train.jsonl (loss on assistant turns only)
       -> eval chrF++/BLEU on out/test.jsonl (unseen books) -> save LoRA -> merge -> GGUF + Ollama Modelfile.

Quick smoke test:   python train_unsloth.py --max_steps 30 --eval_n 20 --no_gguf
Full run (8GB GPU): python train_unsloth.py
48GB+ GPU (RunPod): python train_unsloth.py --bf16_base --batch 16 --grad_accum 1
"""
import argparse, json, os, random
from pathlib import Path

ROOT = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-2507")
p.add_argument("--max_seq", type=int, default=512)        # verses are short; 99%+ fit
p.add_argument("--lora_r", type=int, default=32)          # new language -> higher rank than usual
p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--max_steps", type=int, default=-1)       # overrides epochs when > 0
p.add_argument("--batch", type=int, default=4)
p.add_argument("--grad_accum", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--train_subset", type=int, default=0)     # 0 = all examples
p.add_argument("--eval_n", type=int, default=200)         # test examples to score after training
p.add_argument("--out", default=str(ROOT / "runs" / "thadou-qwen3-4b"))
p.add_argument("--gguf_quant", default="q4_k_m")
p.add_argument("--no_gguf", action="store_true")
p.add_argument("--resume", action="store_true")
p.add_argument("--bf16_base", action="store_true")      # 16-bit LoRA instead of QLoRA (needs ~2.5x VRAM)
args = p.parse_args()

from unsloth import FastLanguageModel  # must import before transformers/trl
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- 1. model ----------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model, max_seq_length=args.max_seq,
    load_in_4bit=not args.bf16_base, load_in_16bit=args.bf16_base)
model = FastLanguageModel.get_peft_model(
    model, r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=48)

# ---------- 2. data ----------
data = load_dataset("json", data_files={"train": str(ROOT / "out/train.jsonl"),
                                        "val": str(ROOT / "out/val.jsonl")})
if args.train_subset:
    data["train"] = data["train"].shuffle(seed=48).select(range(args.train_subset))


def to_text(batch):
    return {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]}


data = data.map(to_text, batched=True, remove_columns=["messages"])
print(data["train"][0]["text"])

# ---------- 3. train ----------
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=data["train"], eval_dataset=data["val"],
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=args.max_seq, packing=False,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        optim="adamw_8bit", weight_decay=0.01, seed=48,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10, eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=500, save_total_limit=2,
        output_dir=str(out_dir / "checkpoints"), report_to="none"),
)
# only learn the translation, not the prompt
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                  response_part="<|im_start|>assistant\n")
trainer.train(resume_from_checkpoint=args.resume or None)

model.save_pretrained(out_dir / "lora")
tokenizer.save_pretrained(out_dir / "lora")

# ---------- 4. evaluate on unseen books ----------
import sacrebleu

FastLanguageModel.for_inference(model)
test = [json.loads(l) for l in open(ROOT / "out/test.jsonl", encoding="utf-8")]
random.Random(48).shuffle(test)
res = {"t2e": ([], []), "e2t": ([], [])}
samples = []
for ex in test[:args.eval_n]:
    msgs, ref = ex["messages"][:-1], ex["messages"][-1]["content"]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to("cuda")
    gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=256,
                         do_sample=False, repetition_penalty=1.1)
    hyp = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
    d = "t2e" if "English:" in msgs[-1]["content"].split("\n")[0] or "in English" in msgs[-1]["content"] else "e2t"
    res[d][0].append(hyp); res[d][1].append(ref)
    samples.append({"dir": d, "src": msgs[-1]["content"], "ref": ref, "hyp": hyp})

report = {}
for d, (h, r) in res.items():
    if h:
        report[d] = {"n": len(h), "chrF++": round(sacrebleu.corpus_chrf(h, [r], word_order=2).score, 2),
                     "BLEU": round(sacrebleu.corpus_bleu(h, [r]).score, 2)}
print(json.dumps(report, indent=2))
(out_dir / "eval.json").write_text(json.dumps({"scores": report, "samples": samples[:50]},
                                              ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 5. export for offline use ----------
if not args.no_gguf:
    model.save_pretrained_gguf(str(out_dir / "gguf"), tokenizer, quantization_method=args.gguf_quant)
    gguf = next((out_dir / "gguf").glob("*.gguf"), None) or next(out_dir.glob("*.gguf"))
    (out_dir / "Modelfile").write_text(
        f'FROM ./{gguf.resolve().relative_to(out_dir.resolve()).as_posix()}\n'
        'TEMPLATE """{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
        '{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}'
        '<|im_start|>assistant\n"""\n'
        'SYSTEM "You are an expert translator between Thadou-Kuki (Thado Chin) and English."\n'
        'PARAMETER temperature 0.2\nPARAMETER stop "<|im_end|>"\n', encoding="utf-8")
    print(f"\nOllama: ollama create thadou -f {out_dir / 'Modelfile'}\n"
          '        ollama run thadou "Thadou-Kuki to English:\\n\\nPathen in vannoi angailut ahi."')
