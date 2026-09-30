"""Explicit pilot/train/predict stages. Original experiment files are never edited."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time

# Initialize NumPy before PyTorch to avoid this Anaconda environment's OpenMP load-order conflict.
import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ['USE_TF'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('HF_HOME', str(ROOT / 'hf_cache'))


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {'id', 'text_az', 'text_ar'} <= set(rows[0]):
        raise ValueError(f'Invalid data file: {path}')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate row IDs: {path}')
    return rows


def clean(text):
    return ' '.join(text.split())


def check_hashes(data_dir, hashes, splits):
    for split in splits:
        actual = hashlib.sha256((data_dir / f'toy_{split}.csv').read_bytes()).hexdigest()
        if actual != hashes[split]:
            raise ValueError(f'{split} split differs from frozen data. Do not silently replace it.')


def pilot_indices(encoded, count, seed):
    n = len(encoded)
    chosen = {max(range(n), key=lambda i: len(encoded[i]['input_ids'])),
              max(range(n), key=lambda i: len(encoded[i]['labels']))}
    remaining = [i for i in range(n) if i not in chosen]
    chosen.update(random.Random(seed).sample(remaining, min(count-len(chosen), len(remaining))))
    return sorted(chosen)


def latest_complete_checkpoint(checkpoint_root):
    """Return the newest resumable checkpoint, ignoring partial Drive writes."""
    checkpoint_root = Path(checkpoint_root)
    candidates = []
    for path in checkpoint_root.glob('checkpoint-*'):
        match = re.fullmatch(r'checkpoint-(\d+)', path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    weight_names = {
        'model.safetensors', 'pytorch_model.bin',
        'model.safetensors.index.json', 'pytorch_model.bin.index.json',
    }
    required = {'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'}
    for _, path in sorted(candidates, reverse=True):
        names = {item.name for item in path.iterdir()}
        if required <= names and names & weight_names:
            return str(path)
    return None


def load_prediction_progress(path, rows):
    """Load the valid prefix of a JSONL prediction file and remove a partial tail."""
    path = Path(path)
    results = []
    if path.exists():
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                break
            if len(results) >= len(rows) or result.get('id') != rows[len(results)]['id']:
                raise ValueError('Prediction progress does not match the frozen test-row order.')
            results.append(result)
        with path.open('w', encoding='utf-8') as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pilot', 'train', 'predict'])
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent / 'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', action='store_true',
                        help='Resume full training or interrupted test prediction.')
    args = parser.parse_args()
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU available to PyTorch. Use a suitable GPU runtime; CPU training is disabled.')
    memory = torch.cuda.get_device_properties(0).total_memory / 2**30
    if memory < 12:
        raise SystemExit(f'GPU has {memory:.1f} GiB. This full-passage baseline requires a larger GPU pilot; no weights downloaded.')
    protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    if args.resume and args.mode == 'pilot':
        raise SystemExit('--resume is valid only with train or predict mode.')
    if args.output.exists():
        if not args.resume:
            raise SystemExit(f'Output directory already exists: {args.output}. Use a new run name or --resume.')
    else:
        if args.resume:
            raise SystemExit(f'Cannot resume because the output directory does not exist: {args.output}')
        if args.mode == 'train':
            free_gib = shutil.disk_usage(args.output.parent).free / 2**30
            if free_gib < 5.0:
                raise SystemExit(
                    f'Only {free_gib:.2f} GiB is free beside the output directory; '
                    'a fresh run requires at least 5.0 GiB free for complete checkpoints.')
        args.output.mkdir(parents=True)
    if args.mode == 'train' and args.resume:
        free_gib = shutil.disk_usage(args.output.parent).free / 2**30
        if free_gib < 3.0:
            raise SystemExit(
                f'Only {free_gib:.2f} GiB is free beside the output directory; '
                'a resumed run requires at least 3.0 GiB free for complete checkpoints.')
    hashes = protocol['split_sha256']
    check_hashes(args.data_dir, hashes, ['test'] if args.mode == 'predict' else ['train', 'val'])
    set_seed(protocol['seed'])
    device_info = {'gpu': torch.cuda.get_device_name(0), 'memory_gib': memory,
                   'torch': torch.__version__, 'mode': args.mode, 'protocol': protocol}
    manifest_path = args.output / 'run_manifest.json'
    if args.resume:
        if not manifest_path.exists():
            raise SystemExit('Cannot resume: run_manifest.json is missing from the output directory.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['mode'] != args.mode or manifest['protocol'] != protocol:
            raise SystemExit('Cannot resume: the existing run mode or frozen protocol does not match.')
        manifest.setdefault('resume_sessions', []).append(
            {'gpu': device_info['gpu'], 'memory_gib': memory, 'torch': torch.__version__})
    else:
        manifest = device_info
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.mode == 'predict':
        if not args.checkpoint:
            raise SystemExit('Provide the selected training checkpoint with --checkpoint.')
        run_record = json.loads((args.checkpoint / 'run_record.json').read_text(encoding='utf-8'))
        if run_record['mode'] != 'train' or run_record['protocol'] != protocol:
            raise SystemExit('Checkpoint must be a full training run with this exact frozen protocol, not a pilot.')
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to('cuda').eval()
        model.config.use_cache = True
        rows = read_rows(args.data_dir / 'toy_test.csv')
        final_path = args.output / 'test_predictions_mt5.csv'
        summary_path = args.output / 'generation_summary.json'
        partial_path = args.output / 'test_predictions_mt5.partial.jsonl'
        progress_path = args.output / 'generation_progress.json'
        if final_path.exists():
            print(json.loads(summary_path.read_text(encoding='utf-8')))
            return
        results = load_prediction_progress(partial_path, rows)
        previous_elapsed = 0.0
        if progress_path.exists():
            previous_elapsed = json.loads(progress_path.read_text(encoding='utf-8')).get('elapsed_seconds', 0.0)
        if results:
            print(f'Resuming predictions at row {len(results) + 1}/{len(rows)}', flush=True)
        started = time.perf_counter()
        with partial_path.open('a', encoding='utf-8') as partial, torch.inference_mode():
            for i in range(len(results), len(rows)):
                row = rows[i]
                batch = tokenizer(clean(row['text_az']).lower(), return_tensors='pt', truncation=False).to('cuda')
                generated = model.generate(
                    **batch, max_new_tokens=protocol['generation_max_new_tokens'],
                    num_beams=1, do_sample=False)
                ids = generated[0].tolist()
                result = {**row, 'gold_ar': clean(row['text_ar']),
                          'pred_mt5': clean(tokenizer.decode(ids, skip_special_tokens=True)),
                          'input_tokens': int(batch['input_ids'].shape[1]),
                          'reached_eos': tokenizer.eos_token_id in ids[1:]}
                results.append(result)
                partial.write(json.dumps(result, ensure_ascii=False) + '\n')
                partial.flush()
                if (i + 1) % 25 == 0 or i + 1 == len(rows):
                    elapsed = previous_elapsed + time.perf_counter() - started
                    progress = {'completed_rows': i + 1, 'total_rows': len(rows),
                                'elapsed_seconds': elapsed}
                    progress_path.write_text(json.dumps(progress, indent=2), encoding='utf-8')
                    print(f'Predicted {i + 1}/{len(rows)} rows', flush=True)
        with final_path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        summary = {'rows': len(results),
                   'elapsed_seconds': previous_elapsed + time.perf_counter() - started,
                   'outputs_without_eos': sum(not r['reached_eos'] for r in results),
                   'input_truncation': False, 'reference_truncation': False}
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(summary)
        return

    tokenizer = AutoTokenizer.from_pretrained(protocol['model'], revision=protocol['revision'], use_fast=False)
    encoded = {}
    for split in ['train', 'val']:
        rows = read_rows(args.data_dir / f'toy_{split}.csv')
        src = tokenizer([clean(r['text_az']).lower() for r in rows], truncation=False)
        tgt = tokenizer(text_target=[clean(r['text_ar']) for r in rows], truncation=False)
        encoded[split] = [{'input_ids': s, 'attention_mask': mask, 'labels': t}
                          for s, mask, t in zip(src['input_ids'], src['attention_mask'], tgt['input_ids'], strict=True)]
    if args.mode == 'pilot':
        selections = {split: pilot_indices(encoded[split], n, protocol['seed']) for split,n in [('train',64),('val',16)]}
        encoded = {split: [values[i] for i in selections[split]] for split,values in encoded.items()}
        (args.output / 'pilot_indices.json').write_text(json.dumps(selections, indent=2), encoding='utf-8')
    model = AutoModelForSeq2SeqLM.from_pretrained(protocol['model'], revision=protocol['revision'])
    model.config.use_cache = False
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output / 'checkpoints'), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=protocol['learning_rate'], optim='adafactor',
        num_train_epochs=protocol['epochs'], max_steps=10 if args.mode == 'pilot' else -1,
        gradient_checkpointing=True, bf16=torch.cuda.is_bf16_supported(), fp16=False,
        eval_strategy='steps', eval_steps=10 if args.mode == 'pilot' else 100,
        save_strategy='steps', save_steps=10 if args.mode == 'pilot' else 100,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss',
        greater_is_better=False, predict_with_generate=False, prediction_loss_only=True,
        logging_steps=1 if args.mode == 'pilot' else 25, report_to='none',
        seed=protocol['seed'], data_seed=protocol['seed'], dataloader_num_workers=0)
    trainer = Seq2SeqTrainer(model=model, args=training_args, processing_class=tokenizer,
                            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
                            train_dataset=encoded['train'], eval_dataset=encoded['val'],
                            callbacks=[] if args.mode == 'pilot' else [EarlyStoppingCallback(early_stopping_patience=6)])
    # mT5 ignores num_items_in_batch; Trainer must normalize accumulated microbatch losses.
    trainer.model_accepts_loss_kwargs = False
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = latest_complete_checkpoint(args.output / 'checkpoints')
        if resume_checkpoint is None:
            raise SystemExit('Cannot resume: no checkpoint containing model weights and complete Trainer state was found. Preserve this run and use a new output name.')
        print(f'Resuming from checkpoint: {resume_checkpoint}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trained = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(args.output / 'selected_model'))
    tokenizer.save_pretrained(args.output / 'selected_model')
    result = {'mode': args.mode, 'protocol': protocol, 'model_parameters': model.num_parameters(),
              'model_accepts_loss_kwargs': trainer.model_accepts_loss_kwargs,
              'resumed_from_checkpoint': resume_checkpoint,
              'loss_normalization': 'Mean non-padding token loss per passage; mean over accumulated microbatches.',
              'training_rows': len(encoded['train']), 'validation_rows': len(encoded['val']),
              'elapsed_seconds': time.perf_counter()-started,
              'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
              'best_checkpoint': trainer.state.best_model_checkpoint,
              'best_validation_loss': trainer.state.best_metric,
              'metrics': trained.metrics, 'log_history': trainer.state.log_history}
    (args.output / 'selected_model/run_record.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['protocol','log_history']}, indent=2))


if __name__ == '__main__':
    main()
