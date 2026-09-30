"""Local CPU unit checks only; never download or train pretrained weights."""
import tempfile
from pathlib import Path
import hashlib
import subprocess
import sys
import unittest

from pretrained_baseline.run_baseline import (
    ROOT,
    check_hashes,
    clean,
    latest_complete_checkpoint,
    load_prediction_progress,
    pilot_indices,
    read_rows,
)


class BaselineTests(unittest.TestCase):
    def test_prediction_progress_ignores_partial_tail(self):
        rows = [{'id': 'a'}, {'id': 'b'}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.jsonl'
            path.write_text('{"id":"a","pred_mt5":"x"}\n{"id":', encoding='utf-8')
            results = load_prediction_progress(path, rows)
            self.assertEqual([result['id'] for result in results], ['a'])
            self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()), 1)

    def test_partial_checkpoint_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / 'checkpoint-200'
            partial.mkdir()
            for name in ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']:
                (partial / name).write_bytes(b'x')
            self.assertIsNone(latest_complete_checkpoint(root))
            (partial / 'model.safetensors').write_bytes(b'x')
            self.assertEqual(latest_complete_checkpoint(root), str(partial))

    def test_resume_cli_is_exposed(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'run_baseline.py'), '--help'],
            check=True, capture_output=True, text=True)
        self.assertIn('--resume', result.stdout)

    def test_normalization(self):
        self.assertEqual(clean('  a\n b  '), 'a b')

    def test_longest_examples_in_pilot(self):
        records = [{'input_ids': [1]*i, 'labels': [1]*(100-i)} for i in range(1, 100)]
        indexes = pilot_indices(records, 16, 42)
        self.assertEqual(len(indexes), 16)
        self.assertIn(0, indexes)
        self.assertIn(98, indexes)
        self.assertEqual(indexes, pilot_indices(records, 16, 42))

    def test_split_hash_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b'id,text_az,text_ar\na,x,y\n'
            (root / 'toy_train.csv').write_bytes(content)
            expected = hashlib.sha256(content).hexdigest()
            check_hashes(root, {'train': expected}, ['train'])
            with self.assertRaises(ValueError):
                check_hashes(root, {'train': 'wrong'}, ['train'])

    def test_duplicate_id_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            path.write_text('id,text_az,text_ar\na,x,y\na,z,w\n')
            with self.assertRaises(ValueError):
                read_rows(path)

    def test_tiny_mt5_forward_and_trainer_api(self):
        import torch
        from transformers import MT5Config, MT5ForConditionalGeneration, Seq2SeqTrainingArguments
        config = MT5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                           num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
                           pad_token_id=0, eos_token_id=1)
        model = MT5ForConditionalGeneration(config)
        loss = model(input_ids=torch.tensor([[2, 3, 1]]),
                     labels=torch.tensor([[4, 1, -100]])).loss
        self.assertTrue(torch.isfinite(loss))
        with tempfile.TemporaryDirectory() as directory:
            args = Seq2SeqTrainingArguments(output_dir=directory, use_cpu=True,
                                            eval_strategy='steps', save_strategy='steps',
                                            eval_steps=10, save_steps=10, optim='adafactor',
                                            load_best_model_at_end=True, report_to='none')
            self.assertEqual(args.eval_steps, 10)

    def test_accumulation_loss_normalization(self):
        import torch
        from transformers import MT5Config, MT5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
        config = MT5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                           num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
                           pad_token_id=0, eos_token_id=1, dropout_rate=0.0)
        model = MT5ForConditionalGeneration(config)
        batch = {'input_ids': torch.tensor([[2, 3, 1]]), 'labels': torch.tensor([[4, 1, -100]])}
        expected = model(**batch).loss.detach().item() / 2
        with tempfile.TemporaryDirectory() as directory:
            args = Seq2SeqTrainingArguments(output_dir=directory, use_cpu=True,
                                            gradient_accumulation_steps=2, report_to='none')
            trainer = Seq2SeqTrainer(model=model, args=args)
            trainer.model_accepts_loss_kwargs = False
            trainer.current_gradient_accumulation_steps = 2
            actual = trainer.training_step(model, batch).item()
            self.assertAlmostEqual(actual, expected, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
