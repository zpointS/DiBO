from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from src.trainer_dllm import (
    masked_entropy_mean,
    masked_importance_weighted_ce_sum,
    masked_reward_weighted_probability_mean,
)


class ChunkedCrossEntropyTest(unittest.TestCase):
    def test_chunked_ce_matches_dense_masked_reference(self) -> None:
        logits = torch.randn(2, 7, 19, dtype=torch.float32, requires_grad=True)
        input_ids = torch.randint(0, logits.size(-1), (2, 7), dtype=torch.long)
        masked = torch.tensor(
            [
                [True, False, True, True, False, False, True],
                [False, True, False, True, False, True, False],
            ],
            dtype=torch.bool,
        )
        p_mask = torch.rand(2, 7, dtype=torch.float32).clamp_min(0.05)

        got = masked_importance_weighted_ce_sum(logits, input_ids, masked, p_mask, chunk_size=3)
        ref = (
            F.cross_entropy(logits[masked], input_ids[masked], reduction="none")
            / p_mask[masked].clamp_min(1e-6)
        ).sum()
        self.assertTrue(torch.allclose(got, ref, atol=1e-6, rtol=1e-6))

        got.backward(retain_graph=True)
        got_grad = logits.grad.detach().clone()
        logits.grad = None
        ref.backward()
        ref_grad = logits.grad.detach().clone()
        self.assertTrue(torch.allclose(got_grad, ref_grad, atol=1e-6, rtol=1e-6))

    def test_masked_entropy_matches_dense_reference(self) -> None:
        logits = torch.randn(3, 5, 11, dtype=torch.float32)
        masked = torch.tensor(
            [
                [True, False, False, True, False],
                [False, True, True, False, False],
                [True, False, True, False, True],
            ],
            dtype=torch.bool,
        )

        got = masked_entropy_mean(logits, masked, chunk_size=2)
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = logprobs.exp()
        ref = (-(probs * logprobs).sum(dim=-1))[masked].mean()
        self.assertTrue(torch.allclose(got, ref, atol=1e-6, rtol=1e-6))

    def test_chunked_reward_probability_matches_dense_reference(self) -> None:
        logits = torch.randn(3, 6, 13, dtype=torch.float32, requires_grad=True)
        input_ids = torch.randint(0, logits.size(-1), (3, 6), dtype=torch.long)
        masked = torch.tensor(
            [
                [True, False, True, False, False, True],
                [False, True, False, True, False, False],
                [True, True, False, False, True, False],
            ],
            dtype=torch.bool,
        )
        reward = torch.tensor([0.5, -0.25, 1.5], dtype=torch.float32)

        got = masked_reward_weighted_probability_mean(
            logits,
            input_ids,
            masked,
            reward,
            chunk_size=2,
        )
        probs = torch.softmax(logits, dim=-1)
        p_correct = probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        batch_index = masked.nonzero(as_tuple=False)[:, 0]
        ref = (reward[batch_index] * p_correct[masked]).mean()
        self.assertTrue(torch.allclose(got, ref, atol=1e-6, rtol=1e-6))

        got.backward(retain_graph=True)
        got_grad = logits.grad.detach().clone()
        logits.grad = None
        ref.backward()
        ref_grad = logits.grad.detach().clone()
        self.assertTrue(torch.allclose(got_grad, ref_grad, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
