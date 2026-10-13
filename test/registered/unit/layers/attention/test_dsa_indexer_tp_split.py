"""CPU contract tests for the DSA prefill indexer row-split helpers.

`Indexer._indexer_tp_split_bounds` decides which extend rows each attn-TP rank
computes before one all_gatherv rebuilds the full top-k, so its bounds must tile
[0, q_offset) in rank order. `_chunk_request_rows` replaces the per-token
page-table copy of the chunked PAGED top-k with one row per request, so it must
map every chunk row to the same request as the per-token path.
"""

import unittest
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import DSAIndexerTPSplitMode, envs
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer, _chunk_request_rows
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsa.dsa_indexer"


def _bounds(seq_lens, extend_lens, tp, q_offset=None):
    return Indexer._indexer_tp_split_bounds(
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        extend_lens=extend_lens,
        q_offset=sum(extend_lens) if q_offset is None else q_offset,
        tp=tp,
    )


def _row_kv_lens(seq_lens, extend_lens):
    # Row j of a request with prefix p attends to p + j + 1 keys.
    return [
        seq - ext + j + 1
        for seq, ext in zip(seq_lens, extend_lens, strict=True)
        for j in range(ext)
    ]


class TestIndexerTPSplitBounds(CustomTestCase):
    CASES = [
        # (seq_lens, extend_lens, tp)
        ([8192], [8192], 8),
        ([4096, 100, 3000], [4096, 100, 1000], 4),
        ([65536, 70000], [1024, 1024], 8),
        ([9, 9, 9, 9, 9, 9, 9, 9], [1, 1, 1, 1, 1, 1, 1, 1], 8),
        ([1, 1, 1, 5000], [1, 1, 1, 5000], 2),
        ([2048, 2048], [2048, 2048], 3),
        # One dominant row: every KV-balanced cut collides, front or back.
        ([100000, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], 8),
        ([1, 1, 1, 1, 1, 1, 1, 100000], [1, 1, 1, 1, 1, 1, 1, 1], 8),
    ]

    def test_bounds_tile_rows_in_rank_order(self):
        # all_gatherv writes rank r's rows at offset sum(sizes[:r]); an empty or
        # unordered rank makes the gathered rows land in the wrong place.
        for seq_lens, extend_lens, tp in self.CASES:
            with self.subTest(seq_lens=seq_lens, extend_lens=extend_lens, tp=tp):
                q_offset = sum(extend_lens)
                bounds = _bounds(seq_lens, extend_lens, tp)
                self.assertEqual(len(bounds), tp + 1)
                self.assertEqual(bounds[0], 0)
                self.assertEqual(bounds[-1], q_offset)
                self.assertTrue(all(lo < hi for lo, hi in zip(bounds, bounds[1:])))

    def test_bounds_balance_kv_work(self):
        # The cut is by KV work, not row count: no rank may carry more than its
        # share plus one row, which a row-count split violates on causal rows.
        for seq_lens, extend_lens, tp in self.CASES:
            with self.subTest(seq_lens=seq_lens, extend_lens=extend_lens, tp=tp):
                bounds = _bounds(seq_lens, extend_lens, tp)
                row_kv = _row_kv_lens(seq_lens, extend_lens)
                total, heaviest = sum(row_kv), max(row_kv)
                for lo, hi in zip(bounds, bounds[1:]):
                    if hi - lo > 1:
                        self.assertLessEqual(sum(row_kv[lo:hi]), total / tp + heaviest)

    def test_balanced_cut_differs_from_row_count_cut(self):
        # One long causal extend: equal row counts give the last rank ~3x the
        # work of the first, so a KV-balanced cut must move the bounds forward.
        bounds = _bounds([8192], [8192], tp=4)
        self.assertNotEqual(bounds, [0, 2048, 4096, 6144, 8192])
        self.assertGreater(bounds[1], 2048)

    def test_mismatched_extend_lens_fall_back_to_row_count_cut(self):
        # extend lengths that do not sum to q_offset carry no per-row KV mapping;
        # the split must still tile all rows instead of indexing past them.
        self.assertEqual(
            _bounds([100, 100], [10, 10], tp=4, q_offset=40), [0, 10, 20, 30, 40]
        )


class TestChunkRequestRows(CustomTestCase):
    @staticmethod
    def _per_token_reference(extend_lens, start, end):
        token_to_batch_idx = [
            req for req, ext in enumerate(extend_lens) for _ in range(ext)
        ]
        return token_to_batch_idx[start:end]

    def _assert_matches_per_token(self, extend_lens, start, end):
        req_cu_cpu = list(accumulate(extend_lens, initial=0))
        req_cu = torch.tensor(req_cu_cpu, dtype=torch.int32)
        counts, rows = _chunk_request_rows(
            req_cu_cpu=req_cu_cpu, req_cu=req_cu, start=start, end=end
        )
        # The kernel maps chunk row i to the request whose cu_seqlens_q range
        # holds i; expanding the per-request counts rebuilds that map.
        requests = torch.arange(rows.start, rows.stop)
        self.assertEqual(counts.shape[0], requests.shape[0])
        self.assertTrue(bool((counts > 0).all()))
        self.assertEqual(
            torch.repeat_interleave(requests, counts).tolist(),
            self._per_token_reference(extend_lens, start, end),
        )
        # The input stays a device-side source: no in-place edits leak out.
        self.assertEqual(req_cu.tolist(), req_cu_cpu)

    def test_every_chunk_window_matches_per_token_gather(self):
        for extend_lens in ([5], [3, 1, 4], [1, 1, 1, 1], [7, 2, 9, 1]):
            q_offset = sum(extend_lens)
            for start in range(q_offset):
                for end in range(start + 1, q_offset + 1):
                    with self.subTest(extend_lens=extend_lens, start=start, end=end):
                        self._assert_matches_per_token(extend_lens, start, end)


class TestIndexerTPSplitMode(CustomTestCase):
    def _split(self, mode, *, in_graph, replaying):
        indexer = SimpleNamespace(
            dsa_enable_prefill_cp=False,
            _tp_split_in_graph_replay=lambda: replaying,
            _indexer_tp_split_bounds=Indexer._indexer_tp_split_bounds,
        )
        group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            pynccl_comm=SimpleNamespace(available=True),
        )
        batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor([4096], dtype=torch.int32),
            extend_seq_lens_cpu=[4096],
        )
        with (
            envs.SGLANG_DSA_INDEXER_TP_SPLIT.override(int(mode)),
            patch(f"{_INDEXER}._is_cuda", True),
            patch(f"{_INDEXER}.is_cp_active", return_value=False),
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=in_graph),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch(
                f"{_INDEXER}.get_parallel",
                return_value=SimpleNamespace(attn_tp_group=group),
            ),
        ):
            return Indexer._indexer_tp_split(
                indexer, forward_batch=batch, q_offset=4096
            )

    def test_graph_split_needs_the_replay_mode(self):
        # Eager-only must keep every graph path replicated: a collective issued
        # from a graph context it was not designed for can desync the ranks.
        M = DSAIndexerTPSplitMode
        expected = {
            # (mode, in_graph, replaying): split?
            (M.OFF, False, False): False,
            (M.EAGER, False, False): True,
            (M.EAGER, True, True): False,
            (M.EAGER_AND_BCG_REPLAY, True, False): False,
            (M.EAGER_AND_BCG_REPLAY, True, True): True,
        }
        for (mode, in_graph, replaying), split in expected.items():
            with self.subTest(mode=mode.name, in_graph=in_graph, replaying=replaying):
                result = self._split(mode, in_graph=in_graph, replaying=replaying)
                self.assertEqual(result is not None, split)

    def test_undefined_mode_raises(self):
        with self.assertRaises(ValueError):
            self._split(3, in_graph=False, replaying=False)


if __name__ == "__main__":
    unittest.main()
