import torch

import fa_tools as fa


def test_gpu_memory_stats_empty_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert fa.gpu_memory_stats() == {}


def test_gpu_memory_stats_reports_gb_and_resets_peak(monkeypatch):
    gb = 1024**3
    resets = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 6 * gb)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 10.5 * gb)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 12 * gb)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: resets.append(1))

    assert fa.gpu_memory_stats() == {"gpu_mem_alloc_gb": 6.0, "gpu_mem_peak_gb": 10.5, "gpu_mem_reserved_gb": 12.0}
    assert resets == []
    fa.gpu_memory_stats(reset_peak=True)
    assert resets == [1]
