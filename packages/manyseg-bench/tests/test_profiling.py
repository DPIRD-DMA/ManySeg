import torch
from manyseg_bench.profiling import count_flops, profile_model


def test_flops_of_conv():
    m = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
    assert count_flops(m, torch.randn(1, 3, 10, 10)) == 2 * 8 * 3 * 9 * 100


def test_profile_cpu_smoke():
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.Conv2d(4, 2, 1))
    r = profile_model(m, in_channels=3, img_size=(32, 48), device="cpu", batch_sizes=(1, 2),
                      precision="fp32", warmup=1, iters=3, train_batch_size=2)
    assert r["params"] == sum(p.numel() for p in m.parameters())
    assert [i["batch_size"] for i in r["inference"]] == [1, 2]
    assert r["inference"][0]["images_per_s"] > 0
    assert r["train"]["images_per_s"] > 0
