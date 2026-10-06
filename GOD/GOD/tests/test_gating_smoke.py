"""Smoke test cho Micro Gating (Dynamic & Adaptive Layer Splitting).

Chạy (từ thư mục GOD/GOD):  python tests/test_gating_smoke.py
Dùng ViT nhỏ khởi tạo ngẫu nhiên (không tải pretrained) nên chạy được trên CPU trong vài giây.
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backbone.Lora_vit import VisionTransformer  # noqa: E402

DEPTH, K = 12, 6


def build(gating=True, gating_type="scalar"):
    cfg = SimpleNamespace(r=4, lora_alpha=1, lora_dropout=0.0, gating=gating,
                          gating_type=gating_type, gate_init_alpha=0.1)
    torch.manual_seed(0)
    return VisionTransformer(img_size=32, patch_size=16, embed_dim=48, depth=DEPTH, num_heads=4,
                             num_classes=0, tuning_config=cfg)


def setup_two_tasks(model):
    # Task 0: SL = LoRA task 0 (gán B != 0 để SL có đóng góp thực sự)
    model.add_task()
    for blk in model.blocks:
        for proj in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj):
            torch.nn.init.normal_(proj.lora_Bs[0], std=0.5)
    model.startEMA()
    # Task 1: mô phỏng đúng trình tự trong GOD._train
    model.add_task()
    for blk in model.blocks:  # TL_1 khác SL để kiểm tra pha trộn
        for proj in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj):
            torch.nn.init.normal_(proj.lora_Bs[1], std=0.5)
    for p in model.parameters():
        p.requires_grad = False
    model.unfreeze_lora([1], K)
    model.sync_shared_lora(1, K)
    model.gating_active = True


def test_scalar_gating():
    model = build(gating=True)
    setup_two_tasks(model)
    x = torch.randn(2, 3, 32, 32)

    # 1) Tầng nông (< K) của task 1 phải là bản sao Shared LoRA
    for blk in model.blocks[:K]:
        assert torch.equal(blk.attn.q_proj.lora_Bs[1], blk.attn.q_proj.lora_Bs[0])
    assert model.task_k[1] == K and model.select == K

    # 2) Gradient chỉ chảy vào gate của task 1 ở vùng TL (l >= K)
    feat, _ = model(x, [1])
    feat.sum().backward()
    for idx, blk in enumerate(model.blocks):
        g = blk.gates[1].logit.grad
        if idx >= K:
            assert g is not None and g.abs() > 0, "gate @block {} should get grad".format(idx)
        else:
            assert g is None, "gate @block {} must stay unused".format(idx)
        assert blk.gates[0].logit.grad is None, "task 0 has no gating"
        if idx < K:
            assert blk.attn.q_proj.lora_As[1].grad is None, "shallow TL must be frozen"

    # 3) α -> 0 phải trùng GOD gốc (thuần TL); α -> 1 phải trùng thuần SL
    with torch.no_grad():
        for blk in model.blocks:
            blk.gates[1].logit.fill_(-50.0)
        gated, _ = model(x, [1])
        model.gating_active = False
        plain, _ = model(x, [1])
        model.gating_active = True
        assert torch.allclose(gated, plain, atol=1e-5), "alpha=0 must equal original GOD"

        for blk in model.blocks:
            blk.gates[1].logit.fill_(50.0)
        gated_sl, _ = model(x, [1])
        sl_only, _ = model(x, [0])
        assert torch.allclose(gated_sl, sl_only, atol=1e-5), "alpha=1 must equal pure Shared LoRA"
        for blk in model.blocks:
            blk.gates[1].logit.fill_(0.0)

    # 4) EMA gate chỉ khởi tạo ở vùng TL của task hiện tại; Coarse + Refined chạy được
    model.EMA(0.9)
    for idx, blk in enumerate(model.blocks):
        assert bool(blk.gate_ema_ready) == (idx >= K)
    with torch.no_grad():
        feat_ema, sl_x = model(x, [], True)
        assert sl_x is not None and feat_ema.shape == (2, 48)
        refined = model.forward_SL(sl_x, [1])
        assert refined.shape == (2, 48)
        # Refined (SL_x + forward_SL) phải khớp forward đầy đủ vì các tầng < select đều là SL
        full, _ = model(x, [1])
        assert torch.allclose(full, refined, atol=1e-5), "refined path must match full forward"
    print("[OK] scalar gating | alphas:", model.get_gate_alphas(1))


def test_router_gating():
    model = build(gating=True, gating_type="router")
    setup_two_tasks(model)
    x = torch.randn(3, 3, 32, 32)
    feat, _ = model(x, [1])
    feat.sum().backward()
    assert model.blocks[-1].gates[1].router.weight.grad is not None
    print("[OK] router gating | alphas:", model.get_gate_alphas(1))


def test_gating_off_is_backward_compatible():
    model = build(gating=False)
    setup_two_tasks(model)
    assert not any(".gates." in n or "gate_ema" in n for n in model.state_dict().keys())
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        model(x, [1])
        model(x, [], True)
    print("[OK] gating=False keeps the original architecture")


if __name__ == "__main__":
    test_scalar_gating()
    test_router_gating()
    test_gating_off_is_backward_compatible()
    print("All gating smoke tests passed.")
