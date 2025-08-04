# 测试导入
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    print("✅ mamba-ssm 导入成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")