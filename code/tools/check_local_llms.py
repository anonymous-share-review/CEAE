# list_hf_cache.py
import os
import csv
from pathlib import Path
from typing import Iterable, Optional
from huggingface_hub import scan_cache_dir, __version__ as HF_VER

def bytes2human(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"

def _as_iter(x) -> Iterable:
    if not x:
        return []
    # frozenset / list / tuple 都能迭代
    return x

def list_hf_cached_models(cache_dir: Optional[str] = None, csv_out: Optional[str] = None):
    info = scan_cache_dir(cache_dir=cache_dir)

    # 有的版本没有 cache_dir；我们就显示入参或默认推断
    cache_root = cache_dir or os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE") or "~/.cache/huggingface/hub"
    print(f"🤗 huggingface_hub = {HF_VER}")
    print(f"🗂 Cache root (guessed): {cache_root}")

    repos = getattr(info, "repos", [])  # 兼容：有的版本叫 repos
    print(f"📦 已缓存的仓库数量: {len(repos)}\n")

    rows = []
    grand_bytes = 0

    # 为了稳定输出顺序，按 (repo_type, repo_id) 排序（缺失字段给空串）
    def _key(r):
        return (getattr(r, "repo_type", "") or "", getattr(r, "repo_id", "") or "")
    for repo in sorted(repos, key=_key):
        repo_id = getattr(repo, "repo_id", "UNKNOWN")
        repo_type = getattr(repo, "repo_type", "model")

        file_count = 0
        bytes_on_disk = 0
        seen_blobs = set()
        file_paths_for_prefix = []

        for rev in _as_iter(getattr(repo, "revisions", [])):
            files = getattr(rev, "files", [])  # 部分版本是这个字段
            # 早期版本可能叫 "files_on_disk" 或类似；做个兜底
            if not files:
                files = getattr(rev, "files_on_disk", [])
            for f in _as_iter(files):
                file_count += 1
                # 优先用 blob_path 做去重；没有就退回 file_path
                blob = getattr(f, "blob_path", None) or getattr(f, "lfs_path", None) or getattr(f, "file_path", None)
                if blob and blob not in seen_blobs:
                    seen_blobs.add(blob)
                    size = getattr(f, "size_on_disk", None)
                    if isinstance(size, int):
                        bytes_on_disk += size
                    else:
                        # 再退回用实际文件大小（如果 file_path 存在）
                        fp = getattr(f, "file_path", None)
                        if fp and os.path.exists(fp):
                            try:
                                bytes_on_disk += os.path.getsize(fp)
                            except OSError:
                                pass
                # 收集 file_path 用来推断本地路径前缀（展示用）
                fp = getattr(f, "file_path", None)
                if fp:
                    file_paths_for_prefix.append(fp)

        # 尝试推断仓库本地路径（公共前缀）
        repo_path = ""
        if file_paths_for_prefix:
            try:
                repo_path = os.path.commonpath(file_paths_for_prefix)
            except ValueError:
                # 不同挂载点可能导致 commonpath 失败，保持为空即可
                repo_path = ""

        grand_bytes += bytes_on_disk

        print(f"🔹 Repo ID: {repo_id}")
        print(f"   ├─ 类型: {repo_type}")
        if repo_path:
            print(f"   ├─ 本地路径(推断): {repo_path}")
        print(f"   ├─ 文件数(含多 revision): {file_count}")
        print(f"   └─ 占用空间(去重后): {bytes2human(bytes_on_disk)}\n")

        rows.append({
            "repo_id": repo_id,
            "repo_type": repo_type,
            "repo_path_inferred": repo_path,
            "files_count_in_revisions": file_count,
            "unique_size_bytes": bytes_on_disk,
            "unique_size_human": bytes2human(bytes_on_disk),
        })

    print(f"📊 汇总占用(按各仓库内部去重后相加): {bytes2human(grand_bytes)}")

    if csv_out:
        csv_p = Path(csv_out)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["repo_id","repo_type","repo_path_inferred","files_count_in_revisions","unique_size_bytes","unique_size_human"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"✅ 已导出 CSV: {csv_p}")

if __name__ == "__main__":
    # 用默认缓存；如需指定，传 cache_dir="/home/you/.cache/huggingface/hub"
    list_hf_cached_models(csv_out="./hf_cache.csv")
