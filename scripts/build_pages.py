from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def prettify_title(file_stem: str) -> str:
    words = re.split(r"[_\-\s]+", file_stem.strip())
    words = [w for w in words if w]
    if not words:
        return "Untitled Plot"
    return " ".join(w.upper() if len(w) <= 3 else w.capitalize() for w in words)


def build_manifest(site_root: Path) -> dict:
    paper_dir = site_root / "Paper_results"
    html_files = sorted(paper_dir.rglob("*.html"))

    plots = []
    for html_file in html_files:
        rel_path = html_file.relative_to(site_root).as_posix()
        parent_rel = html_file.parent.relative_to(paper_dir).as_posix()
        group = "" if parent_rel == "." else parent_rel.replace("/", " / ")

        plots.append(
            {
                "title": prettify_title(html_file.stem),
                "path": rel_path,
                "group": group,
                "filename": html_file.name,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_plots": len(plots),
        "plots": plots,
    }


def copy_static_pages(pages_dir: Path, site_dir: Path) -> None:
    if not pages_dir.exists():
        raise FileNotFoundError(f"Missing pages directory: {pages_dir}")

    for item in pages_dir.iterdir():
        target = site_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pages_dir = repo_root / "pages"
    paper_results_dir = repo_root / "Paper_results"
    site_dir = repo_root / "site"
    temp_site_dir = repo_root / f"site_build_{uuid4().hex[:8]}"

    if not paper_results_dir.exists():
        raise FileNotFoundError(f"Missing Paper_results directory: {paper_results_dir}")

    temp_site_dir.mkdir(parents=True, exist_ok=True)

    copy_static_pages(pages_dir, temp_site_dir)
    shutil.copytree(paper_results_dir, temp_site_dir / "Paper_results")

    manifest = build_manifest(temp_site_dir)
    manifest_path = temp_site_dir / "plots-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (temp_site_dir / ".nojekyll").write_text("", encoding="utf-8")

    # Prefer a clean replace. If files are locked (common on Windows when previewing),
    # merge into existing site directory so builds still succeed.
    try:
        if site_dir.exists():
            shutil.rmtree(site_dir)
        temp_site_dir.replace(site_dir)
    except PermissionError:
        site_dir.mkdir(parents=True, exist_ok=True)
        for item in temp_site_dir.iterdir():
            dst = site_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        shutil.rmtree(temp_site_dir, ignore_errors=True)

    print(f"Built site at: {site_dir}")
    print(f"Discovered HTML plots: {manifest['total_plots']}")


if __name__ == "__main__":
    main()
