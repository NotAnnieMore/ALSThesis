"""Static LaTeX and aggregate-regression checks for the reviewer revision."""

from collections import Counter
import argparse
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "overleaf"


def active_text(path):
    text = path.read_text(encoding="utf-8-sig")
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", text, flags=re.S)
    return text.split(r"\end{document}", 1)[0]


def collect(path, seen=None):
    seen = set() if seen is None else seen
    path = path.resolve()
    # Some front-matter files are intentionally input repeatedly.
    text = active_text(path)
    seen.add(path)
    for raw in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
        if "#" in raw:
            continue
        child = TEX / raw
        if not child.suffix:
            child = child.with_suffix(".tex")
        assert child.exists(), f"Missing input: {raw} from {path}"
        if child.resolve() not in seen:
            collect(child, seen)
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-update", type=Path)
    args = parser.parse_args()
    files = collect(TEX / "main.tex")
    texts = {p:active_text(p) for p in files}
    labels = Counter(label for t in texts.values() for label in re.findall(r"\\label\{([^}]+)\}",t))
    references = {label for t in texts.values() for label in re.findall(r"\\(?:ref|eqref|pageref|autoref)\{([^}]+)\}",t)}
    assert not references-set(labels), f"Undefined labels: {references-set(labels)}"
    assert not [k for k,v in labels.items() if v>1], "Duplicate active labels"
    bibliography = (TEX / "references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"(?m)^@\w+\s*\{\s*([^,\s]+)",bibliography)
    assert len(keys)==len(set(keys)), "Duplicate bibliography keys"
    citations = {k.strip() for t in texts.values() for group in re.findall(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}",t) for k in group.split(",")}
    assert not citations-set(keys), f"Undefined citations: {citations-set(keys)}"
    glossary = {k for t in texts.values() for k in re.findall(r"\\(?:newglossaryentry|newacronym)\{([^}]+)\}",t)}
    uses = {k for t in texts.values() for k in re.findall(r"\\(?:gls|Gls|glspl|Glspl|acrshort|acrlong)\*?\{([^}]+)\}",t)}
    assert not uses-glossary, f"Undefined glossary entries: {uses-glossary}"
    for path,t in texts.items():
        stripped = re.sub(r"\\[{}]","",t)
        depth = 0
        for ch in stripped:
            depth += (ch=="{")-(ch=="}")
            assert depth>=0, f"Closing brace without opening brace: {path}"
        assert depth==0, f"Unbalanced braces: {path}, depth {depth}"
        for raw in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",t):
            if "#" in raw or "\\" in raw:
                continue
            image = TEX / raw
            candidates = [image] if image.suffix else [image.with_suffix(s) for s in [".pdf",".png",".jpg",".jpeg"]]
            assert any(p.exists() for p in candidates), f"Missing graphic: {raw}"
    abstracts = {}
    def words(t):
        t = re.sub(r"\\[A-Za-z]+\*?", "",t).replace("~"," ").replace("{"," ").replace("}"," ")
        return len(t.split())
    for lang in ["EN","PT"]:
        t = active_text(TEX / f"Chapters/abstract/abstract-{lang}.tex")
        title = t.split(r"\textbf{",1)[1].split("}",1)[0]
        body = t.split(r"\label{",1)[1].split("}",1)[1].split(r"\section*{",1)[0]
        abstracts[lang] = dict(body=words(body),with_title=words(title+" "+body))
        assert abstracts[lang]["with_title"]<=150
    # Preserve manually inserted page/float controls in the original chapters.
    for path in (TEX / "Chapters").glob("[1-8]-*.tex"):
        relative = path.relative_to(ROOT).as_posix()
        old = subprocess.check_output(["git","-c",f"safe.directory={ROOT.as_posix()}","show",f"HEAD:{relative}"],cwd=ROOT).decode("utf-8")
        current = path.read_text(encoding="utf-8")
        for control in [r"\clearpage",r"\cleardoublepage",r"\newpage",r"\break"]:
            assert old.count(control)==current.count(control), f"Page control changed: {path} {control}"
    old_bagging = pd.read_csv(ROOT / "replication/03_outputs/step7/step7c_bonferroni_results.csv").set_index("classifier")
    new_bagging = pd.read_csv(ROOT / "04_outputs/reviewer_followup/review_s2_bagging_summary.csv").set_index("classifier")
    assert np.allclose(old_bagging.bal_acc_single_mean,new_bagging.single_balanced_accuracy,rtol=0,atol=1e-12)
    assert np.allclose(old_bagging.bal_acc_ens_mean,new_bagging.bagging_balanced_accuracy,rtol=0,atol=1e-12)
    result = dict(active_tex_files=len(files),labels=len(labels),citations=len(citations),
                  glossary_uses=len(uses),abstract_words=abstracts,
                  bagging_original_balanced_accuracy_reproduced=True,
                  note="Static source checks, not a LaTeX compilation or pagination check.")
    (ROOT / "04_outputs/reviewer_followup/review_validation.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    prefix = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    changed = subprocess.check_output(prefix+["diff","--name-only","-z","HEAD","--","overleaf"],cwd=ROOT).decode("utf-8").split("\0")
    added = subprocess.check_output(prefix+["ls-files","--others","--exclude-standard","-z","--","overleaf"],cwd=ROOT).decode("utf-8").split("\0")
    paths = sorted({p for p in changed+added if p})
    manifest = "\n".join(str(Path(p).relative_to("overleaf")).replace("\\","/") for p in paths)+"\n"
    (ROOT / "04_outputs/reviewer_followup/OVERLEAF_FILES.txt").write_text(manifest,encoding="utf-8")
    if args.package_update:
        args.package_update.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.package_update,"w",zipfile.ZIP_DEFLATED) as archive:
            for p in paths:
                archive.write(ROOT/p,Path(p).relative_to("overleaf").as_posix())
            archive.writestr("README_REVIEW_UPDATE.txt",
                "UPDATE PACKAGE FOR THE EXISTING OVERLEAF THESIS PROJECT\n\n"
                "This archive contains changed/new files only, with paths relative to the Overleaf project root.\n"
                "It is not a standalone new Overleaf project and deliberately does not include the old template main.pdf.\n"
                "Original study cohorts, selected models and thresholds remain unchanged.\n"
                "Compilation/pagination have not been verified locally.\n\n"+manifest)
        print(f"Overleaf update archive: {args.package_update} ({len(paths)} files)")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
