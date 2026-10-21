import importlib.util
import json

from conftest import ROOT


def test_committed_notebook_matches_builder():
    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "scripts" / "build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = (ROOT / "notebooks" / "Audio8_Persian_Finetune.ipynb").read_text(encoding="utf-8")
    assert committed == module.render(), "notebook is stale: run `python scripts/build_notebook.py`"
    notebook = json.loads(committed)
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    parts = [s for s in sources if s.startswith(("%%writefile fa_tools.py", "%%writefile -a fa_tools.py"))]
    assert parts[0].startswith("%%writefile fa_tools.py\n"), "the first part must create the file"
    assert all(p.startswith("%%writefile -a fa_tools.py\n") for p in parts[1:]), "later parts must append"
    embedded = "".join(p.split("\n", 1)[1] for p in parts)
    assert embedded == (ROOT / "src" / "fa_tools.py").read_text(encoding="utf-8")
    compile(embedded, "fa_tools", "exec")
