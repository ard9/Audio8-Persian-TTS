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
    embedded = next(s for s in sources if s.startswith("%%writefile fa_tools.py"))
    compile(embedded.split("\n", 1)[1], "fa_tools", "exec")
