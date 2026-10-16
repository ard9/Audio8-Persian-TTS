"""In-memory stand-in for huggingface_hub.HfApi, backed by a local folder."""

import fnmatch
import shutil
from pathlib import Path


class FakeHfApi:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.squashed = 0

    def _repo(self, repo_id, repo_type):
        return self.root / (repo_type or "model") / repo_id

    def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
        self._repo(repo_id, repo_type).mkdir(parents=True, exist_ok=True)

    def list_repo_files(self, repo_id, repo_type=None):
        base = self._repo(repo_id, repo_type)
        return sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None):
        target = self._repo(repo_id, repo_type) / path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(path_or_fileobj, bytes):
            target.write_bytes(path_or_fileobj)
        else:
            shutil.copyfile(path_or_fileobj, target)

    def upload_folder(self, folder_path, path_in_repo, repo_id, repo_type=None, commit_message=None):
        target = self._repo(repo_id, repo_type) / path_in_repo
        shutil.copytree(folder_path, target, dirs_exist_ok=True)

    def delete_folder(self, path_in_repo, repo_id, repo_type=None, commit_message=None):
        shutil.rmtree(self._repo(repo_id, repo_type) / path_in_repo)

    def super_squash_history(self, repo_id, repo_type=None):
        self.squashed += 1

    def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._repo(repo_id, repo_type) / filename, target)
        return str(target)

    def snapshot_download(self, repo_id, repo_type=None, allow_patterns=None, local_dir=None):
        base = self._repo(repo_id, repo_type)
        for rel in self.list_repo_files(repo_id, repo_type):
            if allow_patterns and not any(fnmatch.fnmatch(rel, pat) for pat in allow_patterns):
                continue
            target = Path(local_dir) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / rel, target)
        return str(local_dir)
