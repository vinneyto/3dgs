"""Reusable Colab bootstrap helpers with Google Drive artifact caching.

This module implements a lightweight "variant A" caching approach:
- each expensive build step is a Python function
- each step is wrapped by a decorator that stores/restores artifacts from Drive

Typical usage from a Colab notebook:
    cache = BuildCache("/content/drive/MyDrive/3dgs_cache")
    bootstrap_colmap_cpu(cache=cache, force_rebuild_steps=set())
    bootstrap_3dgs_extensions(cache=cache, repo_dir="...", wheelhouse_dir="...")
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


def _run(command: str, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> None:
    """Run a shell command with bash and fail on errors."""
    print(f"[run] {command}")
    subprocess.run(command, shell=True, executable="/bin/bash", check=True, cwd=cwd, env=env)


def _safe_cmd(command: str) -> str:
    """Best-effort command output used in environment fingerprinting."""
    try:
        out = subprocess.check_output(
            command, shell=True, executable="/bin/bash", stderr=subprocess.STDOUT, text=True
        )
        return out.strip()[:512]
    except Exception:
        return "n/a"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_dir(path: Any) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        _ensure_dir(dst.parent)
        shutil.copy2(src, dst)


class BuildCache:
    """Artifact cache rooted on Google Drive or local filesystem."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.artifacts_dir = self.root_dir / "artifacts"
        self.meta_dir = self.root_dir / "meta"
        _ensure_dir(self.artifacts_dir)
        _ensure_dir(self.meta_dir)
        self.env_fingerprint = self._collect_env_fingerprint()

    def _collect_env_fingerprint(self) -> Dict[str, str]:
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "nvidia_smi_driver": _safe_cmd(
                "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1"
            ),
            "nvcc": _safe_cmd("nvcc --version | tail -n 1"),
            "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG", "n/a"),
        }

    def make_step_key(
        self,
        *,
        step_name: str,
        step_version: str,
        source_hash: str,
        context: Dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "step_name": step_name,
            "step_version": step_version,
            "source_hash": source_hash,
            "env": self.env_fingerprint,
            "context": context or {},
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def _artifact_path(self, step_name: str, key: str) -> Path:
        safe_name = step_name.replace("/", "_")
        return self.artifacts_dir / f"{safe_name}-{key}.tar.gz"

    def has(self, step_name: str, key: str) -> bool:
        return self._artifact_path(step_name, key).exists()

    def save(self, *, step_name: str, key: str, outputs: Sequence[str]) -> Path:
        artifact_path = self._artifact_path(step_name, key)
        tmp_fd, tmp_archive_path = tempfile.mkstemp(suffix=".tar.gz", prefix=f"{step_name}-")
        os.close(tmp_fd)
        tmp_archive = Path(tmp_archive_path)

        with tempfile.TemporaryDirectory(prefix=f"{step_name}-pack-") as tdir:
            tpath = Path(tdir)
            payload_dir = tpath / "payload"
            payload_dir.mkdir(parents=True, exist_ok=True)
            manifest: List[Dict[str, Any]] = []

            for idx, output in enumerate(outputs):
                src = Path(output)
                if not src.exists():
                    raise FileNotFoundError(f"Output path missing for cache save: {src}")
                staged_name = f"{idx:02d}_{src.name}"
                staged_path = payload_dir / staged_name
                _copy_path(src, staged_path)
                manifest.append(
                    {
                        "target_path": str(src),
                        "staged_name": staged_name,
                        "is_dir": src.is_dir(),
                    }
                )

            manifest_path = tpath / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with tarfile.open(tmp_archive, "w:gz") as tar:
                tar.add(payload_dir, arcname="payload")
                tar.add(manifest_path, arcname="manifest.json")

        _ensure_dir(artifact_path.parent)
        shutil.move(str(tmp_archive), artifact_path)

        meta = {
            "step_name": step_name,
            "key": key,
            "artifact_path": str(artifact_path),
            "saved_at_epoch": int(time.time()),
            "outputs": list(outputs),
        }
        (self.meta_dir / f"{step_name}-{key}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return artifact_path

    def restore(self, *, step_name: str, key: str) -> None:
        artifact_path = self._artifact_path(step_name, key)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Cache artifact not found: {artifact_path}")

        with tempfile.TemporaryDirectory(prefix=f"{step_name}-restore-") as tdir:
            tpath = Path(tdir)
            with tarfile.open(artifact_path, "r:gz") as tar:
                tar.extractall(path=tpath)

            manifest = json.loads((tpath / "manifest.json").read_text(encoding="utf-8"))
            for item in manifest:
                target = Path(item["target_path"])
                staged = tpath / "payload" / item["staged_name"]
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                _ensure_dir(target.parent)
                _copy_path(staged, target)


def cached_build_step(
    *,
    step_name: str,
    step_version: str,
    outputs: Any,
    key_context: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Dict[str, Any]]]:
    """Decorator that caches/restores build artifacts for a step."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Dict[str, Any]]:
        source_hash = _sha256_text(inspect.getsource(func))

        def wrapper(*args: Any, cache: BuildCache, force_rebuild: bool = False, **kwargs: Any) -> Dict[str, Any]:
            ctx = key_context(*args, **kwargs) if key_context else {}
            key = cache.make_step_key(
                step_name=step_name,
                step_version=step_version,
                source_hash=source_hash,
                context=ctx,
            )

            resolved_outputs = list(outputs(*args, **kwargs)) if callable(outputs) else list(outputs)
            if not force_rebuild and cache.has(step_name, key):
                print(f"[cache hit] step={step_name} key={key}")
                try:
                    cache.restore(step_name=step_name, key=key)
                    return {"step": step_name, "key": key, "cache_hit": True, "outputs": resolved_outputs}
                except Exception as exc:
                    print(f"[cache restore failed] step={step_name} key={key} error={exc}. Rebuilding...")

            print(f"[cache miss] step={step_name} key={key}")
            result = func(*args, **kwargs)
            artifact = cache.save(step_name=step_name, key=key, outputs=resolved_outputs)
            print(f"[cache saved] step={step_name} artifact={artifact}")
            return {
                "step": step_name,
                "key": key,
                "cache_hit": False,
                "outputs": resolved_outputs,
                "artifact": str(artifact),
                "result": result,
            }

        return wrapper

    return decorator


def install_colmap_system_dependencies() -> None:
    """Install COLMAP build dependencies (non-cached apt packages)."""
    _run("apt-get update")
    _run(
        "apt-get install -y "
        "build-essential cmake git "
        "libboost-all-dev libeigen3-dev libsuitesparse-dev "
        "qtbase5-dev libglew-dev libglfw3-dev "
        "libx11-dev libopencv-dev libgoogle-glog-dev "
        "libgflags-dev libatlas-base-dev libopencv-core-dev "
        "libopenimageio-dev openimageio-tools libopenexr-dev "
        "libcgal-dev libcgal-qt5-dev libmetis-dev ccache"
    )


def _git_commit(repo_dir: str) -> str:
    """Return current commit hash for a git repo or 'n/a'."""
    return _safe_cmd(f"git -C {shlex.quote(repo_dir)} rev-parse HEAD")


def _3dgs_key_context(repo_dir: str, wheelhouse_dir: str, **_: Any) -> Dict[str, Any]:
    dgr = str(Path(repo_dir) / "submodules" / "diff-gaussian-rasterization")
    sknn = str(Path(repo_dir) / "submodules" / "simple-knn")
    return {
        "repo_dir": repo_dir,
        "wheelhouse_dir": wheelhouse_dir,
        "repo_commit": _git_commit(repo_dir),
        "dgr_commit": _git_commit(dgr),
        "sknn_commit": _git_commit(sknn),
    }


@cached_build_step(
    step_name="build_abseil",
    step_version="1",
    outputs=lambda prefix, **_: [prefix],
    key_context=lambda prefix, git_ref="20230802.0", **_: {"prefix": prefix, "git_ref": git_ref},
)
def build_abseil(prefix: str, git_ref: str = "20230802.0", src_root: str = "/content/src") -> None:
    _ensure_dir(src_root)
    repo_dir = Path(src_root) / "abseil-cpp"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    _run(f"git clone https://github.com/abseil/abseil-cpp.git {shlex.quote(str(repo_dir))}")
    _run(f"git checkout {shlex.quote(git_ref)}", cwd=str(repo_dir))
    build_dir = repo_dir / "build"
    _run("rm -rf build && mkdir build", cwd=str(repo_dir))
    _run(
        "cmake .. "
        "-DCMAKE_BUILD_TYPE=Release "
        f"-DCMAKE_INSTALL_PREFIX={shlex.quote(prefix)} "
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        cwd=str(build_dir),
    )
    _run("make -j$(nproc)", cwd=str(build_dir))
    _run("make install", cwd=str(build_dir))


@cached_build_step(
    step_name="build_ceres",
    step_version="1",
    outputs=lambda prefix, **_: [prefix],
    key_context=lambda prefix, abseil_prefix, git_ref="2.1.0", **_: {
        "prefix": prefix,
        "abseil_prefix": abseil_prefix,
        "git_ref": git_ref,
    },
)
def build_ceres(
    prefix: str,
    abseil_prefix: str,
    git_ref: str = "2.1.0",
    src_root: str = "/content/src",
) -> None:
    _ensure_dir(src_root)
    repo_dir = Path(src_root) / "ceres-solver"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    _run(f"git clone https://github.com/ceres-solver/ceres-solver.git {shlex.quote(str(repo_dir))}")
    _run(f"git checkout {shlex.quote(git_ref)}", cwd=str(repo_dir))
    build_dir = repo_dir / "build"
    _run("rm -rf build && mkdir build", cwd=str(repo_dir))
    _run(
        "cmake .. "
        "-DBUILD_TESTING=OFF "
        "-DBUILD_EXAMPLES=OFF "
        "-DCMAKE_BUILD_TYPE=Release "
        f"-DCMAKE_INSTALL_PREFIX={shlex.quote(prefix)} "
        f"-Dabsl_DIR={shlex.quote(str(Path(abseil_prefix) / 'lib/cmake/absl'))}",
        cwd=str(build_dir),
    )
    _run("make -j$(nproc)", cwd=str(build_dir))
    _run("make install", cwd=str(build_dir))


@cached_build_step(
    step_name="build_colmap_cpu",
    step_version="1",
    outputs=lambda prefix, **_: [prefix],
    key_context=lambda prefix, ceres_prefix, abseil_prefix, git_ref="main", **_: {
        "prefix": prefix,
        "ceres_prefix": ceres_prefix,
        "abseil_prefix": abseil_prefix,
        "git_ref": git_ref,
    },
)
def build_colmap_cpu(
    prefix: str,
    ceres_prefix: str,
    abseil_prefix: str,
    git_ref: str = "main",
    src_root: str = "/content/src",
) -> None:
    _ensure_dir(src_root)
    repo_dir = Path(src_root) / "colmap"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    _run(f"git clone https://github.com/colmap/colmap.git {shlex.quote(str(repo_dir))}")
    if git_ref != "main":
        _run(f"git checkout {shlex.quote(git_ref)}", cwd=str(repo_dir))
    build_dir = repo_dir / "build"
    _run("rm -rf build && mkdir build", cwd=str(repo_dir))
    _run(
        "cmake .. "
        "-DCMAKE_BUILD_TYPE=Release "
        f"-DCMAKE_INSTALL_PREFIX={shlex.quote(prefix)} "
        "-DCUDA_ENABLED=OFF "
        f"-DCeres_DIR={shlex.quote(str(Path(ceres_prefix) / 'lib/cmake/Ceres'))} "
        f"-DAbsl_DIR={shlex.quote(str(Path(abseil_prefix) / 'lib/cmake/absl'))}",
        cwd=str(build_dir),
    )
    _run("make -j$(nproc)", cwd=str(build_dir))
    _run("make install", cwd=str(build_dir))


@cached_build_step(
    step_name="build_3dgs_extension_wheels",
    step_version="1",
    outputs=lambda wheelhouse_dir, **_: [wheelhouse_dir],
    key_context=_3dgs_key_context,
)
def build_3dgs_extension_wheels(repo_dir: str, wheelhouse_dir: str) -> None:
    _ensure_dir(wheelhouse_dir)
    for wheel in Path(wheelhouse_dir).glob("*.whl"):
        wheel.unlink()
    _run(
        f"python -m pip wheel --no-deps -w {shlex.quote(wheelhouse_dir)} "
        f"{shlex.quote(str(Path(repo_dir) / 'submodules/diff-gaussian-rasterization'))}"
    )
    _run(
        f"python -m pip wheel --no-deps -w {shlex.quote(wheelhouse_dir)} "
        f"{shlex.quote(str(Path(repo_dir) / 'submodules/simple-knn'))}"
    )


def install_3dgs_extensions_from_wheelhouse(wheelhouse_dir: str) -> None:
    wheels = sorted(Path(wheelhouse_dir).glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No wheel files found in {wheelhouse_dir}")
    quoted = " ".join(shlex.quote(str(p)) for p in wheels)
    _run(f"python -m pip install -q {quoted}")


def bootstrap_colmap_cpu(
    *,
    cache: BuildCache,
    force_rebuild_steps: Iterable[str] | None = None,
    abseil_prefix: str = "/content/opt/abseil",
    ceres_prefix: str = "/content/opt/ceres",
    colmap_prefix: str = "/content/opt/colmap",
    abseil_ref: str = "20230802.0",
    ceres_ref: str = "2.1.0",
    colmap_ref: str = "main",
) -> Dict[str, Any]:
    """Install deps + build/restore cached COLMAP toolchain."""
    force = set(force_rebuild_steps or [])
    install_colmap_system_dependencies()
    step_results = []
    step_results.append(
        build_abseil(
            prefix=abseil_prefix,
            git_ref=abseil_ref,
            cache=cache,
            force_rebuild="build_abseil" in force,
        )
    )
    step_results.append(
        build_ceres(
            prefix=ceres_prefix,
            abseil_prefix=abseil_prefix,
            git_ref=ceres_ref,
            cache=cache,
            force_rebuild="build_ceres" in force,
        )
    )
    step_results.append(
        build_colmap_cpu(
            prefix=colmap_prefix,
            ceres_prefix=ceres_prefix,
            abseil_prefix=abseil_prefix,
            git_ref=colmap_ref,
            cache=cache,
            force_rebuild="build_colmap_cpu" in force,
        )
    )
    return {
        "abseil_prefix": abseil_prefix,
        "ceres_prefix": ceres_prefix,
        "colmap_prefix": colmap_prefix,
        "step_results": step_results,
    }


def bootstrap_3dgs_extensions(
    *,
    cache: BuildCache,
    repo_dir: str,
    wheelhouse_dir: str = "/content/drive/MyDrive/3dgs_cache/wheelhouse",
    force_rebuild_steps: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """Build/restore cached extension wheels and install them."""
    force = set(force_rebuild_steps or [])
    result = build_3dgs_extension_wheels(
        repo_dir=repo_dir,
        wheelhouse_dir=wheelhouse_dir,
        cache=cache,
        force_rebuild="build_3dgs_extension_wheels" in force,
    )
    install_3dgs_extensions_from_wheelhouse(wheelhouse_dir)
    return {"wheelhouse_dir": wheelhouse_dir, "step_result": result}

