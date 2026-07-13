import importlib
import importlib.util
import types
import difflib
import os
import shutil
import subprocess
import re

_PATCH_FLAG = "_policy_switching_compat_patched"


class _RegistryCompat(dict):
    @property
    def env_specs(self):
        return self


def _is_patched(obj):
    return getattr(obj, _PATCH_FLAG, False)


def _mark_patched(obj):
    setattr(obj, _PATCH_FLAG, True)


def _patch_gym_registry():
    try:
        registration = importlib.import_module("gym.envs.registration")
    except Exception:
        return "gym-missing"

    registry = getattr(registration, "registry", None)
    if registry is None:
        return "registry-missing"

    if hasattr(registry, "env_specs"):
        return "already-compatible"

    if not isinstance(registry, dict):
        return "unsupported-registry"

    registration.registry = _RegistryCompat(registry)
    return "patched"


def _patch_numpy_bool8():
    try:
        np = importlib.import_module("numpy")
    except Exception:
        return "numpy-missing"

    if hasattr(np, "bool8"):
        return "already-compatible"

    try:
        np.bool8 = np.bool_
    except Exception:
        return "patch-failed"
    return "patched"


def _patch_time_limit_step():
    try:
        time_limit_module = importlib.import_module("gym.wrappers.time_limit")
    except Exception:
        return "gym-missing"

    time_limit_cls = getattr(time_limit_module, "TimeLimit", None)
    if time_limit_cls is None:
        return "time-limit-missing"

    original_step = getattr(time_limit_cls, "step", None)
    if original_step is None:
        return "step-missing"

    if _is_patched(original_step):
        return "already-patched"

    def compat_step(self, action):
        step_out = self.env.step(action)
        if not isinstance(step_out, tuple):
            return step_out

        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        elif len(step_out) == 4:
            obs, reward, terminated, info = step_out
            truncated = False
        else:
            raise ValueError(f"Unexpected env.step output length: {len(step_out)}")

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    _mark_patched(compat_step)
    time_limit_cls.step = compat_step
    return "patched"


def _patch_d4rl_normalized_box_env():
    try:
        wrappers = importlib.import_module("d4rl.utils.wrappers")
    except Exception:
        return "d4rl-missing"

    normalized_box_env = getattr(wrappers, "NormalizedBoxEnv", None)
    if normalized_box_env is None:
        return "normalized-box-missing"

    original_step = getattr(normalized_box_env, "step", None)
    if original_step is None:
        return "step-missing"

    if _is_patched(original_step):
        return "already-patched"

    def compat_step(self, action):
        wrapped_env = (
            getattr(self, "_wrapped_env", None)
            or getattr(self, "wrapped_env", None)
            or getattr(self, "env", None)
        )

        if wrapped_env is None:
            step_out = original_step(self, action)
            if isinstance(step_out, tuple) and len(step_out) == 4:
                obs, reward, done, info = step_out
                return obs, reward, bool(done), False, info
            return step_out

        scaled_action = action
        try:
            low = wrapped_env.action_space.low
            high = wrapped_env.action_space.high
            try:
                import numpy as np
                scaled_action = low + (action + 1.0) * 0.5 * (high - low)
                scaled_action = np.clip(scaled_action, low, high)
            except Exception:
                pass
        except Exception:
            pass

        step_out = wrapped_env.step(scaled_action)
        if not isinstance(step_out, tuple):
            return step_out

        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
        elif len(step_out) == 4:
            next_obs, reward, done, info = step_out
            terminated = bool(done)
            truncated = False
        else:
            raise ValueError(f"Unexpected env.step output length: {len(step_out)}")

        if getattr(self, "_should_normalize", False):
            normaliser = getattr(self, "_apply_normalize_obs", None)
            if callable(normaliser):
                next_obs = normaliser(next_obs)

        reward_scale = getattr(self, "_reward_scale", 1.0)
        reward = reward * reward_scale
        return next_obs, reward, terminated, truncated, info

    _mark_patched(compat_step)
    normalized_box_env.step = compat_step
    return "patched"


def _attach_ant_data_aliases(env):
    data = getattr(env, "data", None)
    if data is None:
        return

    sim = getattr(env, "sim", None)
    if sim is None or getattr(sim, "data", None) is None:
        env.sim = types.SimpleNamespace(data=data)

    physics = getattr(env, "physics", None)
    if physics is None or getattr(physics, "data", None) is None:
        env.physics = types.SimpleNamespace(data=data)


def _patch_d4rl_ant_env():
    try:
        ant_module = importlib.import_module("d4rl.locomotion.ant")
    except Exception:
        return "d4rl-missing"

    ant_env = getattr(ant_module, "AntEnv", None)
    if ant_env is None:
        return "ant-env-missing"

    original_init = getattr(ant_env, "__init__", None)
    if original_init is None:
        return "init-missing"

    if not _is_patched(original_init):
        def compat_init(self, *args, **kwargs):
            try:
                original_init(self, *args, **kwargs)
            except TypeError as exc:
                if "observation_space" not in str(exc):
                    raise

                mujoco_env_module = importlib.import_module("gym.envs.mujoco.mujoco_env")
                spaces_module = importlib.import_module("gym.spaces")
                original_mujoco_init = mujoco_env_module.MujocoEnv.__init__

                try:
                    import numpy as np
                    low = -np.inf
                    high = np.inf
                    dtype = np.float64
                except Exception:
                    low = float("-inf")
                    high = float("inf")
                    dtype = float

                def compat_mujoco_init(mujoco_self, model_path, frame_skip, *mujoco_args, **mujoco_kwargs):
                    if not mujoco_args and "observation_space" not in mujoco_kwargs:
                        mujoco_kwargs["observation_space"] = spaces_module.Box(
                            low=low,
                            high=high,
                            shape=(29,),
                            dtype=dtype,
                        )

                    return original_mujoco_init(
                        mujoco_self,
                        model_path,
                        frame_skip,
                        *mujoco_args,
                        **mujoco_kwargs,
                    )

                mujoco_env_module.MujocoEnv.__init__ = compat_mujoco_init
                try:
                    original_init(self, *args, **kwargs)
                finally:
                    mujoco_env_module.MujocoEnv.__init__ = original_mujoco_init

            _attach_ant_data_aliases(self)

        _mark_patched(compat_init)
        ant_env.__init__ = compat_init

    original_reset_model = getattr(ant_env, "reset_model", None)
    if callable(original_reset_model) and not _is_patched(original_reset_model):
        def compat_reset_model(self, *args, **kwargs):
            _attach_ant_data_aliases(self)
            try:
                return original_reset_model(self, *args, **kwargs)
            except AttributeError as exc:
                if "randn" not in str(exc):
                    raise

                rng = getattr(self, "np_random", None)
                if rng is None or not hasattr(rng, "standard_normal"):
                    raise

                class _RNGCompat:
                    def __init__(self, base_rng):
                        self._base_rng = base_rng

                    def randn(self, *shape):
                        return self._base_rng.standard_normal(size=shape)

                    def __getattr__(self, name):
                        return getattr(self._base_rng, name)

                setattr(self, "np_random", _RNGCompat(rng))
                try:
                    return original_reset_model(self, *args, **kwargs)
                finally:
                    setattr(self, "np_random", rng)

        _mark_patched(compat_reset_model)
        ant_env.reset_model = compat_reset_model

    return "patched"


def apply_runtime_compat_patches():
    patchers = {
        "numpy_bool8": _patch_numpy_bool8,
        "gym_registry": _patch_gym_registry,
        "gym_time_limit": _patch_time_limit_step,
        "d4rl_normalized_box_env": _patch_d4rl_normalized_box_env,
        "d4rl_ant_env": _patch_d4rl_ant_env,
    }

    results = {}
    for name, patcher in patchers.items():
        try:
            results[name] = patcher()
        except Exception as exc:
            results[name] = f"error:{type(exc).__name__}:{exc}"

    return results


def _get_registered_env_ids():
    gym_registration = importlib.import_module("gym.envs.registration")
    registry = getattr(gym_registration, "registry")
    if isinstance(registry, dict):
        return set(registry.keys())

    env_specs = getattr(registry, "env_specs", None)
    if isinstance(env_specs, dict):
        return set(env_specs.keys())

    return set()


def _configure_macos_mujoco_build_env():
    if os.uname().sysname != "Darwin":
        return

    # mujoco-py picks Homebrew gcc-* when CC is unset; force clang toolchain.
    os.environ["CC"] = "/usr/bin/clang"
    os.environ["CXX"] = "/usr/bin/clang++"

    if "MUJOCO_PY_MUJOCO_PATH" not in os.environ:
        os.environ["MUJOCO_PY_MUJOCO_PATH"] = os.path.expanduser("~/.mujoco/mujoco210")

    if shutil.which("xcrun"):
        try:
            sdkroot = subprocess.check_output(
                ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                text=True,
            ).strip()
            if sdkroot:
                os.environ.setdefault("SDKROOT", sdkroot)
                os.environ.setdefault("CONDA_BUILD_SYSROOT", sdkroot)
                os.environ.setdefault("CFLAGS", f"-isysroot {sdkroot}")
                os.environ.setdefault("CXXFLAGS", f"-isysroot {sdkroot}")
                os.environ.setdefault("CPPFLAGS", f"-isysroot {sdkroot}")
        except Exception:
            pass


def _patch_mujoco_py_builder_for_macos():
    if os.uname().sysname != "Darwin":
        return "not-macos"

    spec = importlib.util.find_spec("mujoco_py")
    if spec is None or not spec.origin:
        return "mujoco-py-missing"

    builder_path = os.path.join(os.path.dirname(spec.origin), "builder.py")
    if not os.path.isfile(builder_path):
        return "builder-missing"

    try:
        with open(builder_path, "r", encoding="utf-8") as f:
            original = f.read()
    except Exception:
        return "read-failed"

    patched = original
    patched = re.sub(
        r"\n\s*'-fopenmp',\s*# needed for OpenMP",
        "",
        patched,
        count=1,
    )
    patched = patched.replace("extra_link_args=['-fopenmp'],", "extra_link_args=[],")

    if patched == original:
        return "already-patched"

    try:
        with open(builder_path, "w", encoding="utf-8") as f:
            f.write(patched)
    except Exception:
        return "write-failed"
    return "patched"


def ensure_d4rl_registered(env_id=None):
    apply_runtime_compat_patches()
    _configure_macos_mujoco_build_env()
    openmp_patch_status = _patch_mujoco_py_builder_for_macos()
    try:
        import d4rl
    except Exception as exc:
        if os.uname().sysname == "Darwin":
            hint = (
                "d4rl import failed while loading mujoco_py on macOS. "
                "Re-run ./setup.sh to repin cython/mujoco-py and patch OpenMP flags. "
                f"mujoco_py builder patch status: {openmp_patch_status}. "
                f"Original error: {type(exc).__name__}: {exc}"
            )
            raise ImportError(hint) from exc
        raise ImportError(
            "d4rl is required to register environments such as halfcheetah-medium-v2."
        ) from exc

    # Some d4rl builds do not register all envs from __init__, so load modules explicitly.
    import_errors = {}
    for module_name in (
        "d4rl.gym_mujoco",
        "d4rl.locomotion",
        "d4rl.hand_manipulation_suite",
        "d4rl.pointmaze",
        "d4rl.kitchen",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            import_errors[module_name] = f"{type(exc).__name__}: {exc}"

    if env_id is not None:
        env_ids = _get_registered_env_ids()
        if env_id not in env_ids:
            matches = difflib.get_close_matches(env_id, sorted(env_ids), n=5, cutoff=0.4)
            if matches:
                msg = f"'{env_id}' is not registered. Close matches: {', '.join(matches)}"
            else:
                sample = ", ".join(sorted([x for x in env_ids if "halfcheetah" in x.lower()])[:5])
                msg = f"'{env_id}' is not registered. HalfCheetah-like ids found: {sample or 'none'}"

            gym_mujoco_error = import_errors.get("d4rl.gym_mujoco")
            if gym_mujoco_error is not None:
                msg += f". d4rl.gym_mujoco import failed: {gym_mujoco_error}"
            raise RuntimeError(msg)

    return d4rl
