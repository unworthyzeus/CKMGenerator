"""Runtime checks for PyTorch, DirectML, and CUDA."""
from __future__ import annotations

import ctypes
import json
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import winreg
except Exception:  # pragma: no cover - non-Windows fallback
    winreg = None


class RuntimeDependencyError(RuntimeError):
    """Raised when the generator runtime cannot run the requested operation."""


@dataclass
class RuntimeReport:
    torch_installed: bool
    torch_version: Optional[str] = None
    directml_installed: bool = False
    directml_available: bool = False
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_devices: List[Dict[str, object]] = field(default_factory=list)
    system_ram_total_bytes: Optional[int] = None
    system_ram_available_bytes: Optional[int] = None
    windows_video_adapters: List[Dict[str, object]] = field(default_factory=list)
    selected_device: Optional[str] = None
    recommended_batch_size: int = 1
    recommendation_reason: str = ""
    errors: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "torch_installed": self.torch_installed,
            "torch_version": self.torch_version,
            "directml_installed": self.directml_installed,
            "directml_available": self.directml_available,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_devices": self.cuda_devices,
            "system_ram_total_bytes": self.system_ram_total_bytes,
            "system_ram_available_bytes": self.system_ram_available_bytes,
            "windows_video_adapters": self.windows_video_adapters,
            "selected_device": self.selected_device,
            "recommended_batch_size": self.recommended_batch_size,
            "recommendation_reason": self.recommendation_reason,
            "errors": self.errors,
        }


def inspect_runtime(preferred_device: str = "auto") -> RuntimeReport:
    try:
        import torch
    except Exception as exc:
        return RuntimeReport(torch_installed=False, errors={"torch": f"{type(exc).__name__}: {exc}"})

    report = RuntimeReport(torch_installed=True, torch_version=getattr(torch, "__version__", "unknown"))
    total_ram, available_ram = _windows_memory_status()
    report.system_ram_total_bytes = total_ram
    report.system_ram_available_bytes = available_ram
    report.windows_video_adapters = _windows_video_adapters()

    try:
        import torch_directml

        report.directml_installed = True
        try:
            torch_directml.device()
            report.directml_available = True
        except Exception as exc:
            report.errors["directml"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        report.errors["directml"] = f"{type(exc).__name__}: {exc}"

    try:
        report.cuda_available = bool(torch.cuda.is_available())
        report.cuda_device_count = int(torch.cuda.device_count()) if report.cuda_available else 0
        if report.cuda_available:
            for idx in range(report.cuda_device_count):
                props = torch.cuda.get_device_properties(idx)
                report.cuda_devices.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                        "major": int(props.major),
                        "minor": int(props.minor),
                    }
                )
    except Exception as exc:
        report.errors["cuda"] = f"{type(exc).__name__}: {exc}"

    report.selected_device = select_device_name(report, preferred_device)
    report.recommended_batch_size, report.recommendation_reason = recommend_batch_size(report)
    return report


def select_device_name(report: RuntimeReport, preferred_device: str = "auto") -> str:
    requested = str(preferred_device).lower()
    if requested == "auto":
        # Project priority: CUDA, then DirectML, then CPU.
        if report.cuda_available:
            return "cuda"
        if report.directml_available:
            return "directml"
        return "cpu"
    return requested


def recommend_batch_size(report: RuntimeReport) -> tuple[int, str]:
    selected = report.selected_device or "cpu"
    gb = 1024 ** 3
    if selected == "cuda":
        total = max((int(d.get("total_memory_bytes", 0)) for d in report.cuda_devices), default=0)
        if total >= 24 * gb:
            return 3, "CUDA VRAM >= 24 GB; batch 3 should be reasonable for bulk inference."
        if total >= 12 * gb:
            return 2, "CUDA VRAM >= 12 GB; batch 2 should be reasonable for bulk inference."
        return 1, "CUDA VRAM is limited or unknown; use batch 1."
    if selected == "directml":
        total = max(
            (
                max(
                    int(a.get("adapter_ram_bytes", 0) or 0),
                    int(a.get("registry_adapter_ram_bytes", 0) or 0),
                )
                for a in report.windows_video_adapters
            ),
            default=0,
        )
        if total >= 20 * gb:
            return 2, "DirectML VRAM appears >= 20 GB, but DirectML overhead is high; batch 2 max."
        return 1, "DirectML uses extra VRAM on this model; batch 1 is the safe default."
    available = int(report.system_ram_available_bytes or 0)
    if selected == "cpu" and available >= 32 * gb:
        return 2, "CPU fallback with >= 32 GB available RAM; batch 2 may work, batch 1 is safer."
    return 1, "Batch 1 is recommended."


def ensure_torch_runtime(preferred_device: str = "auto") -> RuntimeReport:
    report = inspect_runtime(preferred_device)
    if not report.torch_installed:
        raise RuntimeDependencyError(
            "PyTorch is not installed in this Python environment. "
            "Install the project requirements or run with the C:/TFG/.venv Python."
        )

    selected = report.selected_device
    if selected == "directml" and not report.directml_available:
        raise RuntimeDependencyError(
            "DirectML was requested, but torch-directml is not available. "
            f"Details: {report.errors.get('directml', 'unknown error')}"
        )
    if selected == "cuda" and not report.cuda_available:
        raise RuntimeDependencyError(
            "CUDA was requested, but this PyTorch build cannot see a CUDA device. "
            f"Details: {report.errors.get('cuda', 'torch.cuda.is_available() is false')}"
        )
    if selected not in {"auto", "directml", "cuda", "cpu"}:
        raise RuntimeDependencyError(f"Unsupported device '{selected}'. Use auto, directml, cuda, or cpu.")
    return report


def build_torch_device(device_name: str):
    import torch

    selected = str(device_name).lower()
    if selected == "directml":
        import torch_directml

        return torch_directml.device()
    if selected == "cuda":
        return torch.device("cuda")
    if selected == "cpu":
        return torch.device("cpu")
    raise RuntimeDependencyError(f"Unsupported resolved device '{device_name}'.")


def _windows_memory_status() -> tuple[Optional[int], Optional[int]]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        if not ok:
            return None, None
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except Exception:
        return None, None


def _windows_video_adapters() -> List[Dict[str, object]]:
    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID | "
        "ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        payload = json.loads(proc.stdout)
        if isinstance(payload, dict):
            payload = [payload]
        out = []
        for item in payload:
            ram = item.get("AdapterRAM")
            out.append(
                {
                    "name": item.get("Name"),
                    "adapter_ram_bytes": int(ram) if ram is not None else None,
                    "adapter_ram_note": "Win32_VideoController.AdapterRAM can be capped on some AMD/Windows drivers.",
                    "driver_version": item.get("DriverVersion"),
                    "pnp_device_id": item.get("PNPDeviceID"),
                }
            )
        registry = _windows_registry_video_memory()
        for reg in registry:
            matched = False
            reg_name = str(reg.get("name") or "").lower()
            for item in out:
                item_name = str(item.get("name") or "").lower()
                if reg_name and item_name and (reg_name in item_name or item_name in reg_name):
                    current = int(item.get("registry_adapter_ram_bytes", 0) or 0)
                    item["registry_adapter_ram_bytes"] = max(current, int(reg.get("registry_adapter_ram_bytes", 0) or 0))
                    item["registry_source"] = reg.get("registry_source")
                    matched = True
            if not matched:
                out.append(reg)
        return out
    except Exception:
        return _windows_registry_video_memory()


def _windows_registry_video_memory() -> List[Dict[str, object]]:
    if winreg is None:
        return []
    base = r"SYSTEM\CurrentControlSet\Control\Video"
    out: List[Dict[str, object]] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            guid_count = winreg.QueryInfoKey(root)[0]
            for gi in range(guid_count):
                guid = winreg.EnumKey(root, gi)
                guid_path = base + "\\" + guid
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, guid_path) as guid_key:
                        sub_count = winreg.QueryInfoKey(guid_key)[0]
                        for si in range(sub_count):
                            sub = winreg.EnumKey(guid_key, si)
                            sub_path = guid_path + "\\" + sub
                            try:
                                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, sub_path) as adapter_key:
                                    name = _query_registry_value(adapter_key, "HardwareInformation.AdapterString")
                                    mem = _query_registry_value(adapter_key, "HardwareInformation.qwMemorySize")
                                    if name or mem:
                                        out.append(
                                            {
                                                "name": _decode_registry_string(name),
                                                "registry_adapter_ram_bytes": int(mem) if mem else None,
                                                "registry_source": "HKLM/SYSTEM/CurrentControlSet/Control/Video/HardwareInformation.qwMemorySize",
                                            }
                                        )
                            except OSError:
                                continue
                except OSError:
                    continue
    except OSError:
        return []

    dedup: Dict[tuple[str, Optional[int]], Dict[str, object]] = {}
    for item in out:
        key = (str(item.get("name")), item.get("registry_adapter_ram_bytes"))
        dedup[key] = item
    return list(dedup.values())


def _query_registry_value(key, name: str):
    try:
        return winreg.QueryValueEx(key, name)[0]
    except OSError:
        return None


def _decode_registry_string(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-16-le", errors="ignore").rstrip("\x00")
        except Exception:
            return value.decode(errors="ignore").rstrip("\x00")
    return str(value).rstrip("\x00")
