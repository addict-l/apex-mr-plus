"""Lossless task-data TOML serialization, including optional metadata fields."""
from datetime import date, datetime, time
import json
import math
import re
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _key(value):
    return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else json.dumps(value, ensure_ascii=False)


def _value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("任务数据不能包含非有限浮点数")
        return repr(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(_key(str(key))+" = "+_value(item)
                               for key, item in value.items()) + "}"
    raise TypeError("不支持的 TOML 任务值：" + type(value).__name__)


def task_data_to_toml(data):
    """Retain every parsed field; schema-specific writers must not drop data."""
    rendered = "\n".join(_key(str(key))+" = "+_value(value)
                         for key, value in data.items()) + "\n"
    if tomllib.loads(rendered) != data:
        raise ValueError("任务 TOML 数据往返不一致，拒绝写出")
    return rendered


def write_task_data(path, data):
    """Create a project-local copy; never overwrite a different existing task."""
    destination = Path(path).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("任务副本必须位于项目目录内")
    text = task_data_to_toml(data)
    if destination.exists() and destination.read_text(encoding="utf-8-sig") != text:
        raise FileExistsError("保留已有不同任务副本：" + str(destination))
    from tools.overnight_batch import atomic_text
    atomic_text(destination, text)
    return destination
