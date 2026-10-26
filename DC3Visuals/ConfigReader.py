"""Config reader in a separate file."""

from dataclasses import dataclass
import configparser
import os
import sys


@dataclass
class VisualConfig:
    showers_db: str
    events_db: str
    antennas_csv: str
    center_n: float
    center_w: float
    center_u: float
    showers_fig: bool
    events_fig: bool
    save_fig: bool
    show_fig: bool
    fig_dir: str


def resolve_ini_path(config_file_path, path_str):
    """
    Resolves a path from the INI file. If relative, it is strictly resolved
    relative to the directory containing the INI file.
    """
    if not path_str or not path_str.strip():
        return ""
    path_str = path_str.strip()
    if os.path.isabs(path_str):
        return path_str
    config_dir = os.path.dirname(os.path.abspath(config_file_path))
    return os.path.abspath(os.path.join(config_dir, path_str))


def load_config(config_file: str) -> VisualConfig:
    config = configparser.ConfigParser()
    config.read(config_file)

    if "Visuals" not in config:
        raise ValueError(f"Section [Visuals] not found in '{config_file}'.")

    sec = config["Visuals"]

    return VisualConfig(
        showers_db=resolve_ini_path(config_file, sec.get("showers_db", fallback="ExampleFluxWithDummyXmax.sqlite")),
        events_db=resolve_ini_path(config_file, sec.get("events_db", fallback="ExampleDummyTriggeredEvents.sqlite")),
        antennas_csv=resolve_ini_path(config_file, sec.get("antennas_csv", fallback="4-DC3EventGenerator/GRAND_GP65_RTK_positions.csv")),
        center_n=sec.getfloat("center_n", fallback=0.0),
        center_w=sec.getfloat("center_w", fallback=0.0),
        center_u=sec.getfloat("center_u", fallback=1250.0),
        showers_fig=sec.getboolean("showers_fig", fallback=False),
        events_fig=sec.getboolean("events_fig", fallback=False),
        save_fig=sec.getboolean("save_fig", fallback=False),
        show_fig=sec.getboolean("show_fig", fallback=False),
        fig_dir=resolve_ini_path(config_file, sec.get("fig_dir", fallback="./DC3Visuals/figures")),
    )
