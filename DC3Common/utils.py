import os
import sys
import configparser
import time
import select

# ----------------------------------------------------------------------------- #
# ---------------------- PATH RESOLUTION & CONFIGURATION ---------------------- #
# ----------------------------------------------------------------------------- #

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

def parse_config_argv(argv):
    """
    Parses command line arguments and returns the configuration file path.
    Exits with an error message if the arguments are invalid.
    """
    if len(argv) != 2:
        print(f"Usage: python3 {os.path.basename(argv[0])} <config.ini>")
        sys.exit(1)

    config_file = os.path.abspath(argv[1])
    if not os.path.isfile(config_file):
        print(f"ERROR: Configuration file '{config_file}' not found.")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_file)
    return config_file, config

# ----------------------------------------------------------------------------- #
# ---------------------------- IO HELPER FUNCTIONS ---------------------------- #
# ----------------------------------------------------------------------------- #

def pause_for_review(timeout=5.0):
    """
    Pauses for `timeout` seconds to let the user review configuration.
    If a key is pressed during the countdown, execution pauses until another
    key is pressed, then resumes.
    """
    print(f"Proceeding in {int(timeout)} seconds. Press any key to pause and review...")
    if not sys.stdin.isatty():
        time.sleep(timeout)
        return

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            _ = sys.stdin.read(1)
            print("\n[PAUSED] Configuration review paused. Press any key to resume...")
            _ = sys.stdin.read(1)
            print("Resuming execution...\n")
        else:
            print("Continuing...\n")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# ----------------------------------------------------------------------------- #
# ------------------------ RESOLVE ZHAIRES UTILS PATHS ------------------------ #
# ----------------------------------------------------------------------------- #

def get_zhairestools_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    candidate_dirs = [
        os.environ.get("ZHAIRESPYTHON", ""),
        os.path.join(repo_root, "ZHAireSPython"),
        os.path.join(os.getcwd(), "ZHAireSPython"),
        os.getcwd(),
        script_dir,
    ]

    zhaires_path = None
    for cdir in candidate_dirs:
        if cdir and os.path.isfile(os.path.join(cdir, "AiresInfoFunctions.py")) and os.path.isfile(os.path.join(cdir, "AiresInpFunctions.py")):
            zhaires_path = cdir
            break

    if zhaires_path is None:
        print("Error: Could not locate AiresInfoFunctions.py and AiresInpFunctions.py.")
        print("Please set the ZHAIRESPYTHON environment variable to the directory containing them.")
        sys.exit(1)

    sys.path.insert(0, os.path.abspath(zhaires_path))

    try:
        import AiresInfoFunctions as AiresInfo
        import AiresInpFunctions as AiresInp
    except ImportError as e:
        print(f"Error: Could not import AiresInfoFunctions or AiresInpFunctions from '{zhaires_path}': {e}")
        sys.exit(1)

    return AiresInfo, AiresInp