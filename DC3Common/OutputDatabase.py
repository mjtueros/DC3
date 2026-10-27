"""
################################################################################
# What to do with an output database that already exists (issue #51)
################################################################################
# Shared by the scripts that write a database: SimpleFluxGenerator.py,
# RebuildAiresFluxDatabase.py, SimpleXmaxGetter.py, SimpleDC3EventGenerator.py
# and CombineHexRings.py.
#
# A script goes ahead without asking only when its output database does not
# exist. When it exists, ask_about_existing_output() asks whether to overwrite
# it, append to it or exit without touching it. Appending is offered only when
# it gives the same result as one run: the database must have the tables the
# script writes and have been written with the same settings, which each
# database records in its RunSettings table (settings_problem()).
#
# The scripts import this file with:
#   sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DC3Common"))
#   import OutputDatabase
################################################################################
"""

import os
import sys
import sqlite3
from pathlib import Path

# The table where a database records the settings it was written with
RUN_SETTINGS_TABLE_SQL = "CREATE TABLE IF NOT EXISTS RunSettings (Name TEXT PRIMARY KEY, Value TEXT)"


def ask_about_existing_output(db_path, cannot_append):
    """
    Asks what to do with an output database that already exists: overwrite it, append to it, or exit
    without touching it. cannot_append is None when appending is possible, or the reason it is not, and
    then only overwrite and exit are offered. The answer is read from the standard input, so it can also
    be piped in; with nothing to read (e.g. a batch job without input) the script exits.
    Returns "overwrite" or "append".
    """
    if not os.path.isfile(db_path):
        print(f"ERROR: The output database '{db_path}' exists but is not a file.")
        sys.exit(1)
    print(f"The output database already exists: '{db_path}'.")
    if cannot_append:
        print(f"It cannot be appended to: {cannot_append}.")
        question, answers, valid = "Overwrite it (o) or exit (e)? ", {"o": "overwrite", "e": "exit"}, "o or e"
    else:
        question, answers, valid = ("Overwrite it (o), append to it (a) or exit (e)? ",
                                    {"o": "overwrite", "a": "append", "e": "exit"}, "o, a or e")
    while True:
        print(question, end="", flush=True)
        line = sys.stdin.readline()
        if not line:
            print("\nNo answer to read: exiting without touching the database.")
            sys.exit(1)
        answer = answers.get(line.strip().lower()[:1])
        if answer == "exit":
            print("Exiting without touching the database.")
            sys.exit(1)
        if answer:
            return answer
        print(f"Please answer {valid}.")


def open_read_only(db_path):
    """Opens an existing database without any possibility of changing it."""
    return sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)


def settings_problem(db_path, tables_sql, run_settings, script, defaults=None):
    """
    Why the existing database db_path cannot be appended to, or None if it can as far as its tables and
    settings go: it must have the tables that tables_sql creates (the tables the script writes) and
    RunSettings, each with the same columns, and its RunSettings must hold the values of run_settings.
    script names the script in the messages, e.g. "XmaxGetter". defaults holds the settings that were added
    after the first databases were written, each with the value those databases were written with: a
    database, or a run, without such a setting is taken to have that value.
    """
    expected = sqlite3.connect(":memory:")
    try:
        expected.executescript(tables_sql)
        expected.execute(RUN_SETTINGS_TABLE_SQL)
        tables = [name for (name,) in expected.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        try:
            conn = open_read_only(db_path)
            try:
                for table in tables:
                    have = [col[1] for col in conn.execute(f"PRAGMA table_info({table})")]
                    want = [col[1] for col in expected.execute(f"PRAGMA table_info({table})")]
                    if not have:
                        return ("it does not record the settings it was written with (it was written before they were recorded)"
                                if table == "RunSettings" else f"it has no {table} table")
                    if have != want:
                        return f"its {table} table does not have the columns this {script} writes"
                stored = dict(conn.execute("SELECT Name, Value FROM RunSettings"))
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            return "it is not an SQLite database"
    finally:
        expected.close()
    defaults = defaults or {}
    current = {**defaults, **run_settings}
    names = list(run_settings) + [name for name in defaults if name not in run_settings]
    changed = [f"{name} {stored.get(name, defaults.get(name, '(not recorded)'))} instead of {current[name]}"
               for name in names if stored.get(name, defaults.get(name)) != current[name]]
    if changed:
        return "it was written with other settings (" + "; ".join(changed) + ")"
    return None


def write_run_settings(conn, run_settings):
    """Records run_settings in the RunSettings table of the database being written (conn)."""
    conn.execute(RUN_SETTINGS_TABLE_SQL)
    conn.executemany("INSERT OR REPLACE INTO RunSettings (Name, Value) VALUES (?, ?)", run_settings.items())


def path_setting(path, db_path):
    """
    A path to record in RunSettings: relative to the output database, so that a moved or cloned project
    can still append to it.
    """
    return os.path.relpath(path, os.path.dirname(os.path.abspath(db_path)))


def delete_old_output(db_path):
    """Deletes the database you chose to overwrite; the scripts call it just before writing the new one."""
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted the old '{db_path}'.")
