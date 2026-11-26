"""Lanza estudio.py evitando la suspensión del equipo mientras dure el proceso."""
import ctypes, runpy, sys
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED
sys.argv[0] = "estudio.py"
runpy.run_path("estudio.py", run_name="__main__")
