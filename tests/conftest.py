"""Reference imports must never write bytecode into the frozen archive."""
import sys

sys.dont_write_bytecode = True
