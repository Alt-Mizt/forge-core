"""The pure core: how much time an account has left, and where the next session runs.

Nothing in this package reads a clock or a file, calls a network or touches a provider.
The current time and every input are passed in, so each answer can be checked by hand.
import-linter enforces this on every check.
"""
