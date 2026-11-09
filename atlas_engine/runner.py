"""Terminal panel and the runner that executes fixed steps in child processes."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal
from PySide6.QtGui import QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

from . import paths, runtime
from . import theme as T

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LEVELS = {"info": T.TEXT, "warn": T.AMBER, "error": T.ROSE, "cmd": T.ACCENT, "muted": T.MUTED, "ok": T.ACCENT, "raw": T.MUTED}


class Terminal(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setObjectName("terminal")
        self.setReadOnly(True)
        self.setFont(T.font(9.5))
        self.setMaximumBlockCount(6000)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)

    def write(self, text, level="info"):
        fmt = QTextCharFormat()
        fmt.setForeground(T.color(LEVELS.get(level, T.TEXT)))
        c = self.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        c.insertText(text + "\n", fmt)
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def table(self, rows):
        for k, v in rows:
            self.write(f"  {k:<8}{v}", "muted")


class Runner(QObject):
    """Runs a list of runtime.Step one after another and reports worker events."""

    finished = Signal(bool, bool)    # succeeded, cancelled

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = None
        self.term = None
        self.steps = []
        self.buf = ""
        self.cancelled = False
        self.on_event = None
        self.current = None
        self.governor = None

    @property
    def running(self):
        return self.proc is not None

    def start(self, steps, term, on_event=None):
        self.steps, self.term, self.on_event = list(steps), term, on_event
        self.cancelled = False
        self._next()

    def _next(self):
        if not self.steps:
            self.finished.emit(True, False)
            return
        self.current = self.steps.pop(0)
        self.term.write(f"$ {self.current.title}", "cmd")
        self.buf = ""
        p = QProcess(self)
        env = QProcessEnvironment()
        for k, v in {**paths.worker_env(), **self.current.env}.items():
            env.insert(k, str(v))
        p.setProcessEnvironment(env)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._exited)
        p.errorOccurred.connect(self._failed_to_start)
        self.proc = p
        argv = self.current.argv
        if self.current.governed and self.governor is not None:
            argv = self.governor.prepare(argv)
            p.started.connect(lambda: self.governor.attach(p.processId()))
        p.start(str(argv[0]), [str(a) for a in argv[1:]])

    def _failed_to_start(self, err):
        if err == QProcess.ProcessError.FailedToStart and self.proc is not None:
            self.term.write(f"could not start {self.current.argv[0]}", "error")
            self.proc = None
            self.steps = []
            self.finished.emit(False, False)

    def _read(self):
        if self.proc is None:
            return
        self.buf += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        parts = re.split(r"[\r\n]", self.buf)
        self.buf = parts.pop()
        for line in parts:
            self._line(ANSI.sub("", line))

    def _line(self, line):
        if line.startswith("@@"):
            try:
                ev = json.loads(line[2:])
            except ValueError:
                self.term.write(line, "raw")
                return
            if self.on_event:
                self.on_event(ev)
        elif line.strip():
            self.term.write(line, "raw")

    def _exited(self, code, status):
        self._read()
        if self.buf.strip():
            self._line(ANSI.sub("", self.buf))
        self.proc = None
        step = self.current
        if step.governed and self.governor is not None:
            self.governor.detach()
        if self.cancelled:
            self.steps = []
            self.finished.emit(False, True)
        elif code == 0 and status == QProcess.ExitStatus.NormalExit:
            if step.stamp:
                runtime.record_group(*step.stamp)
            self._next()
        else:
            self.steps = []
            self.finished.emit(False, False)

    def cancel(self):
        """Stop the current process and any children it started."""
        if self.proc is None:
            return
        self.cancelled = True
        pid = self.proc.processId()
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)   # the worker leads its own group
                except ProcessLookupError:
                    os.kill(pid, signal.SIGKILL)     # a setup step, which has no group of its own
        except OSError:
            self.proc.kill()
