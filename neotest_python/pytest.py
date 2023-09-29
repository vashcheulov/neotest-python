from io import StringIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from .base import NeotestAdapter, NeotestError, NeotestResult, NeotestResultStatus

import pytest
from _pytest._code.code import (
    ExceptionInfo,
    ExceptionRepr,
    ReprTraceback,
    ReprEntry,
    TerminalRepr,
)
from _pytest.terminal import TerminalReporter


class PytestNeotestAdapter(NeotestAdapter):
    def run(
        self,
        args: List[str],
        stream: Callable[[str, NeotestResult], None],
    ) -> Dict[str, NeotestResult]:
        result_collector = NeotestResultCollector(self, stream=stream)
        pytest.main(args=args, plugins=[
            result_collector,
            NeotestDebugpyPlugin(),
        ])
        return result_collector.results


class NeotestResultCollector:
    def __init__(
        self,
        adapter: PytestNeotestAdapter,
        stream: Callable[[str, NeotestResult], None],
    ):
        self.stream = stream
        self.adapter = adapter

        self.pytest_config: Optional["pytest.Config"] = None  # type: ignore
        self.results: Dict[str, NeotestResult] = {}
        self.nodes: Dict[
            Tuple[Union["pytest.TestReport", str], object], Node
        ] = {}

    def get_node(self, nodeid: str, worker_node: Optional[object] = None) -> "Node":
        key = nodeid, worker_node

        if key in self.nodes:
            return self.nodes[key]

        node = Node(nodeid, self)
        self.nodes[key] = node

        return node

    def pytest_cmdline_main(self, config: "pytest.Config"):
        self.pytest_config = config

    def pytest_runtest_logreport(self, report: "pytest.TestReport") -> None:
        # Local hack to handle xdist report order.
        worker_node = getattr(report, "node", None)
        node = self.get_node(report.nodeid, worker_node)
        if report.passed:
            if report.when == "call":
                node.collect_passed(report)
        elif report.failed:
            node.collect_failed(report)
        elif report.skipped:
            node.collect_skipped(report)

    def pytest_internalerror(self, excrepr: ExceptionRepr) -> None:
        node = self.get_node("internal")
        node.collect_error(excrepr)


class Node:
    def __init__(
        self,
        nodeid: Union[str, "pytest.TestReport"],
        collector: "NeotestResultCollector",
    ) -> None:
        self.id = nodeid
        self.collector = collector
        self.file_path, *name_path = nodeid.split("::")
        self.abs_path = str(Path(collector.pytest_config.rootdir, self.file_path))
        *namespaces, test_name = name_path
        valid_test_name, *params = test_name.split("[")  # ]
        self.pos_id = "::".join([self.abs_path, *namespaces, valid_test_name])
        self.params = params

    def get_short_output(self, report: "pytest.TestReport") -> Optional[str]:
        buffer = StringIO()
        # Hack to get pytest to write ANSI codes
        setattr(buffer, "isatty", lambda: True)
        reporter = TerminalReporter(self.collector.pytest_config, buffer)

        # Taked from `_pytest.terminal.TerminalReporter
        msg = reporter._getfailureheadline(report)
        if report.outcome == NeotestResultStatus.FAILED:
            reporter.write_sep("_", msg, red=True, bold=True)
        elif report.outcome == NeotestResultStatus.SKIPPED:
            reporter.write_sep("_", msg, cyan=True, bold=True)
        else:
            reporter.write_sep("_", msg, green=True, bold=True)
        reporter._outrep_summary(report)
        reporter.print_teardown_sections(report)

        buffer.seek(0)
        return buffer.read()

    def collect(
        self,
        short: Optional[str],
        status: NeotestResultStatus,
        errors: List[NeotestError],
    ) -> None:
        result = self.collector.adapter.update_result(
            self.collector.results.get(self.pos_id),
            {
                "short": short,
                "status": status,
                "errors": errors,
            },
        )
        if not self.params:
            self.collector.stream(self.pos_id, result)
        self.collector.results[self.pos_id] = result

    def collect_passed(self, report: "pytest.TestReport") -> None:
        self.collect(
            short=self.get_short_output(report),
            status=NeotestResultStatus.PASSED,
            errors=[],
        )

    def find_error_line(self, repr_traceback: ReprTraceback) -> Optional[int]:
        error_line: Optional[int] = None
        for repr_entry in reversed(repr_traceback.reprentries):
            if not (isinstance(repr_entry, ReprEntry) and repr_entry.reprfileloc):
                continue
            location = repr_entry.reprfileloc
            if self.file_path == location.path:
                error_line = location.lineno - 1
                break
        return error_line

    def collect_failed(self, report: "pytest.TestReport") -> None:
        errors: List[NeotestError] = []
        exc_repr: Union[
            None, ExceptionInfo[BaseException], Tuple[str, int, str], str, TerminalRepr
        ] = report.longrepr
        # Test fails due to condition outside of test e.g. xfail
        if isinstance(exc_repr, str):
            errors.append({"message": exc_repr, "line": None})
        # Test failed internally
        elif isinstance(exc_repr, ReprTraceback):
            error_line = self.find_error_line(exc_repr)
            errors.append({"message": "error", "line": error_line})
        elif isinstance(exc_repr, ExceptionRepr):
            error_message = exc_repr.reprcrash.message  # type: ignore
            error_line = self.find_error_line(exc_repr.reprtraceback)
            errors.append({"message": error_message, "line": error_line})
        else:
            raise Exception(
                f"Unhandled error type ({type(exc_repr)}), please report to"
                " neotest-python repo"
            )
        self.collect(
            short=self.get_short_output(report),
            status=NeotestResultStatus.FAILED,
            errors=errors,
        )

    def collect_error(self, exc_repr: ExceptionRepr) -> None:
        errors: List[NeotestError] = []
        error_message = exc_repr.reprcrash.message  # type: ignore
        error_line = self.find_error_line(exc_repr.reprtraceback)
        errors.append({"message": error_message, "line": error_line})

        self.collect(
            short=None,
            status=NeotestResultStatus.FAILED,
            errors=errors,
        )

    def collect_skipped(self, report: "pytest.TestReport") -> None:
        errors: List[NeotestError] = []
        exc_repr: Union[
            None, ExceptionInfo[BaseException], Tuple[str, int, str], str, TerminalRepr
        ] = report.longrepr

        # consider if we should add this error
        if isinstance(exc_repr, tuple):
            errors.append({"message": exc_repr[2], "line": exc_repr[1]})

        self.collect(
            short=self.get_short_output(report),
            status=NeotestResultStatus.SKIPPED,
            errors=errors,
        )


class NeotestDebugpyPlugin:
    """A pytest plugin that would make debugpy stop at thrown exceptions."""

    def pytest_exception_interact(
        self,
        node: Union["pytest.Item", "pytest.Collector"],
        call: "pytest.CallInfo",
        report: Union["pytest.CollectReport", "pytest.TestReport"],
    ):
        # call.excinfo: _pytest._code.ExceptionInfo
        self.maybe_debugpy_postmortem(call.excinfo._excinfo)

    @staticmethod
    def maybe_debugpy_postmortem(excinfo):
        """Make the debugpy debugger enter and stop at a raised exception.

        excinfo: A (type(e), e, e.__traceback__) tuple. See sys.exc_info()
        """
        # Reference: https://github.com/microsoft/debugpy/issues/723
        import threading
        try:
            import pydevd
        except ImportError:
            return  # debugpy or pydevd not available, do nothing

        py_db = pydevd.get_global_debugger()
        if py_db is None:
            # Do nothing if not running with a DAP debugger,
            # e.g. neotest was invoked with {strategy = dap}
            return

        thread = threading.current_thread()
        additional_info = py_db.set_additional_thread_info(thread)
        additional_info.is_tracing += 1
        try:
            py_db.stop_on_unhandled_exception(py_db, thread, additional_info, excinfo)
        finally:
            additional_info.is_tracing -= 1
