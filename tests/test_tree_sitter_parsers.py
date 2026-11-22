"""Tests for multi-language parser, tree-sitter integration, and tricky syntax fixtures."""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import khaslana.intelligence.parser as parser_module
from khaslana.intelligence.callgraph import CallGraph
from khaslana.intelligence.parser import parse_source, parse_tree_sitter

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "tricky_syntax"


def test_typescript_tricky_parsing() -> None:
    path = FIXTURES_DIR / "sample.ts"
    content = path.read_text(encoding="utf-8")
    parsed = parse_source(str(path), content, language="typescript")

    assert parsed.language == "typescript"
    sym_map = {s.name: s for s in parsed.symbols}

    assert "Repository" in sym_map
    assert sym_map["Repository"].symbol_type == "interface"

    assert "OrderService" in sym_map
    assert sym_map["OrderService"].symbol_type == "class"

    assert "processOrder" in sym_map
    assert sym_map["processOrder"].symbol_type == "method"
    assert sym_map["processOrder"].parent_symbol == "OrderService"

    assert "constructor" in sym_map
    assert sym_map["constructor"].symbol_type == "method"
    assert sym_map["constructor"].parent_symbol == "OrderService"

    assert "calculateDiscount" in sym_map
    assert sym_map["calculateDiscount"].symbol_type == "function"
    assert sym_map["calculateDiscount"].parent_symbol is None

    # Check imports
    imported_mods = {i.imported_module for i in parsed.imports}
    assert "express" in imported_mods
    assert "./types" in imported_mods

    # Check calls
    callees = {c.callee_name for c in parsed.calls}
    assert "validateOrder" in callees
    assert "executePayment" in callees
    assert "logTrace" in callees
    assert "applyFactor" in callees


def test_go_tricky_parsing() -> None:
    path = FIXTURES_DIR / "sample.go"
    content = path.read_text(encoding="utf-8")
    parsed = parse_source(str(path), content, language="go")

    assert parsed.language == "go"
    sym_map = {s.name: s for s in parsed.symbols}

    assert "WorkerPool" in sym_map
    assert sym_map["WorkerPool"].symbol_type == "interface"

    assert "Server" in sym_map
    assert sym_map["Server"].symbol_type == "class"

    assert "NewServer" in sym_map
    assert sym_map["NewServer"].symbol_type == "function"
    assert sym_map["NewServer"].parent_symbol is None

    # Go receiver methods should have parent_symbol set to receiver struct
    assert "HandleRequest" in sym_map
    assert sym_map["HandleRequest"].symbol_type == "method"
    assert sym_map["HandleRequest"].parent_symbol == "Server"

    assert "logAccess" in sym_map
    assert sym_map["logAccess"].symbol_type == "method"
    assert sym_map["logAccess"].parent_symbol == "Server"

    # Check multi-line import block
    imported_mods = {i.imported_module for i in parsed.imports}
    assert "context" in imported_mods
    assert "fmt" in imported_mods
    assert "net/http" in imported_mods

    # Check calls
    callees = {c.callee_name for c in parsed.calls}
    assert "initMetrics" in callees
    assert "logAccess" in callees
    assert "processRequest" in callees
    assert "Println" in callees


def test_rust_tricky_parsing() -> None:
    path = FIXTURES_DIR / "sample.rs"
    content = path.read_text(encoding="utf-8")
    parsed = parse_source(str(path), content, language="rust")

    assert parsed.language == "rust"
    sym_map = {s.name: s for s in parsed.symbols}

    assert "Handler" in sym_map
    assert sym_map["Handler"].symbol_type == "interface"

    assert "ServiceRegistry" in sym_map
    assert sym_map["ServiceRegistry"].symbol_type == "class"

    assert "emit_metric" in sym_map
    assert sym_map["emit_metric"].symbol_type == "function"

    # Methods defined inside impl ServiceRegistry block
    assert "new" in sym_map
    assert sym_map["new"].symbol_type == "method"
    assert sym_map["new"].parent_symbol == "ServiceRegistry"

    assert "register_service" in sym_map
    assert sym_map["register_service"].symbol_type == "method"
    assert sym_map["register_service"].parent_symbol == "ServiceRegistry"

    # Check imports
    imported_mods = {i.imported_module for i in parsed.imports}
    assert "std::collections::HashMap" in imported_mods
    assert "std::sync::Arc" in imported_mods

    # Check calls
    callees = {c.callee_name for c in parsed.calls}
    assert "initialize_registry" in callees
    assert "validate_service" in callees
    assert "insert" in callees
    assert "record_counter" in callees


def test_java_tricky_parsing() -> None:
    path = FIXTURES_DIR / "sample.java"
    content = path.read_text(encoding="utf-8")
    parsed = parse_source(str(path), content, language="java")

    assert parsed.language == "java"
    sym_map = {s.name: s for s in parsed.symbols}

    assert "AsyncBatchProcessor" in sym_map
    assert sym_map["AsyncBatchProcessor"].symbol_type == "interface"

    assert "TransactionCoordinator" in sym_map
    assert sym_map["TransactionCoordinator"].symbol_type in ("class", "method")

    assert "executeTransaction" in sym_map
    assert sym_map["executeTransaction"].symbol_type == "method"
    assert sym_map["executeTransaction"].parent_symbol == "TransactionCoordinator"

    assert "processBatch" in sym_map
    assert sym_map["processBatch"].symbol_type == "method"
    assert sym_map["processBatch"].parent_symbol == "AsyncBatchProcessor"

    # Check imports
    imported_mods = {i.imported_module for i in parsed.imports}
    assert "java.util.List" in imported_mods
    assert "java.util.Map" in imported_mods
    assert "java.util.concurrent.CompletableFuture" in imported_mods

    # Check calls
    callees = {c.callee_name for c in parsed.calls}
    assert "initJournal" in callees
    assert "logTransactionStart" in callees
    assert "validatePayload" in callees
    assert "commitTransaction" in callees


def test_tree_sitter_warning_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    # Reset warning state
    parser_module._TREE_SITTER_WARNED = False

    with caplog.at_level(logging.WARNING):
        parse_source("test1.ts", "export class Test1 {}", language="typescript")
        assert "tree-sitter is not available" in caplog.text

        caplog.clear()
        parse_source("test2.go", "package main\ntype S struct{}", language="go")
        # Should not warn a second time
        assert "tree-sitter is not available" not in caplog.text


def test_tree_sitter_mock_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test tree-sitter parser path when mock tree-sitter is present."""
    mock_tree_sitter = MagicMock()
    mock_ts_module = MagicMock()
    monkeypatch.setitem(sys.modules, "tree_sitter", mock_tree_sitter)
    monkeypatch.setitem(sys.modules, "tree_sitter_typescript", mock_ts_module)

    # Mock node structure
    mock_class_node = MagicMock()
    mock_class_node.type = "class_declaration"
    mock_class_node.start_point.row = 0
    mock_class_node.end_point.row = 10
    name_node = MagicMock()
    name_node.text = b"MockClass"
    mock_class_node.child_by_field_name.return_value = name_node
    mock_class_node.children = []

    mock_root = MagicMock()
    mock_root.children = [mock_class_node]

    mock_parser_instance = MagicMock()
    mock_tree = MagicMock()
    mock_tree.root_node = mock_root
    mock_parser_instance.parse.return_value = mock_tree
    mock_tree_sitter.Parser.return_value = mock_parser_instance

    res = parse_tree_sitter("file.ts", "class MockClass {}", "typescript")
    assert res is not None
    assert len(res.symbols) >= 1
    assert res.symbols[0].name == "MockClass"
    assert res.symbols[0].symbol_type == "class"


def test_dynamic_dispatch_approximation_docstring() -> None:
    """Verify that dynamic dispatch approximations are documented."""
    assert parse_source.__doc__ is not None
    assert "Dynamic dispatch" in parse_source.__doc__
    assert "polymorphic calls" in parse_source.__doc__
    assert CallGraph.__doc__ is not None
    assert "approximate for dynamic dispatch" in CallGraph.__doc__
