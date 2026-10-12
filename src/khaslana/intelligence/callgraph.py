"""Static call graph constructor, query engine, cycle detector, and visualizer."""

from __future__ import annotations

from collections import defaultdict, deque

from khaslana.intelligence.symbols import SymbolTable


class CallGraph:
    """Builds and queries static caller/callee graphs using the SymbolTable."""

    def __init__(self, symbol_table: SymbolTable) -> None:
        self.symbol_table = symbol_table
        self._caller_to_callees: dict[str, set[str]] = defaultdict(set)
        self._callee_to_callers: dict[str, set[str]] = defaultdict(set)
        self._build_graph()

    @classmethod
    def from_symbol_table(cls, symbol_table: SymbolTable) -> CallGraph:
        """Create a CallGraph instance from a SymbolTable."""
        return cls(symbol_table)

    def _build_graph(self) -> None:
        """Populate adjacency lists from symbol table calls."""
        cur = self.symbol_table._conn.execute(
            "SELECT caller_symbol, callee_name FROM calls"
        )
        for row in cur.fetchall():
            caller = row["caller_symbol"]
            callee = row["callee_name"]
            self._caller_to_callees[caller].add(callee)
            self._callee_to_callers[callee].add(caller)

    def refresh(self) -> None:
        """Rebuild in-memory adjacency maps from the symbol table."""
        self._caller_to_callees.clear()
        self._callee_to_callers.clear()
        self._build_graph()

    def get_transitive_callees(self, symbol_name: str, max_depth: int = 5) -> list[str]:
        """Find all transitive callees up to max_depth."""
        return self.get_callees(symbol_name, max_depth=max_depth)

    def get_transitive_callers(self, symbol_name: str, max_depth: int = 5) -> list[str]:
        """Find all transitive callers up to max_depth."""
        return self.get_callers(symbol_name, max_depth=max_depth)

    def get_callees(self, symbol_name: str, max_depth: int = 1) -> list[str]:
        """Find all functions/methods called by symbol_name up to max_depth."""
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(symbol_name, 0)])

        while queue:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Look up callees
            found_callees = set()
            for key in [curr, curr.split(".")[-1]]:
                found_callees.update(self._caller_to_callees.get(key, set()))

            for callee in found_callees:
                if callee not in visited and callee != symbol_name:
                    visited.add(callee)
                    queue.append((callee, depth + 1))

        return sorted(visited)

    def get_callers(self, symbol_name: str, max_depth: int = 1) -> list[str]:
        """Find all functions/methods that call symbol_name up to max_depth."""
        visited: set[str] = set()
        clean_name = symbol_name.split(".")[-1]
        queue: deque[tuple[str, int]] = deque([(clean_name, 0), (symbol_name, 0)])

        while queue:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue

            callers = self._callee_to_callers.get(curr, set())
            for caller in callers:
                if caller not in visited and caller != symbol_name:
                    visited.add(caller)
                    queue.append((caller, depth + 1))

        return sorted(visited)

    def find_cycles(self) -> list[list[str]]:
        """Detect circular call dependencies using DFS."""
        visited: dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited
        cycles: list[list[str]] = []
        path: list[str] = []

        all_nodes = set(self._caller_to_callees.keys()).union(
            set(self._callee_to_callers.keys())
        )

        def _dfs(node: str) -> None:
            visited[node] = 1
            path.append(node)

            for neighbor in self._caller_to_callees.get(node, set()):
                state = visited.get(neighbor, 0)
                if state == 1:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append([*path[cycle_start:], neighbor])
                elif state == 0:
                    _dfs(neighbor)

            path.pop()
            visited[node] = 2

        for node in all_nodes:
            if visited.get(node, 0) == 0:
                _dfs(node)

        return cycles

    def export_mermaid(self, root_symbol: str, depth: int = 2) -> str:
        """Generate a Mermaid flowchart diagram around root_symbol."""
        lines = ["```mermaid", "graph TD"]
        visited_edges: set[tuple[str, str]] = set()

        def _collect(sym: str, d: int) -> None:
            if d >= depth:
                return

            # Add callees
            clean = sym.split(".")[-1]
            callees = set()
            for key in [sym, clean]:
                callees.update(self._caller_to_callees.get(key, set()))

            for callee in callees:
                edge = (sym, callee)
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    lines.append(f'    "{sym}" --> "{callee}"')
                    _collect(callee, d + 1)

            # Add callers
            callers = self._callee_to_callers.get(clean, set())
            for caller in callers:
                edge = (caller, sym)
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    lines.append(f'    "{caller}" --> "{sym}"')
                    _collect(caller, d + 1)

        _collect(root_symbol, 0)

        if len(lines) == 2:
            lines.append(f'    "{root_symbol}"["{root_symbol} (Isolated)"]')

        lines.append("```")
        return "\n".join(lines)

    def export_dot(self, root_symbol: str | None = None, depth: int = 2) -> str:
        """Generate a Graphviz DOT representation around root_symbol, or full graph if None."""
        lines = ["digraph CallGraph {", '  rankdir="LR";', '  node [shape="box"];']
        visited_edges: set[tuple[str, str]] = set()

        if root_symbol is None:
            for caller, callees in self._caller_to_callees.items():
                for callee in callees:
                    lines.append(f'  "{caller}" -> "{callee}";')
            lines.append("}")
            return "\n".join(lines)

        def _collect(sym: str, d: int) -> None:
            if d >= depth:
                return

            clean = sym.split(".")[-1]
            callees = set()
            for key in [sym, clean]:
                callees.update(self._caller_to_callees.get(key, set()))

            for callee in callees:
                edge = (sym, callee)
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    lines.append(f'  "{sym}" -> "{callee}";')
                    _collect(callee, d + 1)

            callers = self._callee_to_callers.get(clean, set())
            for caller in callers:
                edge = (caller, sym)
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    lines.append(f'  "{caller}" -> "{sym}";')
                    _collect(caller, d + 1)

        _collect(root_symbol, 0)

        if len(lines) == 3:
            lines.append(f'  "{root_symbol}";')

        lines.append("}")
        return "\n".join(lines)
