"""LangGraph wiring and shared pipeline state.

``graph.py`` compiles the 8-node StateGraph and exports the ``app`` callable;
``state.py`` defines ``AgentState`` (the shared mutable state flowing through
every node); ``checkpoint.py`` persists per-challenge progress.

Inputs:  AgentState initial values — challenge IDs, paths, config flags
Outputs: AgentState final values — generated courses, ranking results, all
         reports; checkpoint files under KB_DIR/progress/
"""
