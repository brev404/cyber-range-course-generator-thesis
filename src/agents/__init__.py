"""LangGraph agent nodes for the course-generation pipeline.

One module per LLM/graph node — coordinator, validator, content_generation,
course_terminology_checker, mapping, ranking, hitl — each exporting a single
``run_*`` function consumed by ``src/core/graph.py``.

Inputs:  AgentState (src.core.state), llm_service, vector_db_service,
         app_settings (src.config.settings)
Outputs: Updated AgentState fields — generated_courses, ranking_reports,
         validation_reports, course_terminology_issues
"""
