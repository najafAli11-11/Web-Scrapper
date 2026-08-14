"""UI package: read-only data access + Streamlit app (Milestone 9).

The UI is a consumer of the pipeline: it reads queue/event state read-only,
spawns the M7 orchestrator CLI as a detached child for ingestion, and drives
chat through the corpus and the M8 live-query path. It never writes pipeline
state itself.
"""
