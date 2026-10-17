"""NIST AI RMF ingestion pipeline: official PDFs -> validated page-level records.

Why a package: every stage reads the previous stage's cache file, so each can
be rerun and inspected on its own (see docs/INGESTION_PLAN.md).
"""
