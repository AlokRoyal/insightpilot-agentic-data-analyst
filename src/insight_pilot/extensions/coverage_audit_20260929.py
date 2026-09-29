"""Curated daily analytics extension generated without a cloud API key."""

from insight_pilot.extensions.daily_features import run_feature


def run(frame):
    return run_feature(frame, 'coverage_audit')

