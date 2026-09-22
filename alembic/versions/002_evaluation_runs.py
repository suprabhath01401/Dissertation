"""evaluation_runs

Revision ID: 002
Revises: 001
Create Date: 2026-07-14 00:00:00.000000

"""
from typing import Sequence, Union
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("evaluation_results", sa.Column("run_id", UUID(as_uuid=True)))
    op.create_index("ix_evaluation_results_run_id", "evaluation_results", ["run_id"])

    op.create_table(
        "evaluation_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", UUID(as_uuid=True), nullable=False),
        sa.Column("dataset", sa.String(100), nullable=False),
        sa.Column("config", sa.String(100), nullable=False),
        sa.Column("metric", sa.String(100), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("n_samples", sa.Integer(), nullable=False),
        sa.Column("run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("notes", sa.Text()),
    )
    op.create_index("ix_evaluation_runs_run_id", "evaluation_runs", ["run_id"])


def downgrade() -> None:
    op.drop_table("evaluation_runs")
    op.drop_index("ix_evaluation_results_run_id", table_name="evaluation_results")
    op.drop_column("evaluation_results", "run_id")
