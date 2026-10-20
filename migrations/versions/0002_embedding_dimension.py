"""persist embedding dimensions for provider compatibility

Revision ID: 0002_embedding_dimension
Revises: 0001_initial_schema
Create Date: 2026-10-20
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0002_embedding_dimension"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("experiences")}
    if "embedding_dimension" not in columns:
        op.add_column(
            "experiences",
            sa.Column("embedding_dimension", sa.Integer(), nullable=False, server_default="256"),
        )


def downgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("experiences")}
    if "embedding_dimension" in columns:
        op.drop_column("experiences", "embedding_dimension")
