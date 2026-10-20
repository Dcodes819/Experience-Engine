"""initial experience engine schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-10-13
"""

from alembic import op

from packages.engine.infrastructure.db import Base

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
