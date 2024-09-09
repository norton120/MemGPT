"""init

Revision ID: 8ab5757fa7a1
Revises:
Create Date: 2024-07-05 18:58:31.038011

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8ab5757fa7a1"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "organization",
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("_id", sa.UUID(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.UUID(), nullable=True),
        sa.Column("_last_updated_by_id", sa.UUID(), nullable=True),
        sa.PrimaryKeyConstraint("_id"),
    )
    op.create_table(
        "agent",
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("persona", sa.String(), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("_metadata", sa.JSON(), nullable=False),
        sa.Column("human", sa.String(), nullable=False),
        sa.Column("preset", sa.String(), nullable=False),
        sa.Column("llm_config", sa.JSON(), nullable=False),
        sa.Column("embedding_config", sa.JSON(), nullable=False),
        sa.Column("_id", sa.UUID(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.UUID(), nullable=True),
        sa.Column("_last_updated_by_id", sa.UUID(), nullable=True),
        sa.Column("_organization_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["_organization_id"],
            ["organization._id"],
        ),
        sa.PrimaryKeyConstraint("_id"),
    )
    op.create_table(
        "document",
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("data_source", sa.String(), nullable=True),
        sa.Column("metadata_", sa.JSON(), nullable=True),
        sa.Column("_organization_id", sa.UUID(), nullable=False),
        sa.Column("_id", sa.UUID(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.UUID(), nullable=True),
        sa.Column("_last_updated_by_id", sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(
            ["_organization_id"],
            ["organization._id"],
        ),
        sa.PrimaryKeyConstraint("_id"),
    )
    op.create_table(
        "memory_template",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("_id", sa.UUID(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.UUID(), nullable=True),
        sa.Column("_last_updated_by_id", sa.UUID(), nullable=True),
        sa.Column("_organization_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["_organization_id"],
            ["organization._id"],
        ),
        sa.PrimaryKeyConstraint("_id"),
    )
    op.create_table(
        "preset",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("system", sa.String(), nullable=True),
        sa.Column("human", sa.String(), nullable=False),
        sa.Column("human_name", sa.String(), nullable=False),
        sa.Column("persona", sa.String(), nullable=False),
        sa.Column("persona_name", sa.String(), nullable=False),
        sa.Column("functions_schema", sa.JSON(), nullable=False),
        sa.Column("_id", sa.UUID(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.UUID(), nullable=True),
        sa.Column("_last_updated_by_id", sa.UUID(), nullable=True),
        sa.Column("_organization_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["_organization_id"],
            ["organization._id"],
        ),
        sa.PrimaryKeyConstraint("_id"),
        sa.UniqueConstraint("_organization_id", "name", name="unique_name_organization"),
    )
    op.create_table(
        "source",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("_organization_id", sa.UUID(), nullable=False),
        sa.Column("_id", sa.UUID(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.UUID(), nullable=True),
        sa.Column("_last_updated_by_id", sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(
            ["_organization_id"],
            ["organization._id"],
        ),
        sa.PrimaryKeyConstraint("_id"),
    )
    op.create_table(
        "tool",
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("source_code", sa.String(), nullable=True),
        sa.Column("json_schema", sa.JSON(), nullable=False),
        sa.Column("_id", sa.UUID(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.UUID(), nullable=True),
        sa.Column("_last_updated_by_id", sa.UUID(), nullable=True),
        sa.Column("_organization_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["_organization_id"],
            ["organization._id"],
        ),
        sa.PrimaryKeyConstraint("_id"),
    )
    op.create_table(
        "user",
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("_id", sa.UUID(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.UUID(), nullable=True),
        sa.Column("_last_updated_by_id", sa.UUID(), nullable=True),
        sa.Column("_organization_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["_organization_id"],
            ["organization._id"],
        ),
        sa.PrimaryKeyConstraint("_id"),
    )
    op.create_table(
        "job",
        sa.Column("status", sa.Enum("created", "running", "completed", "failed", name="jobstatus"), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("metadata_", sa.JSON(), nullable=True),
        sa.Column("_user_id", sa.UUID(), nullable=False),
        sa.Column("_id", sa.UUID(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.UUID(), nullable=True),
        sa.Column("_last_updated_by_id", sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(
            ["_user_id"],
            ["user._id"],
        ),
        sa.PrimaryKeyConstraint("_id"),
    )
    op.create_table(
        "passage",
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("embedding_config", sa.JSON(), nullable=True),
        sa.Column("data_source", sa.String(), nullable=True),
        sa.Column("metadata_", sa.JSON(), nullable=True),
        sa.Column("_document_id", sa.UUID(), nullable=False),
        sa.Column("_id", sa.UUID(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.UUID(), nullable=True),
        sa.Column("_last_updated_by_id", sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(
            ["_document_id"],
            ["document._id"],
        ),
        sa.PrimaryKeyConstraint("_id"),
    )
    op.create_table(
        "sources_agents",
        sa.Column("_agent_id", sa.UUID(), nullable=False),
        sa.Column("_source_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["_agent_id"],
            ["agent._id"],
        ),
        sa.ForeignKeyConstraint(
            ["_source_id"],
            ["source._id"],
        ),
        sa.PrimaryKeyConstraint("_agent_id", "_source_id"),
    )
    op.create_table(
        "sources_presets",
        sa.Column("_preset_id", sa.UUID(), nullable=False),
        sa.Column("_source_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["_preset_id"],
            ["preset._id"],
        ),
        sa.ForeignKeyConstraint(
            ["_source_id"],
            ["source._id"],
        ),
        sa.PrimaryKeyConstraint("_preset_id", "_source_id"),
    )
    op.create_table(
        "token",
        sa.Column("_temporary_shim_api_key", sa.String(), nullable=True),
        sa.Column("hash", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("_id", sa.UUID(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.UUID(), nullable=True),
        sa.Column("_last_updated_by_id", sa.UUID(), nullable=True),
        sa.Column("_user_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["_user_id"],
            ["user._id"],
        ),
        sa.PrimaryKeyConstraint("_id"),
    )
    op.create_table(
        "tools_agents",
        sa.Column("_agent_id", sa.UUID(), nullable=False),
        sa.Column("_tool_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["_agent_id"],
            ["agent._id"],
        ),
        sa.ForeignKeyConstraint(
            ["_tool_id"],
            ["tool._id"],
        ),
        sa.PrimaryKeyConstraint("_agent_id", "_tool_id"),
    )
    op.create_table(
        "tools_presets",
        sa.Column("_preset_id", sa.UUID(), nullable=False),
        sa.Column("_tool_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["_preset_id"],
            ["preset._id"],
        ),
        sa.ForeignKeyConstraint(
            ["_tool_id"],
            ["tool._id"],
        ),
        sa.PrimaryKeyConstraint("_preset_id", "_tool_id"),
    )
    op.create_table(
        "users_agents",
        sa.Column("_agent_id", sa.UUID(), nullable=False),
        sa.Column("_user_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["_agent_id"],
            ["agent._id"],
        ),
        sa.ForeignKeyConstraint(
            ["_user_id"],
            ["user._id"],
        ),
        sa.PrimaryKeyConstraint("_agent_id", "_user_id"),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("users_agents")
    op.drop_table("tools_presets")
    op.drop_table("tools_agents")
    op.drop_table("token")
    op.drop_table("sources_presets")
    op.drop_table("sources_agents")
    op.drop_table("passage")
    op.drop_table("job")
    op.drop_table("user")
    op.drop_table("tool")
    op.drop_table("source")
    op.drop_table("preset")
    op.drop_table("memory_template")
    op.drop_table("document")
    op.drop_table("agent")
    op.drop_table("organization")
    # ### end Alembic commands ###
