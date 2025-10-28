"""Add privacy and GDPR compliance tables

Revision ID: a1b2c3d4e5f6
Revises: dfb89bc7649d
Create Date: 2025-10-28 06:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'dfb89bc7649d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - add privacy tables."""

    # Create user_consents table
    op.create_table(
        'user_consents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('consent_type', sa.Enum('necessary', 'functional', 'analytics', 'marketing', name='consenttype'), nullable=False),
        sa.Column('status', sa.Enum('granted', 'denied', 'pending', 'revoked', name='consentstatus'), nullable=False),
        sa.Column('granted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_consents_user_id'), 'user_consents', ['user_id'], unique=False)
    op.create_index('idx_user_consents_user_type', 'user_consents', ['user_id', 'consent_type'], unique=False)

    # Create telemetry_preferences table
    op.create_table(
        'telemetry_preferences',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('telemetry_mode', sa.Enum('disabled', 'minimal', 'standard', 'full', name='telemetrymode'), nullable=False),
        sa.Column('do_not_track', sa.Boolean(), nullable=False),
        sa.Column('collect_errors', sa.Boolean(), nullable=False),
        sa.Column('collect_performance', sa.Boolean(), nullable=False),
        sa.Column('collect_usage', sa.Boolean(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_telemetry_preferences_user_id'), 'telemetry_preferences', ['user_id'], unique=True)

    # Create data_retention table
    op.create_table(
        'data_retention',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('conversation_days', sa.Integer(), nullable=False),
        sa.Column('memory_days', sa.Integer(), nullable=False),
        sa.Column('audit_log_days', sa.Integer(), nullable=False),
        sa.Column('telemetry_days', sa.Integer(), nullable=False),
        sa.Column('auto_purge_enabled', sa.Boolean(), nullable=False),
        sa.Column('last_purge_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_data_retention_user_id'), 'data_retention', ['user_id'], unique=True)

    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('resource_type', sa.String(length=100), nullable=True),
        sa.Column('resource_id', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('performed_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_action'), 'audit_logs', ['action'], unique=False)
    op.create_index(op.f('ix_audit_logs_performed_at'), 'audit_logs', ['performed_at'], unique=False)
    op.create_index('idx_audit_logs_user_action', 'audit_logs', ['user_id', 'action', 'performed_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema - remove privacy tables."""

    # Drop audit_logs table
    op.drop_index('idx_audit_logs_user_action', table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_performed_at'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_action'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_user_id'), table_name='audit_logs')
    op.drop_table('audit_logs')

    # Drop data_retention table
    op.drop_index(op.f('ix_data_retention_user_id'), table_name='data_retention')
    op.drop_table('data_retention')

    # Drop telemetry_preferences table
    op.drop_index(op.f('ix_telemetry_preferences_user_id'), table_name='telemetry_preferences')
    op.drop_table('telemetry_preferences')

    # Drop user_consents table
    op.drop_index('idx_user_consents_user_type', table_name='user_consents')
    op.drop_index(op.f('ix_user_consents_user_id'), table_name='user_consents')
    op.drop_table('user_consents')
