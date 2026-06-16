"""
config/tenant.py — Tenant resolution for the Agentic Drift Detector.

Tenant ID is resolved from the TENANT_ID environment variable.
Default: "default" (single-tenant / local development mode).

To switch tenants, set TENANT_ID in your .env file or shell:
    TENANT_ID=acme-corp python run.py
"""

import logging
import os

log = logging.getLogger(__name__)

DEFAULT_TENANT = "default"


def get_current_tenant() -> str:
    """Return the active tenant ID from the environment."""
    tenant = os.getenv("TENANT_ID", DEFAULT_TENANT).strip()
    if not tenant:
        log.warning("TENANT_ID is blank — falling back to '%s'", DEFAULT_TENANT)
        return DEFAULT_TENANT
    return tenant


def ensure_tenant_exists(conn, tenant_id: str) -> None:
    """
    Upsert the tenant record into the tenants table.

    Safe to call on every connection — uses ON CONFLICT DO NOTHING.
    Requires a psycopg2 connection (PostgreSQL mode only).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tenants (id, name)
            VALUES (%s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (tenant_id, tenant_id),
        )
    conn.commit()
    log.debug("Tenant '%s' ensured in registry.", tenant_id)
