#!/usr/bin/env bash
# AgentInception — bring up ClickHouse and apply the schema. Idempotent.
# Usage: scripts/ch_init.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$SCRIPT_DIR/../infra"
SCHEMA_FILE="$INFRA_DIR/clickhouse/schema.sql"
CH_HTTP="${CLICKHOUSE_URL:-http://localhost:8123}"

echo "==> Starting ClickHouse (docker compose up -d)"
docker compose -f "$INFRA_DIR/docker-compose.yml" up -d

echo "==> Waiting for ClickHouse to answer /ping at $CH_HTTP ..."
for i in $(seq 1 60); do
  if curl -fsS "$CH_HTTP/ping" >/dev/null 2>&1; then
    echo "    ClickHouse is up."
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "ERROR: ClickHouse did not become ready in time." >&2
    exit 1
  fi
  sleep 2
done

echo "==> Applying schema ($SCHEMA_FILE)"
docker compose -f "$INFRA_DIR/docker-compose.yml" exec -T clickhouse \
  clickhouse-client --multiquery < "$SCHEMA_FILE"

echo "==> Verifying tables exist"
docker compose -f "$INFRA_DIR/docker-compose.yml" exec -T clickhouse \
  clickhouse-client --query "SHOW TABLES FROM agentinception"

echo "==> ClickHouse ready. CLICKHOUSE_URL=$CH_HTTP"
