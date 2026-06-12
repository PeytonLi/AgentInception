-- AgentInception — ClickHouse schema. CONTRACTS.md §5.
-- Applied idempotently by scripts/ch_init.sh.

CREATE DATABASE IF NOT EXISTS agentinception;

CREATE TABLE IF NOT EXISTS agentinception.latent_memory_banks (
    page_key            String,
    domain              String,
    layer_id            UInt32,
    num_slots           UInt32,
    k_bank              String,   -- raw float32 bytes [8, num_slots, 128]
    v_bank              String,   -- raw float32 bytes [8, num_slots, 128]
    dom_structural_hash String,
    compiled_at         DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (page_key, layer_id);

CREATE TABLE IF NOT EXISTS agentinception.agent_steps (
    session_id      String,
    step            UInt32,
    mode            Enum8('baseline' = 1, 'mi' = 2),
    url             String,
    page_key        String,
    action_json     String,
    visible_tokens  UInt64,
    baseline_tokens UInt64,
    bank_found      UInt8,
    ts              DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (session_id, step);
