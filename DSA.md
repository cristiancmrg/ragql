# DSA: Context-Assisted SQLite Querying

Status: directional design; implementation is incremental.

## Purpose

RagQL was developed to make SQLite data easy for an LLM to query by finding relevant context first, then grounding every generated answer in bounded, local database evidence.

This document defines that direction as the project's Design and System Architecture (DSA). It distinguishes the intended query experience from the narrower behavior implemented today.

## Current baseline

RagQL already uses Python's built-in `sqlite3` module for its chunk and vector metadata and can ingest `.db` files as searchable context. The current SQLite loader is intentionally small: it recognizes only the `Events` and `SystemStatus` tables and serializes their complete contents before embedding them.

The next architecture should generalize this without requiring a system `sqlite3` executable. Python's SQLite binding is the portable baseline; a CLI may remain an optional diagnostic tool.

## Prepared source contract

Before retrieval or SQL generation, RagQL must prepare each SQLite source:

1. Resolve the source to an absolute path and verify that it exists, is a regular file, and is readable by the RagQL process.
2. Verify SQLite support through Python and report the runtime version. Do not treat a missing `sqlite3` shell command as a database failure.
3. Open the source through a read-only URI connection and enable `PRAGMA query_only = ON` plus a short busy timeout.
4. Discover tables, views, columns, keys, indexes, and declared relationships without assuming application-specific table names.
5. Build a compact catalog containing schema text, table descriptions, bounded representative samples, and provenance.
6. Record optional capabilities such as FTS5 or JSON support. Missing optional extensions must reduce features, not make the source unusable.
7. Return normalized, actionable errors for missing paths, permissions, locks, malformed databases, unsupported encryption, and empty schemas.

The source is prepared only after these checks succeed. Retrieval and query planning consume the prepared catalog rather than reopening an unknown path ad hoc.

## Query flow

```text
natural-language question
  -> search the prepared schema and sample catalog
  -> select relevant tables, columns, relationships, and examples
  -> give only that context to the SQL planner
  -> validate the proposed statement against the selected schema
  -> execute one bounded read-only query
  -> convert rows plus provenance into answer context
  -> ask the LLM to answer from that evidence
```

Context search is the first stage, not a substitute for SQL. It narrows the database surface enough for the model to plan a precise query while avoiding full-schema and full-table prompt dumps. Query results become a second, stronger context layer for the final answer.

## Agent-native, zero-key interfaces

RagQL should expose a Graphify-style, query-first CLI that coding agents can call directly. Codex, Claude Code, and any other agent with shell or MCP tool access should receive useful database context without configuring a model provider inside RagQL.

Core preparation, cataloging, context search, SQL validation, and read-only execution must therefore require no API key. The zero-key retrieval baseline should use SQLite FTS5/BM25 when available and deterministic lexical ranking when it is not. Optional local embeddings may improve ranking, but failure to install or run an embedding model must not disable schema search or safe SQL execution.

The host agent supplies semantic reasoning:

1. RagQL prepares the source and returns a compact, searchable catalog.
2. The host asks RagQL for question-relevant schema and sample context.
3. The host proposes SQL using that evidence.
4. RagQL validates and executes the statement within local policy limits.
5. The host composes the answer from structured rows and provenance.

Standalone Ollama or cloud-model answering remains an optional adapter. Cloud credentials must never be needed by the core CLI or MCP server, and no provider fallback may silently send local database content off-machine.

### Proposed CLI

```bash
ragql prepare ./data.db --name app
ragql query app "Which operations retried?" --budget 1200 --format json
ragql execute app --sql-file query.sql --row-budget 100 --format json
ragql outcome TRACE_ID --status useful
ragql mcp --stdio
```

These commands are proposed interfaces, not current behavior. Their contracts should be stable:

- `prepare` validates the source and refreshes its local catalog only when stale.
- `query` searches prepared context; it does not require or invoke an LLM.
- `execute` accepts one validated, read-only statement plus separately encoded parameters.
- `outcome` stores local feedback as `useful`, `dead_end`, or `corrected` so later searches can prefer good evidence and avoid repeated dead ends.
- `mcp --stdio` exposes the same operations as MCP tools without changing their safety or budget rules.

Human output may be concise, but `--format json` is the canonical agent contract. Every response should contain source identity, selected catalog objects, provenance, limits, truncation state, trace ID, and normalized errors. Exit codes should distinguish invalid input, inaccessible sources, rejected SQL, exhausted budgets, and internal faults.

### MCP mapping

The local MCP server should be a thin adapter over the same application services used by the CLI:

- `sqlite_prepare`
- `sqlite_query_context`
- `sqlite_execute_read_only`
- `sqlite_record_outcome`

No MCP tool owns model credentials or model selection. This keeps behavior identical across Codex, Claude Code, editor agents, scripts, and future clients while avoiding provider-specific code in the data-access core.

## Budget contract

Every query must declare or inherit a budget. `--budget` caps the approximate tokens returned as context, not model billing. Separate hard controls protect database work:

- `--top-k` limits retrieved catalog entries;
- `--row-budget` limits fetched rows;
- `--byte-budget` limits serialized evidence;
- `--timeout-ms` limits wall-clock execution;
- `--operation-budget` drives a SQLite progress handler;
- local policy sets maximum values that callers cannot exceed.

Budget priority is: safety policy, caller request, then defaults. Results must report requested, effective, and consumed values plus whether output was truncated and why. RagQL should rank evidence before truncation, preserve provenance for every retained item, and never hide partial execution behind a successful-looking answer.

Prepared catalogs enable a fast path: unchanged sources skip discovery and indexing. Query expansion may use only vocabulary present in that catalog; selected expansion terms should be returned in the trace so agents can audit retrieval instead of retrying unexplained searches.

## Caveman-compatible outcomes

RagQL should support compact outcomes without coupling core logic to a specific agent:

- `--style caveman` renders a deterministic terse human summary without an LLM or API key.
- `--format json` remains lossless and lets a host-side Caveman skill compress only presentation.
- SQL, identifiers, error codes, provenance, numeric limits, and correction text are never abbreviated or dropped.
- Outcome feedback records `useful`, `dead_end`, or `corrected`; compression never changes that state.
- Summary, evidence, execution trace, and diagnostics stay separate so agents can request only the layer their remaining budget allows.

This integration saves context tokens while keeping the structured result authoritative. A concise renderer is an output policy, not a second reasoning pass.

## Proposed components

- `SQLiteSourcePreparer`: validates access, opens read-only connections, detects capabilities, and produces normalized diagnostics.
- `SchemaCatalog`: stores table, column, key, index, relationship, and provenance metadata.
- `ContextRetriever`: ranks catalog entries and safe samples against the user's question.
- `SQLPlanner`: produces one statement from the question and retrieved schema context.
- `ReadOnlyExecutor`: validates and executes the statement with limits.
- `AnswerComposer`: grounds the response in returned rows and identifies its source tables.
- `QueryTrace`: records retrieval choices, validated SQL, limits, timings, and non-sensitive error categories for debugging and evaluation.
- `BudgetPolicy`: resolves local caps, caller budgets, and defaults for retrieval and execution.
- `AgentAdapter`: exposes shared application services through JSON CLI and MCP stdio interfaces.
- `OutcomeStore`: records useful, dead-end, and corrected traces locally.
- `ResultRenderer`: produces normal, Caveman-style, or canonical JSON output without changing evidence.

These boundaries keep source access deterministic even when retrieval or model providers change.

## Safety boundary

LLM-produced SQL is untrusted input. The executor must:

- accept a single `SELECT` or read-only `WITH` statement only;
- reject writes, DDL, `ATTACH`, extension loading, and write-capable pragmas;
- bind values as parameters instead of interpolating user text;
- restrict referenced objects to the prepared catalog;
- enforce row, byte, time, and SQLite operation limits;
- preserve the original database and its WAL/sidecar behavior;
- redact configured sensitive columns before remote model calls; and
- keep all database content local unless the user explicitly selects a remote provider.

Read-only URI mode and `query_only` are defense layers, not replacements for statement validation and execution limits.

## Error model

Preparation should convert low-level exceptions into stable categories:

- `SOURCE_NOT_FOUND`
- `SOURCE_NOT_READABLE`
- `SQLITE_OPEN_FAILED`
- `SQLITE_LOCKED`
- `SQLITE_MALFORMED`
- `SQLITE_ENCRYPTED_OR_UNSUPPORTED`
- `SCHEMA_EMPTY`
- `QUERY_REJECTED`
- `QUERY_LIMIT_EXCEEDED`

Each error should include the failed stage and a safe remediation hint, never credentials, row contents, or secret-bearing paths. This lets agents distinguish an unavailable shell executable from a real database-access problem and avoids blind retries.

## Adaptation plan

1. Generalize the SQLite loader from two named tables to schema discovery and bounded samples.
2. Add a preparation API and CLI preflight backed by Python `sqlite3`.
3. Add zero-key FTS5/BM25 retrieval with a deterministic lexical fallback.
4. Index schema/catalog entries separately from row content and retrieve both with provenance.
5. Add shared budget policy and canonical JSON result schemas.
6. Add the validated read-only planner/executor path.
7. Add CLI commands, then expose those same services through MCP stdio.
8. Add Caveman-style rendering and local outcome feedback without modifying canonical evidence.
9. Feed bounded query results into the existing optional answer pipeline.
10. Add deterministic tests for permissions, locks, malformed files, arbitrary schemas, rejected writes, query limits, adapter parity, and answers grounded in returned rows.
11. Add evaluation cases comparing context-assisted queries with one-shot SQL generation, measuring validity, retries, latency, budget use, and answer grounding.

## Acceptance criteria

- A prepared, readable SQLite database can be inspected and queried without a system `sqlite3` command.
- Core CLI and MCP operations work without model-provider API keys or a running model server.
- Arbitrary application schemas are discoverable; no fixed table name is required.
- Generated SQL cannot mutate the source database.
- Every answer can identify the schema context and result rows that grounded it.
- CLI and MCP calls enforce identical budgets and return equivalent canonical JSON.
- Every truncated result reports its effective budget and truncation reason.
- Caveman-style output remains technically equivalent to the canonical result.
- Expected access and query failures produce one actionable diagnostic instead of repeated opaque retries.
- Local mode sends no schema, samples, rows, or prompts to a remote service.
