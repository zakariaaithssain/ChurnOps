-- Drops everything created by create_schemas.sql.
-- Schema list must match config.SCHEMAS plus config.RAW_SCHEMA.
-- Keep in sync with data.execute_queries.reset() prompt, which builds the same list.
BEGIN;
DROP SCHEMA IF EXISTS r1 CASCADE;
DROP SCHEMA IF EXISTS r2 CASCADE;
DROP SCHEMA IF EXISTS raw CASCADE;
COMMIT;
