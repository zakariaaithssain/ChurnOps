BEGIN;
DROP SCHEMA IF EXISTS r1 CASCADE;
DROP SCHEMA IF EXISTS r2 CASCADE;
DROP SCHEMA IF EXISTS raw CASCADE;
DROP TABLE IF EXISTS public.members, public.labels, public.transactions,
                     public.userlogs, public.userlogs_raw CASCADE;
COMMIT;