-- SpaceHASTEN legacy SQLite schema (frozen reference).
-- Source of truth: docs/CODEBASE_REFERENCE.md §A.1.
-- Statements appear in the order the legacy code creates them at first use.
-- Any future schema change MUST be additive and nullable; this file is immutable.

CREATE TABLE data (spacehastenid INTEGER PRIMARY KEY,reghash TEXT,smiles TEXT,smilesid TEXT,dock_score REAL,pred_score REAL,spacelight REAL,ftrees REAL,query INTEGER,dock_iteration INTEGER,pred_version INTEGER,simsearch_cycle INTEGER);
CREATE TABLE docking_param (dock_param BLOB);
CREATE TABLE docking_grid (dock_grid BLOB);
CREATE TABLE models (model_version INTEGER UNIQUE,model_tar BLOB);
CREATE TABLE properties (property TEXT,is_double INTEGER,min_limit TEXT,max_limit TEXT);
CREATE TABLE clusters(spacehastenid INTEGER PRIMARY KEY,clusterid INTEGER);
CREATE INDEX idx_reghash ON data(reghash);
