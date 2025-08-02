-- Database Query Optimization for Media Server
-- PostgreSQL 16+ with advanced performance features

-- =====================================================
-- PERFORMANCE SCHEMA AND MONITORING SETUP
-- =====================================================

-- Create performance monitoring schema
CREATE SCHEMA IF NOT EXISTS performance;

-- Enable query performance tracking
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_stat_monitor;
CREATE EXTENSION IF NOT EXISTS pg_wait_sampling;
CREATE EXTENSION IF NOT EXISTS hypopg; -- Hypothetical indexes

-- =====================================================
-- OPTIMIZED TABLE STRUCTURES
-- =====================================================

-- Media items table with partitioning
CREATE TABLE IF NOT EXISTS media_items (
    id BIGSERIAL,
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    duration INTEGER,
    codec VARCHAR(50),
    resolution VARCHAR(20),
    bitrate INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    popularity_score FLOAT DEFAULT 0,
    storage_tier INTEGER DEFAULT 1,
    metadata JSONB,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create partitions for better performance
CREATE TABLE media_items_2024_01 PARTITION OF media_items
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE media_items_2024_02 PARTITION OF media_items
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- Continue for all months...

-- Create optimized indexes
CREATE INDEX CONCURRENTLY idx_media_title_gin ON media_items USING gin(title gin_trgm_ops);
CREATE INDEX CONCURRENTLY idx_media_file_path ON media_items USING btree(file_path);
CREATE INDEX CONCURRENTLY idx_media_popularity ON media_items USING btree(popularity_score DESC);
CREATE INDEX CONCURRENTLY idx_media_accessed ON media_items USING btree(accessed_at DESC);
CREATE INDEX CONCURRENTLY idx_media_metadata ON media_items USING gin(metadata);
CREATE INDEX CONCURRENTLY idx_media_codec_res ON media_items USING btree(codec, resolution);

-- Covering index for common queries
CREATE INDEX CONCURRENTLY idx_media_covering ON media_items 
    (popularity_score DESC, accessed_at DESC) 
    INCLUDE (title, file_path, duration, resolution);

-- =====================================================
-- USER ACTIVITY TRACKING (HYPERTABLE)
-- =====================================================

-- Create TimescaleDB extension for time-series data
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- User activity hypertable
CREATE TABLE user_activity (
    time TIMESTAMPTZ NOT NULL,
    user_id BIGINT NOT NULL,
    media_id BIGINT NOT NULL,
    action VARCHAR(50) NOT NULL,
    position INTEGER,
    quality VARCHAR(20),
    bandwidth INTEGER,
    buffer_health FLOAT,
    client_info JSONB
);

-- Convert to hypertable for better time-series performance
SELECT create_hypertable('user_activity', 'time', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for activity queries
CREATE INDEX idx_activity_user_time ON user_activity (user_id, time DESC);
CREATE INDEX idx_activity_media_time ON user_activity (media_id, time DESC);
CREATE INDEX idx_activity_action ON user_activity (action, time DESC);

-- Continuous aggregate for real-time analytics
CREATE MATERIALIZED VIEW activity_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    media_id,
    COUNT(DISTINCT user_id) as unique_viewers,
    COUNT(*) as total_views,
    AVG(CASE WHEN action = 'play' THEN position END) as avg_watch_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY bandwidth) as p95_bandwidth
FROM user_activity
WHERE action IN ('play', 'pause', 'complete')
GROUP BY hour, media_id
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('activity_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- =====================================================
-- OPTIMIZED SEARCH FUNCTIONALITY
-- =====================================================

-- Full-text search configuration
CREATE TEXT SEARCH CONFIGURATION media_search (COPY = english);

-- Add custom dictionaries for media terms
ALTER TEXT SEARCH CONFIGURATION media_search
    ALTER MAPPING FOR word WITH simple, english_stem;

-- Materialized view for search with pre-computed vectors
CREATE MATERIALIZED VIEW media_search_index AS
SELECT 
    id,
    title,
    file_path,
    to_tsvector('media_search', 
        title || ' ' || 
        COALESCE(metadata->>'description', '') || ' ' ||
        COALESCE(metadata->>'tags', '')
    ) as search_vector,
    popularity_score,
    accessed_at
FROM media_items;

-- Create GiST index for full-text search
CREATE INDEX idx_media_search_vector ON media_search_index 
    USING gist(search_vector);

-- Function for smart search with ranking
CREATE OR REPLACE FUNCTION search_media(
    query_text TEXT,
    limit_count INTEGER DEFAULT 50
)
RETURNS TABLE (
    media_id BIGINT,
    title TEXT,
    file_path TEXT,
    rank FLOAT,
    popularity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id,
        m.title,
        m.file_path,
        ts_rank_cd(s.search_vector, query::tsquery) AS rank,
        m.popularity_score
    FROM media_search_index s
    JOIN media_items m ON m.id = s.id
    WHERE s.search_vector @@ plainto_tsquery('media_search', query_text)
    ORDER BY 
        ts_rank_cd(s.search_vector, plainto_tsquery('media_search', query_text)) * 
        (1 + LOG(GREATEST(m.popularity_score, 1))) DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- RECOMMENDATION ENGINE QUERIES
-- =====================================================

-- Materialized view for collaborative filtering
CREATE MATERIALIZED VIEW user_similarity AS
WITH user_vectors AS (
    SELECT 
        user_id,
        array_agg(media_id ORDER BY media_id) as watched_media
    FROM (
        SELECT DISTINCT user_id, media_id 
        FROM user_activity 
        WHERE action = 'complete'
    ) t
    GROUP BY user_id
)
SELECT 
    a.user_id as user_a,
    b.user_id as user_b,
    cardinality(
        ARRAY(SELECT unnest(a.watched_media) 
        INTERSECT 
        SELECT unnest(b.watched_media))
    )::FLOAT / 
    GREATEST(cardinality(a.watched_media), cardinality(b.watched_media)) as similarity
FROM user_vectors a
CROSS JOIN user_vectors b
WHERE a.user_id < b.user_id
AND cardinality(
    ARRAY(SELECT unnest(a.watched_media) 
    INTERSECT 
    SELECT unnest(b.watched_media))
) > 3;

CREATE INDEX idx_user_similarity ON user_similarity(user_a, similarity DESC);

-- Function for generating recommendations
CREATE OR REPLACE FUNCTION get_recommendations(
    p_user_id BIGINT,
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    media_id BIGINT,
    title TEXT,
    score FLOAT,
    reason TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH similar_users AS (
        SELECT user_b as similar_user, similarity
        FROM user_similarity
        WHERE user_a = p_user_id
        ORDER BY similarity DESC
        LIMIT 50
    ),
    watched_by_user AS (
        SELECT DISTINCT media_id 
        FROM user_activity 
        WHERE user_id = p_user_id 
        AND action IN ('play', 'complete')
    ),
    recommendations AS (
        SELECT 
            ua.media_id,
            SUM(su.similarity) as score,
            COUNT(DISTINCT su.similar_user) as recommender_count
        FROM similar_users su
        JOIN user_activity ua ON ua.user_id = su.similar_user
        WHERE ua.action = 'complete'
        AND ua.media_id NOT IN (SELECT media_id FROM watched_by_user)
        GROUP BY ua.media_id
        HAVING COUNT(DISTINCT su.similar_user) >= 2
    )
    SELECT 
        r.media_id,
        m.title,
        r.score * (1 + LN(GREATEST(m.popularity_score, 1))) as final_score,
        'Recommended based on ' || r.recommender_count || ' similar users' as reason
    FROM recommendations r
    JOIN media_items m ON m.id = r.media_id
    ORDER BY final_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PERFORMANCE OPTIMIZATION FUNCTIONS
-- =====================================================

-- Function to update media popularity scores
CREATE OR REPLACE FUNCTION update_popularity_scores()
RETURNS void AS $$
BEGIN
    UPDATE media_items m
    SET popularity_score = subquery.score,
        updated_at = NOW()
    FROM (
        SELECT 
            media_id,
            (
                COUNT(DISTINCT user_id) * 1.0 +
                SUM(CASE WHEN action = 'complete' THEN 2 ELSE 1 END) * 0.5 +
                COUNT(CASE WHEN time > NOW() - INTERVAL '7 days' THEN 1 END) * 2.0
            ) / (EXTRACT(EPOCH FROM (NOW() - MIN(time))) / 86400 + 1) as score
        FROM user_activity
        WHERE time > NOW() - INTERVAL '30 days'
        GROUP BY media_id
    ) as subquery
    WHERE m.id = subquery.media_id;
END;
$$ LANGUAGE plpgsql;

-- Automatic table maintenance
CREATE OR REPLACE FUNCTION auto_vacuum_analyze()
RETURNS void AS $$
DECLARE
    table_name TEXT;
BEGIN
    FOR table_name IN 
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public'
    LOOP
        EXECUTE 'VACUUM ANALYZE ' || table_name;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- QUERY PERFORMANCE MONITORING
-- =====================================================

-- View for slow queries
CREATE OR REPLACE VIEW performance.slow_queries AS
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    stddev_exec_time,
    rows,
    100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY mean_exec_time DESC
LIMIT 50;

-- View for index usage statistics
CREATE OR REPLACE VIEW performance.index_usage AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan;

-- Function to suggest missing indexes
CREATE OR REPLACE FUNCTION performance.suggest_indexes()
RETURNS TABLE (
    table_name TEXT,
    column_name TEXT,
    index_type TEXT,
    estimated_benefit FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.tablename::TEXT,
        a.attname::TEXT,
        CASE 
            WHEN s.n_distinct > 100 THEN 'btree'
            WHEN s.n_distinct <= 100 THEN 'hash'
            ELSE 'btree'
        END as index_type,
        (s.null_frac * 100)::FLOAT as estimated_benefit
    FROM pg_stats s
    JOIN pg_attribute a ON a.attname = s.attname
    JOIN pg_class c ON c.oid = a.attrelid
    JOIN pg_tables t ON t.tablename = c.relname
    WHERE s.schemaname = 'public'
    AND NOT EXISTS (
        SELECT 1 FROM pg_index i
        WHERE i.indrelid = c.oid
        AND a.attnum = ANY(i.indkey)
    )
    AND s.n_distinct > 50
    ORDER BY (s.n_distinct * (1 - s.null_frac)) DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- SCHEDULED MAINTENANCE JOBS
-- =====================================================

-- Create pg_cron extension for scheduled jobs
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Schedule popularity score updates
SELECT cron.schedule('update-popularity', '*/15 * * * *', 'SELECT update_popularity_scores()');

-- Schedule search index refresh
SELECT cron.schedule('refresh-search', '0 */6 * * *', 'REFRESH MATERIALIZED VIEW CONCURRENTLY media_search_index');

-- Schedule similarity matrix update
SELECT cron.schedule('update-similarity', '0 2 * * *', 'REFRESH MATERIALIZED VIEW CONCURRENTLY user_similarity');

-- Schedule statistics update
SELECT cron.schedule('update-stats', '0 */4 * * *', 'ANALYZE');

-- Schedule old partition cleanup
SELECT cron.schedule('cleanup-old-data', '0 3 * * 0', 
    $$DELETE FROM user_activity WHERE time < NOW() - INTERVAL '90 days'$$);