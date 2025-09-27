-- Migration script for face recognition system
-- Run this in your Supabase SQL editor

-- Step 1: Enable vector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "vector";

-- Step 2: Create the tables
\i supabase_schema.sql

-- Step 3: Verify the setup
SELECT 
    schemaname,
    tablename,
    tableowner
FROM pg_tables 
WHERE tablename IN ('persons', 'face_embeddings')
ORDER BY tablename;

-- Step 4: Test the similarity search function
-- This will return empty results until you have data
SELECT * FROM find_similar_faces(
    array_fill(0.1, ARRAY[128])::vector(128),
    0.6,
    5
);

-- Step 5: Show table structure
\d persons;
\d face_embeddings;
