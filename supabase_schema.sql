-- Face Recognition Database Schema for Supabase
-- This schema supports storing face embeddings and person details

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create persons table
CREATE TABLE IF NOT EXISTS persons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create face_embeddings table with vector support
CREATE TABLE IF NOT EXISTS face_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding VECTOR(128), -- 128-dimensional face embedding
    image_url TEXT,
    confidence_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS face_embeddings_embedding_idx 
ON face_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for person lookup
CREATE INDEX IF NOT EXISTS face_embeddings_person_id_idx 
ON face_embeddings(person_id);

-- Create index for email lookup
CREATE INDEX IF NOT EXISTS persons_email_idx 
ON persons(email);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at
CREATE TRIGGER update_persons_updated_at 
    BEFORE UPDATE ON persons 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_face_embeddings_updated_at 
    BEFORE UPDATE ON face_embeddings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for face similarity search
CREATE OR REPLACE FUNCTION find_similar_faces(
    query_embedding VECTOR(128),
    similarity_threshold FLOAT DEFAULT 0.6,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    person_id UUID,
    person_name VARCHAR(255),
    person_email VARCHAR(255),
    similarity_score FLOAT,
    embedding_id UUID,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.id as person_id,
        p.name as person_name,
        p.email as person_email,
        1 - (fe.embedding <=> query_embedding) as similarity_score,
        fe.id as embedding_id,
        fe.created_at
    FROM face_embeddings fe
    JOIN persons p ON fe.person_id = p.id
    WHERE 1 - (fe.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY fe.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Create function to get person with best match
CREATE OR REPLACE FUNCTION find_best_face_match(
    query_embedding VECTOR(128),
    similarity_threshold FLOAT DEFAULT 0.6
)
RETURNS TABLE (
    person_id UUID,
    person_name VARCHAR(255),
    person_email VARCHAR(255),
    person_phone VARCHAR(50),
    similarity_score FLOAT,
    embedding_id UUID,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.id as person_id,
        p.name as person_name,
        p.email as person_email,
        p.phone as person_phone,
        1 - (fe.embedding <=> query_embedding) as similarity_score,
        fe.id as embedding_id,
        fe.created_at
    FROM face_embeddings fe
    JOIN persons p ON fe.person_id = p.id
    WHERE 1 - (fe.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY fe.embedding <=> query_embedding
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Insert sample data (optional)
INSERT INTO persons (name, email, phone) VALUES 
('John Doe', 'john.doe@example.com', '+1234567890'),
('Jane Smith', 'jane.smith@example.com', '+0987654321')
ON CONFLICT (email) DO NOTHING;

-- Create RLS (Row Level Security) policies
ALTER TABLE persons ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_embeddings ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users (adjust as needed)
CREATE POLICY "Allow all operations for authenticated users" ON persons
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON face_embeddings
    FOR ALL USING (auth.role() = 'authenticated');

-- Allow read access for anonymous users (for public face recognition)
CREATE POLICY "Allow read access for anonymous users" ON persons
    FOR SELECT USING (true);

CREATE POLICY "Allow read access for anonymous users" ON face_embeddings
    FOR SELECT USING (true);
