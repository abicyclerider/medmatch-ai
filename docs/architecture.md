# MedMatch AI Service Architecture

This document describes the production service architecture for MedMatch AI, enabling integration with PostgreSQL databases and frontend applications via REST + WebSocket APIs.

## Overview

MedMatch can be deployed in two modes:

1. **Library Mode** - Use core matching algorithms directly in Python
2. **Service Mode** - Full production service with database, API, and Docker

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Frontend                                    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ REST + WebSocket
┌───────────────────────────────▼─────────────────────────────────────────┐
│                         API Layer (FastAPI)                              │
│  /api/v1/patients  /api/v1/matching  /api/v1/review  /ws/jobs/{id}      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                       Service Layer                                      │
│  MatchingService  │  GoldenRecordService  │  ReviewService  │  BatchService │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                    Repository Layer (Unit of Work)                       │
│  PatientRepository │ MatchRepository │ GoldenRecordRepository │ ...     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────┐                         ┌───────────────────┐
│   PostgreSQL      │                         │  File Adapter     │
│   (Production)    │                         │  (Legacy/Dev)     │
└───────────────────┘                         └───────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    Core Matching (UNCHANGED)                             │
│  PatientMatcher  │  Blocking  │  Rules  │  Scoring  │  AI (Ollama)      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Package Structure

MedMatch uses a monorepo with two installable packages:

```
medmatch-ai/
├── src/
│   ├── medmatch/               # Package 1: Core library
│   │   ├── matching/           # Entity resolution algorithms
│   │   ├── data/               # Data models
│   │   └── evaluation/         # Metrics
│   │
│   └── medmatch_server/        # Package 2: Server
│       ├── persistence/        # Database layer
│       │   ├── base.py         # Abstract interfaces
│       │   ├── models.py       # SQLAlchemy ORM
│       │   ├── repositories/   # PostgreSQL implementations
│       │   ├── unit_of_work.py # Transaction management
│       │   └── migrations/     # Alembic migrations
│       │
│       ├── service/            # Business logic
│       │   ├── matching_service.py
│       │   ├── golden_record_service.py
│       │   ├── review_service.py
│       │   └── batch_service.py
│       │
│       └── api/                # REST + WebSocket
│           ├── main.py         # FastAPI app
│           ├── routers/        # Endpoint modules
│           ├── schemas/        # Request/response models
│           └── websocket/      # Real-time handlers
│
└── docker/                     # Deployment
    ├── Dockerfile
    ├── Dockerfile.worker
    └── docker-compose.yml
```

## Installation

### Core Library Only

For projects that only need matching algorithms:

```bash
pip install medmatch
```

```python
from medmatch.matching import PatientMatcher
matcher = PatientMatcher()
result = matcher.match_pair(record1, record2)
```

### Full Service

For production deployment with database and API:

```bash
pip install medmatch-server

# Or with Docker
docker-compose up
```

## Database Schema

### Tables

#### source_records

Incoming patient data from various systems:

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| external_id | VARCHAR(100) | Original ID from source system |
| source_system | VARCHAR(50) | Hospital A, Clinic B, etc. |
| demographics | JSONB | Name, DOB, gender, contact info |
| medical_data | JSONB | Conditions, medications (optional) |
| blocking_keys | JSONB | Pre-computed for fast lookup |
| received_at | TIMESTAMP | When record was received |

#### golden_records

Master Patient Index - canonical patient identities:

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| golden_id | VARCHAR(50) | Human-readable: "GR-000001" |
| merged_demographics | JSONB | Best values from linked records |
| merged_medical | JSONB | Combined medical history |
| status | VARCHAR(20) | active, merged, retired |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Last update timestamp |

#### record_links

Links between source records and golden records:

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| source_record_id | BIGINT FK | Source record |
| golden_record_id | BIGINT FK | Golden record |
| link_type | VARCHAR(20) | auto_match, manual_match, manual_split |
| confidence | DECIMAL(4,3) | Match confidence 0.000-1.000 |
| match_result_id | BIGINT FK | Reference to match result |

#### match_results

Every comparison stored for audit:

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| record_1_id | BIGINT FK | First record |
| record_2_id | BIGINT FK | Second record |
| is_match | BOOLEAN | Match decision |
| confidence | DECIMAL(4,3) | Confidence score |
| match_type | VARCHAR(20) | exact, probable, possible, no_match |
| evidence | JSONB | Feature scores, rules triggered |
| stage | VARCHAR(20) | rules, scoring, ai |
| explanation | TEXT | Human-readable explanation |
| medical_similarity | DECIMAL(4,3) | AI similarity score (nullable) |
| ai_reasoning | TEXT | AI explanation (nullable) |

#### review_queue

Human-in-the-loop for ambiguous matches:

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| match_result_id | BIGINT FK | Match to review |
| priority | INTEGER | 1=highest, 10=lowest |
| status | VARCHAR(20) | pending, assigned, resolved |
| assigned_to | VARCHAR(100) | User email |
| resolution | VARCHAR(20) | confirmed_match, confirmed_non_match, defer |
| resolver_notes | TEXT | Human notes |

#### batch_jobs

Async job tracking:

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| job_id | UUID | Unique job identifier |
| job_type | VARCHAR(50) | full_match, incremental |
| status | VARCHAR(20) | pending, running, completed, failed |
| total_records | INTEGER | Total to process |
| processed_records | INTEGER | Completed count |
| matched_pairs | INTEGER | Matches found |
| config | JSONB | Matcher configuration |
| error_message | TEXT | Error details (if failed) |

## REST API Endpoints

### Patients

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/patients` | Add new patient record |
| GET | `/api/v1/patients/{id}` | Get patient by ID |
| GET | `/api/v1/patients` | List patients (paginated) |

### Matching

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/matching/lookup` | Real-time single record matching |
| POST | `/api/v1/matching/batch` | Start batch job (returns job_id) |
| GET | `/api/v1/matching/batch/{job_id}` | Get batch job status |
| GET | `/api/v1/matching/results/{id}` | Get match result details |

### Golden Records

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/golden-records/{id}` | Get golden record |
| GET | `/api/v1/golden-records/{id}/sources` | Get linked source records |
| POST | `/api/v1/golden-records/merge` | Merge multiple golden records |
| POST | `/api/v1/golden-records/split` | Split golden record |

### Review Queue

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/review/pending` | Get pending reviews |
| POST | `/api/v1/review/{id}/assign` | Assign to user |
| POST | `/api/v1/review/{id}/resolve` | Submit resolution |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/jobs/{job_id}` | Subscribe to batch job progress |
| `/ws/events` | Subscribe to all matching events |

## Single Record Lookup Flow

```python
POST /api/v1/matching/lookup
{
    "demographics": {
        "name_first": "John",
        "name_last": "Smith",
        "date_of_birth": "1980-01-15",
        "gender": "M",
        "phone": "555-0100"
    },
    "medical_data": {
        "conditions": ["Type 2 Diabetes", "Hypertension"],
        "medications": ["Metformin", "Lisinopril"]
    },
    "source_system": "HospitalA"
}
```

Response:

```python
{
    "decision": "auto_match",  # or "review_required" or "new_identity"
    "source_record_id": 12345,
    "golden_record_id": "GR-000042",
    "confidence": 0.94,
    "explanation": "Exact name and DOB match with existing patient. Medical history confirms (T2DM, HTN)."
}
```

### Decision Logic

| Confidence | Decision | Action |
|------------|----------|--------|
| >= 0.90 | auto_match | Link to existing golden record |
| 0.65 - 0.90 | review_required | Add to human review queue |
| < 0.65 | new_identity | Create new golden record |

## Docker Deployment

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: medmatch
      POSTGRES_USER: medmatch
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"

  medmatch-api:
    build: .
    environment:
      DATABASE_URL: postgresql://medmatch:${POSTGRES_PASSWORD}@postgres:5432/medmatch
      OLLAMA_BASE_URL: http://ollama:11434
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - ollama

  medmatch-worker:
    build:
      dockerfile: Dockerfile.worker
    environment:
      DATABASE_URL: postgresql://medmatch:${POSTGRES_PASSWORD}@postgres:5432/medmatch
      OLLAMA_BASE_URL: http://ollama:11434
    depends_on:
      - postgres
      - ollama
    deploy:
      replicas: 2

volumes:
  postgres_data:
  ollama_data:
```

### Quick Start

```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Single record lookup
curl -X POST http://localhost:8000/api/v1/matching/lookup \
  -H "Content-Type: application/json" \
  -d '{"demographics": {"name_first": "John", "name_last": "Smith", "date_of_birth": "1980-01-15"}}'
```

## Backward Compatibility

Existing code continues to work without modification:

```python
# Mode 1: Library mode (no DATABASE_URL)
from medmatch.matching import PatientMatcher
matcher = PatientMatcher()
result = matcher.match_pair(record1, record2)

# Mode 2: Service mode (DATABASE_URL set)
# Full API + PostgreSQL available
```

The CLI also works in both modes:

```bash
# File-based (existing behavior)
python scripts/run_matcher.py --demographics data.csv

# Service mode (new)
docker-compose up
curl http://localhost:8000/api/v1/matching/lookup -d '{...}'
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | PostgreSQL | Complex queries, JSONB, healthcare standard |
| ORM | SQLAlchemy 2.0 | Mature, async support, Alembic migrations |
| API | FastAPI | Async, WebSocket native, auto OpenAPI docs |
| Pattern | Repository + UoW | Testable, swappable backends, transaction safety |
| AI | Ollama + MedGemma | HIPAA-compliant local inference |

## Success Criteria

- [ ] Existing CLI works unchanged
- [ ] Single record lookup < 2 seconds
- [ ] Batch job 10K records in < 30 minutes
- [ ] WebSocket delivers real-time progress
- [ ] Review queue workflow complete
- [ ] Golden record merge/split operations work
- [ ] Docker deployment runs all services
- [ ] 90%+ test coverage on new code

## Related Documentation

- [Matching Module README](../src/medmatch/matching/README.md) - Core algorithms
- [Ollama Setup Guide](ollama_setup.md) - Local MedGemma deployment
- [Quick Start Guide](quickstart.md) - Getting started
- [Plan File](../.claude/plans/gentle-booping-treehouse.md) - Detailed implementation plan
