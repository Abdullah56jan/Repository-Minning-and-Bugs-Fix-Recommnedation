
"""
test_api_server.py — Unit tests for the FastAPI server (api/server.py).
 
Rewritten to match the ACTUAL production server, which exposes two
separate endpoints rather than a fused design:
 
  - GET  /health         -> status, index_loaded, codet5_loaded,
                            pairs_indexed, index_size_mb, version
  - POST /recommend      -> BM25 retrieval only. Returns FixRecommendation
                            items: rank, score, fixed_code, buggy_code,
                            commit_message, repo, file_path, pair_id
  - POST /generate-fix   -> CodeT5 generation only. Returns fixed_code, model
  - GET  /               -> root info dict with docs/health/recommend/
                            generate-fix URLs
 
This replaces an earlier test file written against a different, unused
API design (fused /recommend response with fix_source/fix_confidence/
generated_fix fields, and a _run_codet5 module helper). That design does
not exist in api/server.py and running the old file against the live
server produces AttributeError at fixture setup for 23 of its 25 tests
(confirmed via direct pytest execution on 2026-07-03). This file mocks
the module's actual globals (_engine, _codet5_model, _codet5_tokenizer)
instead.
 
Run with: pytest tests/ -v
"""
 
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
 
 
# ── Fixtures ──────────────────────────────────────────────────
 
def make_mock_result(rank=1, score=42.5, repo="apache/kafka"):
    """Builds a mock BugFixResult matching src/retrieval/bm25_engine.py's dataclass shape."""
    result = MagicMock()
    result.rank           = rank
    result.score          = score
    result.fixed_code     = "public void run() { if (obj != null) obj.call(); }"
    result.buggy_code     = "public void run() { obj.call(); }"
    result.commit_message = "fix: null pointer in run()"
    result.repo           = repo
    result.file_path      = "src/main/java/Runner.java"
    result.pair_id        = f"test-pair-{rank}"
    return result
 
 
def make_mock_codet5(fixed_code="public boolean isEqual(String s1, String s2) { return s1.equals(s2); }"):
    """
    Builds mocks for _codet5_model and _codet5_tokenizer that satisfy the
    real generate_fix() code path:
        inputs = _codet5_tokenizer(text, ...).to(device)
        outputs = _codet5_model.generate(**inputs, ...)
        fixed = _codet5_tokenizer.decode(outputs[0], ...).strip()
    """
    mock_tokenizer_output = MagicMock()
    mock_tokenizer_output.to.return_value = {}  # empty dict so **inputs unpacks safely
 
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = mock_tokenizer_output
    mock_tokenizer.decode.return_value = fixed_code
 
    mock_model = MagicMock()
    mock_model.generate.return_value = [MagicMock()]
 
    return mock_model, mock_tokenizer
 
 
@pytest.fixture
def client_full():
    """
    Both BM25 index and CodeT5 model loaded — the normal healthy state.
 
    IMPORTANT: TestClient is deliberately NOT used as a context manager
    ('with TestClient(...) as c:') here. Doing so triggers the app's real
    lifespan startup function, which loads the actual production BM25
    index (1.1GB) and actual CodeT5 model weights from disk on every
    single test — since those real files exist in this repository, that
    would make every test extremely slow (or worse) instead of using the
    mocks below. Skipping the context-manager form means lifespan never
    runs, so the module's _engine/_codet5_model/_codet5_tokenizer globals
    stay at their default None until we set them directly.
    """
    import api.server as server_module
 
    mock_engine = MagicMock()
    mock_engine.is_ready.return_value = True
    mock_engine.stats.return_value = {
        "loaded": True,
        "pairs_indexed": 30000,
        "index_size_mb": 1130.05,
        "k1": 1.5,
        "b": 0.75,
        "index_file": "checkpoints/bm25_index.pkl",
    }
    mock_engine.query.return_value = [
        make_mock_result(rank=1, score=79.19, repo="apache/tomcat"),
        make_mock_result(rank=2, score=45.3, repo="netty/netty"),
    ]
    mock_model, mock_tokenizer = make_mock_codet5()
 
    server_module._engine = mock_engine
    server_module._codet5_model = mock_model
    server_module._codet5_tokenizer = mock_tokenizer
 
    client = TestClient(server_module.app, raise_server_exceptions=True)
    yield client
 
    server_module._engine = None
    server_module._codet5_model = None
    server_module._codet5_tokenizer = None
 
 
@pytest.fixture
def client_no_index():
    """BM25 index not loaded (e.g. server started before 'python main.py --step index')."""
    import api.server as server_module
    mock_model, mock_tokenizer = make_mock_codet5()
 
    server_module._engine = None
    server_module._codet5_model = mock_model
    server_module._codet5_tokenizer = mock_tokenizer
 
    client = TestClient(server_module.app, raise_server_exceptions=True)
    yield client
 
    server_module._engine = None
    server_module._codet5_model = None
    server_module._codet5_tokenizer = None
 
 
@pytest.fixture
def client_no_codet5():
    """BM25 index loaded, but CodeT5 model failed to load (e.g. missing model files)."""
    import api.server as server_module
 
    mock_engine = MagicMock()
    mock_engine.is_ready.return_value = True
    mock_engine.stats.return_value = {
        "loaded": True,
        "pairs_indexed": 30000,
        "index_size_mb": 1130.05,
        "k1": 1.5,
        "b": 0.75,
        "index_file": "checkpoints/bm25_index.pkl",
    }
    mock_engine.query.return_value = [make_mock_result(rank=1, score=51.7)]
 
    server_module._engine = mock_engine
    server_module._codet5_model = None
    server_module._codet5_tokenizer = None
 
    client = TestClient(server_module.app, raise_server_exceptions=True)
    yield client
 
    server_module._engine = None
    server_module._codet5_model = None
    server_module._codet5_tokenizer = None
 
 
# ── Health endpoint ───────────────────────────────────────────
 
class TestHealthEndpoint:
 
    def test_health_returns_ok_when_loaded(self, client_full):
        response = client_full.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["index_loaded"] is True
        assert data["pairs_indexed"] == 30000
 
    def test_health_contains_codet5_loaded_field(self, client_full):
        data = client_full.get("/health").json()
        assert "codet5_loaded" in data
        assert data["codet5_loaded"] is True
 
    def test_health_returns_degraded_when_no_index(self, client_no_index):
        response = client_no_index.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["index_loaded"] is False
        assert data["pairs_indexed"] == 0
 
    def test_health_codet5_loaded_false_when_model_missing(self, client_no_codet5):
        data = client_no_codet5.get("/health").json()
        assert data["codet5_loaded"] is False
 
    def test_health_contains_version(self, client_full):
        assert client_full.get("/health").json()["version"] == "2.0.0"
 
    def test_health_contains_index_size_mb(self, client_full):
        data = client_full.get("/health").json()
        assert data["index_size_mb"] == 1130.05
 
 
# ── /recommend endpoint (BM25 only) ────────────────────────────
 
class TestRecommendEndpoint:
 
    def test_valid_request_returns_200(self, client_full):
        response = client_full.post("/recommend", json={
            "buggy_code": "public void run() { obj.call(); }",
            "top_k": 3,
        })
        assert response.status_code == 200
 
    def test_response_contains_required_top_level_fields(self, client_full):
        data = client_full.post("/recommend", json={
            "buggy_code": "public void run() { obj.call(); }",
        }).json()
        assert "results" in data
        assert "total_results" in data
        assert "query_time_ms" in data
        assert "pairs_indexed" in data
 
    def test_results_have_expected_fields(self, client_full):
        results = client_full.post("/recommend", json={
            "buggy_code": "public void run() { String s = null; s.trim(); }",
        }).json()["results"]
        assert len(results) > 0
        first = results[0]
        for field in ("rank", "score", "fixed_code", "buggy_code",
                      "commit_message", "repo", "file_path", "pair_id"):
            assert field in first
 
    def test_results_ordered_by_rank(self, client_full):
        results = client_full.post("/recommend", json={
            "buggy_code": "public void connect() { socket.connect(); }",
        }).json()["results"]
        ranks = [r["rank"] for r in results]
        assert ranks == sorted(ranks)
 
    def test_pairs_indexed_matches_engine_stats(self, client_full):
        data = client_full.post("/recommend", json={
            "buggy_code": "public void run() {}",
        }).json()
        assert data["pairs_indexed"] == 30000
 
    def test_503_when_no_index(self, client_no_index):
        response = client_no_index.post("/recommend", json={
            "buggy_code": "public void run() { obj.call(); }",
        })
        assert response.status_code == 503
 
    def test_empty_buggy_code_rejected(self, client_full):
        assert client_full.post("/recommend", json={"buggy_code": ""}).status_code == 422
 
    def test_missing_buggy_code_rejected(self, client_full):
        assert client_full.post("/recommend", json={}).status_code == 422
 
    def test_buggy_code_exceeding_max_length_rejected(self, client_full):
        # RecommendRequest caps buggy_code at 50,000 characters
        oversized = "x" * 50_001
        response = client_full.post("/recommend", json={"buggy_code": oversized})
        assert response.status_code == 422
 
    def test_top_k_above_20_rejected(self, client_full):
        assert client_full.post("/recommend", json={
            "buggy_code": "public void run() {}",
            "top_k": 99,
        }).status_code == 422
 
    def test_top_k_zero_rejected(self, client_full):
        assert client_full.post("/recommend", json={
            "buggy_code": "public void run() {}",
            "top_k": 0,
        }).status_code == 422
 
    def test_query_time_is_a_number(self, client_full):
        qt = client_full.post("/recommend", json={
            "buggy_code": "public void run() { return null; }",
        }).json()["query_time_ms"]
        assert isinstance(qt, (int, float)) and qt >= 0
 
 
# ── /generate-fix endpoint (CodeT5 only) ───────────────────────
 
class TestGenerateFixEndpoint:
 
    def test_valid_request_returns_200(self, client_full):
        response = client_full.post("/generate-fix", json={
            "buggy_code": "public boolean isEqual(String s1, String s2) { return s1 == s2; }",
        })
        assert response.status_code == 200
 
    def test_response_contains_fixed_code_and_model(self, client_full):
        data = client_full.post("/generate-fix", json={
            "buggy_code": "public boolean isEqual(String s1, String s2) { return s1 == s2; }",
        }).json()
        assert "fixed_code" in data
        assert "model" in data
        assert data["model"] == "codet5-base"
 
    def test_generated_fix_is_nonempty_string(self, client_full):
        data = client_full.post("/generate-fix", json={
            "buggy_code": "public boolean isEqual(String s1, String s2) { return s1 == s2; }",
        }).json()
        assert isinstance(data["fixed_code"], str)
        assert len(data["fixed_code"]) > 0
 
    def test_503_when_codet5_not_loaded(self, client_no_codet5):
        response = client_no_codet5.post("/generate-fix", json={
            "buggy_code": "public void run() { obj.call(); }",
        })
        assert response.status_code == 503
 
    def test_empty_buggy_code_rejected(self, client_full):
        assert client_full.post("/generate-fix", json={"buggy_code": ""}).status_code == 422
 
    def test_missing_buggy_code_rejected(self, client_full):
        assert client_full.post("/generate-fix", json={}).status_code == 422
 
    def test_buggy_code_exceeding_max_length_rejected(self, client_full):
        # GenerateFixRequest caps buggy_code at 5,000 characters — different
        # limit from /recommend's 50,000, since CodeT5 truncates to 2,000 anyway
        oversized = "x" * 5_001
        response = client_full.post("/generate-fix", json={"buggy_code": oversized})
        assert response.status_code == 422
 
    def test_generation_error_returns_500(self, client_full):
        import api.server as server_module
        server_module._codet5_model.generate.side_effect = RuntimeError("out of memory")
        response = client_full.post("/generate-fix", json={
            "buggy_code": "public void run() {}",
        })
        assert response.status_code == 500
 
 
# ── Root endpoint ─────────────────────────────────────────────
 
class TestRootEndpoint:
 
    def test_root_returns_200(self, client_full):
        assert client_full.get("/").status_code == 200
 
    def test_root_contains_docs_url(self, client_full):
        assert "docs" in client_full.get("/").json()
 
    def test_root_contains_endpoint_references(self, client_full):
        data = client_full.get("/").json()
        assert "recommend" in data
        assert "generate-fix" in data
        assert "health" in data
 