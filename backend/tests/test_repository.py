from unittest.mock import MagicMock

from src.database.client import SupabaseRepository


def _make_repo(data=None):
    """Build a SupabaseRepository with a mocked Supabase client."""
    mock_client = MagicMock()
    mock_client.table.return_value.insert.return_value.execute.return_value.data = [data or {}]
    mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
        data or {}
    ]
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
        data or {}
    ]
    repo = SupabaseRepository(client=mock_client)
    return repo, mock_client


def test_create_evaluation_inserts_row():
    repo, mock_client = _make_repo({"session_id": "sess_1", "user_id": "u1", "status": "PLANNING"})
    result = repo.create_evaluation("sess_1", "u1")
    assert result["session_id"] == "sess_1"
    mock_client.table.assert_called_with("evaluations")


def test_update_evaluation_sets_fields():
    repo, mock_client = _make_repo({"session_id": "sess_1", "status": "EXECUTING"})
    repo.update_evaluation("sess_1", status="EXECUTING", current_step="check_stock")
    mock_client.table.assert_called_with("evaluations")


def test_get_evaluation_returns_row():
    repo, mock_client = _make_repo({"session_id": "sess_1", "status": "AWAITING_APPROVAL"})
    result = repo.get_evaluation("sess_1")
    assert result["session_id"] == "sess_1"


def test_rpc_calls_postgres_function_and_returns_rows():
    mock_client = MagicMock()
    mock_client.rpc.return_value.execute.return_value.data = [
        {"item_id": "IT-XPS-15", "name": "Dell XPS 15 Laptop", "similarity": 0.8}
    ]
    repo = SupabaseRepository(client=mock_client)
    result = repo.rpc("search_items_by_name", {"query": "dell xps", "match_limit": 5})
    assert result == [{"item_id": "IT-XPS-15", "name": "Dell XPS 15 Laptop", "similarity": 0.8}]
    mock_client.rpc.assert_called_with(
        "search_items_by_name", {"query": "dell xps", "match_limit": 5}
    )
