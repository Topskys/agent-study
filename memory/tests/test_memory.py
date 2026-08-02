import os
import tempfile
import time
from datetime import datetime, timedelta

import pytest

from src.core.memory_cleaner import MemoryCleaner
from src.core.memory_filter import MemoryFilter
from src.core.memory_governance import MemoryGovernance
from src.core.memory_scorer import MemoryScorer
from src.core.memory_system import MemorySystem
from src.core.memory_versioner import MemoryVersioner
from src.core.session_memory import SessionMemory
from src.core.user_memory import UserMemory
from src.core.working_memory import WorkingMemory
from src.models.event import Event
from src.models.graph import GraphEdge, GraphNode
from src.models.memory_item import MemoryItem, MemoryType
from src.models.user_profile import UserProfile
from src.models.version import MemoryVersion
from src.stores.event_store import EventStreamStore
from src.stores.graph_store import GraphStore
from src.stores.kv_store import KeyValueStore
from src.stores.vector_store import VectorStore
from src.utils.embeddings import cosine_similarity, generate_embedding
from src.utils.id_generator import generate_id


# ===== Fixtures =====


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def vector_store(db_path):
    return VectorStore(db_path)


@pytest.fixture
def graph_store():
    return GraphStore()


@pytest.fixture
def kv_store():
    return KeyValueStore()


@pytest.fixture
def event_store():
    return EventStreamStore()


@pytest.fixture
def memory_system(db_path):
    return MemorySystem(db_path)


# ===== 1. Model Tests =====


class TestMemoryItemModel:
    def test_create_memory_item(self):
        item = MemoryItem(memory_id="m1", content="test memory")
        assert item.memory_id == "m1"
        assert item.content == "test memory"
        assert item.memory_type == MemoryType.WORKING
        assert item.created_at is not None
        assert item.updated_at is not None

    def test_memory_type_enum(self):
        assert MemoryType.WORKING.value == "working"
        assert MemoryType.SESSION.value == "session"
        assert MemoryType.LONG_TERM.value == "long_term"


class TestUserProfileModel:
    def test_create_profile(self):
        profile = UserProfile(user_id="u1")
        assert profile.user_id == "u1"
        assert profile.base_info == {}
        assert profile.lock_keys == []

    def test_profile_with_data(self):
        profile = UserProfile(
            user_id="u1",
            base_info={"name": "Alice"},
            preferences={"theme": "dark"},
            scene_profiles={"work": {"mode": "focus"}},
            lock_keys=["name"],
        )
        assert profile.base_info["name"] == "Alice"
        assert "name" in profile.lock_keys


class TestGraphModel:
    def test_graph_node(self):
        node = GraphNode(node_id="n1", label="Person", properties={"name": "Bob"})
        assert node.node_id == "n1"
        assert node.label == "Person"

    def test_graph_edge(self):
        edge = GraphEdge(
            edge_id="e1", from_node_id="n1", to_node_id="n2", relationship="knows"
        )
        assert edge.from_node_id == "n1"
        assert edge.to_node_id == "n2"


class TestEventModel:
    def test_create_event(self):
        event = Event(event_id="ev1", event_type="test")
        assert event.event_id == "ev1"
        assert event.timestamp is not None


class TestVersionModel:
    def test_create_version(self):
        v = MemoryVersion(version_id="v1", memory_id="m1", content="original")
        assert v.memory_id == "m1"
        assert v.content == "original"


# ===== 2. Store Tests =====


class TestVectorStore:
    def test_insert_and_get_memory(self, vector_store):
        item = MemoryItem(memory_id="m1", content="hello", user_id="u1")
        vector_store.insert_memory(item)
        retrieved = vector_store.get_memory("m1")
        assert retrieved is not None
        assert retrieved.content == "hello"

    def test_delete_memory(self, vector_store):
        item = MemoryItem(memory_id="m1", content="hello", user_id="u1")
        vector_store.insert_memory(item)
        vector_store.delete_memory("m1")
        assert vector_store.get_memory("m1") is None

    def test_search_memories(self, vector_store):
        for i in range(3):
            item = MemoryItem(memory_id=f"m{i}", content=f"memory {i}", user_id="u1")
            vector_store.insert_memory(item)
        results = vector_store.search_memories("u1", [0.1] * 128, top_k=2)
        assert len(results) <= 2

    def test_user_profile_crud(self, vector_store):
        profile = UserProfile(user_id="u1", base_info={"name": "Alice"})
        vector_store.insert_user_profile(profile)
        retrieved = vector_store.get_user_profile("u1")
        assert retrieved is not None
        assert retrieved.base_info["name"] == "Alice"

    def test_query_by_time_range(self, vector_store):
        now = datetime.now()
        item = MemoryItem(
            memory_id="m1", content="test", user_id="u1", created_at=now, updated_at=now
        )
        vector_store.insert_memory(item)
        results = vector_store.query_by_time_range(
            "u1", now - timedelta(hours=1), now + timedelta(hours=1)
        )
        assert len(results) == 1


class TestGraphStore:
    def test_add_and_query_node(self, graph_store):
        node = GraphNode(node_id="n1", label="Person")
        graph_store.add_node(node)
        results = graph_store.query_nodes(label="Person")
        assert len(results) == 1
        assert results[0].node_id == "n1"

    def test_add_edge(self, graph_store):
        n1 = GraphNode(node_id="n1", label="Person")
        n2 = GraphNode(node_id="n2", label="Person")
        graph_store.add_node(n1)
        graph_store.add_node(n2)
        edge = GraphEdge(
            edge_id="e1", from_node_id="n1", to_node_id="n2", relationship="knows"
        )
        graph_store.add_edge(edge)
        edges = graph_store.get_edges("n1")
        assert len(edges) == 1

    def test_find_path(self, graph_store):
        for i in range(3):
            graph_store.add_node(GraphNode(node_id=f"n{i}", label="Node"))
        graph_store.add_edge(GraphEdge("e1", "n0", "n1", "connected"))
        graph_store.add_edge(GraphEdge("e2", "n1", "n2", "connected"))
        paths = graph_store.find_path("n0", "n2")
        assert len(paths) >= 1


class TestKeyValueStore:
    def test_put_get_delete(self, kv_store):
        kv_store.put("key1", "value1")
        assert kv_store.get("key1") == "value1"
        kv_store.delete("key1")
        assert kv_store.get("key1") is None

    def test_ttl_expiry(self, kv_store):
        kv_store.put("key1", "value1", ttl=0.1)
        assert kv_store.get("key1") == "value1"
        time.sleep(0.15)
        assert kv_store.get("key1") is None

    def test_keys_pattern(self, kv_store):
        kv_store.put("user:1", "alice")
        kv_store.put("user:2", "bob")
        kv_store.put("config:1", "dark")
        keys = kv_store.keys(pattern="user:")
        assert len(keys) == 2


class TestEventStreamStore:
    def test_add_and_query(self, event_store):
        event_store.add_event(Event(event_id="e1", event_type="login"))
        event_store.add_event(Event(event_id="e2", event_type="logout"))
        events = event_store.query_events(event_type="login")
        assert len(events) == 1

    def test_count_by_type(self, event_store):
        event_store.add_event(Event(event_id="e1", event_type="login"))
        event_store.add_event(Event(event_id="e2", event_type="login"))
        assert event_store.count_by_type("login") == 2


# ===== 3. Governance Tests =====


class TestMemoryFilter:
    def test_filter_allowed_type(self):
        mf = MemoryFilter(allowed_types=[MemoryType.LONG_TERM])
        item = MemoryItem(memory_id="m1", content="ok", memory_type=MemoryType.WORKING)
        assert mf.filter_item(item) is False

    def test_filter_blocked_keyword(self):
        mf = MemoryFilter(blocked_keywords=["bad", "spam"])
        item = MemoryItem(memory_id="m1", content="this is bad content")
        assert mf.filter_item(item) is False

    def test_filter_valid_item(self):
        mf = MemoryFilter()
        item = MemoryItem(
            memory_id="m1", content="good content", memory_type=MemoryType.LONG_TERM
        )
        assert mf.filter_item(item) is True


class TestMemoryScorer:
    def test_score_default(self):
        scorer = MemoryScorer()
        item = MemoryItem(memory_id="m1", content="test", metadata={"importance": 0.8})
        score = scorer.score(item)
        assert 0 <= score <= 1

    def test_score_high_importance(self):
        scorer = MemoryScorer(
            importance_weight=1.0, stability_weight=0, reuse_weight=0, recency_weight=0
        )
        item = MemoryItem(memory_id="m1", content="test", metadata={"importance": 1.0})
        assert scorer.score(item) == 1.0


class TestMemoryCleaner:
    def test_clean_duplicates(self):
        cleaner = MemoryCleaner()
        items = [
            MemoryItem(memory_id="m1", content="hello", user_id="u1", score=0.9),
            MemoryItem(memory_id="m2", content="hello", user_id="u1", score=0.5),
            MemoryItem(memory_id="m3", content="world", user_id="u1", score=0.7),
        ]
        cleaned = cleaner.clean_duplicates(items)
        assert len(cleaned) == 2

    def test_clean_low_score(self):
        cleaner = MemoryCleaner()
        items = [
            MemoryItem(memory_id="m1", content="a", score=0.9),
            MemoryItem(memory_id="m2", content="b", score=0.1),
        ]
        cleaned = cleaner.clean_low_score(items, threshold=0.5)
        assert len(cleaned) == 1

    def test_enforce_capacity(self):
        cleaner = MemoryCleaner()
        items = [
            MemoryItem(memory_id=f"m{i}", content=f"item {i}", score=i / 10)
            for i in range(10)
        ]
        capped = cleaner.enforce_capacity(items, 3)
        assert len(capped) == 3


class TestMemoryVersioner:
    def test_create_and_rollback(self):
        versioner = MemoryVersioner()
        item = MemoryItem(memory_id="m1", content="original")
        versioner.create_version(item)
        item.content = "modified"
        version = versioner.create_version(item)
        assert len(versioner.get_versions("m1")) == 2

        result = versioner.rollback(item, version.version_id)
        assert result is not None
        assert result.content == "modified"

        versions = versioner.versions["m1"]
        result = versioner.rollback(item, versions[0].version_id)
        assert result is not None
        assert result.content == "original"

    def test_diff(self):
        versioner = MemoryVersioner()
        item = MemoryItem(memory_id="m1", content="v1")
        v1 = versioner.create_version(item)
        item.content = "v2"
        v2 = versioner.create_version(item)
        diff = versioner.diff("m1", v1.version_id, v2.version_id)
        assert diff is not None
        assert diff["old"] == "v1"
        assert diff["new"] == "v2"


class TestMemoryGovernance:
    def test_should_enter_long_term_above_threshold(self, db_path):
        vs = VectorStore(db_path)
        gov = MemoryGovernance(vector_store=vs, long_term_threshold=0.5)
        item = MemoryItem(
            memory_id="m1", content="important info", metadata={"importance": 0.9}
        )
        assert gov.should_enter_long_term(item) is True

    def test_should_enter_long_term_below_threshold(self, db_path):
        vs = VectorStore(db_path)
        gov = MemoryGovernance(vector_store=vs, long_term_threshold=0.8)
        item = MemoryItem(
            memory_id="m1", content="trivial", metadata={"importance": 0.2}
        )
        assert gov.should_enter_long_term(item) is False

    def test_detect_conflicts(self, db_path):
        vs = VectorStore(db_path)
        gov = MemoryGovernance(vector_store=vs)
        items = [
            MemoryItem(
                memory_id="m1", content="用户爱吃辣", memory_type=MemoryType.LONG_TERM
            ),
            MemoryItem(
                memory_id="m2", content="用户忌辣", memory_type=MemoryType.LONG_TERM
            ),
        ]
        conflicts = gov.detect_conflicts(items)
        assert len(conflicts) >= 1


# ===== 4. Core Memory Tests =====


class TestWorkingMemory:
    def test_add_and_trim(self):
        wm = WorkingMemory(max_tokens=50)
        wm.add_item(MemoryItem(memory_id="m1", content="a" * 30))
        wm.add_item(MemoryItem(memory_id="m2", content="b" * 30))
        assert wm.current_tokens() <= 50

    def test_clear(self):
        wm = WorkingMemory()
        wm.add_item(MemoryItem(memory_id="m1", content="test"))
        wm.clear()
        assert wm.size() == 0


class TestSessionMemory:
    def test_add_and_end(self):
        sm = SessionMemory("session_1")
        sm.add_item(MemoryItem(memory_id="m1", content="hello"))
        sm.add_item(MemoryItem(memory_id="m2", content="world"))
        assert sm.size() == 2

        items = sm.end_session()
        assert len(items) == 2
        assert sm.size() == 0

    def test_session_context(self):
        sm = SessionMemory("session_1")
        sm.update_context("topic", "python")
        assert sm.get_session_context()["topic"] == "python"


class TestUserMemory:
    def test_add_working_memory(self, db_path):
        vs = VectorStore(db_path)
        gov = MemoryGovernance(vector_store=vs)
        um = UserMemory("u1", vs, gov)
        item = um.add_working_memory("test working")
        assert item.content == "test working"
        assert item.memory_type == MemoryType.WORKING

    def test_session_flow(self, db_path):
        vs = VectorStore(db_path)
        gov = MemoryGovernance(vector_store=vs)
        um = UserMemory("u1", vs, gov)
        um.start_session("s1")
        item = um.add_session_memory("session msg")
        assert item is not None
        assert item.content == "session msg"

        end_items = um.end_session()
        assert len(end_items) == 1

    def test_add_long_term_memory(self, db_path):
        vs = VectorStore(db_path)
        gov = MemoryGovernance(vector_store=vs, long_term_threshold=0.1)
        um = UserMemory("u1", vs, gov)
        item = um.add_long_term_memory("important long term", {"importance": 0.9})
        assert item is not None

    def test_retrieve_relevant_memories(self, db_path):
        vs = VectorStore(db_path)
        gov = MemoryGovernance(vector_store=vs, long_term_threshold=0.1)
        um = UserMemory("u1", vs, gov)
        um.add_working_memory("working item")
        um.start_session("s1")
        um.add_session_memory("session item")
        um.end_session()
        um.add_long_term_memory("long term item", {"importance": 0.9})
        results = um.retrieve_relevant_memories("item", top_k=5)
        assert len(results) >= 1

    def test_update_profile(self, db_path):
        vs = VectorStore(db_path)
        gov = MemoryGovernance(vector_store=vs)
        um = UserMemory("u1", vs, gov)
        um.update_profile({"base_info": {"name": "Alice"}})
        assert um.profile is not None
        assert um.profile.base_info["name"] == "Alice"


class TestMemorySystem:
    def test_get_user_memory(self, memory_system):
        um = memory_system.get_user_memory("u1")
        assert um.user_id == "u1"
        assert memory_system.get_user_memory("u1") is um

    def test_remove_user_memory(self, memory_system):
        memory_system.get_user_memory("u1")
        memory_system.remove_user_memory("u1")
        assert "u1" not in memory_system.user_memories

    def test_multiple_users(self, memory_system):
        um1 = memory_system.get_user_memory("u1")
        um2 = memory_system.get_user_memory("u2")
        assert um1 is not um2

    def test_full_workflow(self, memory_system):
        um = memory_system.get_user_memory("u1")
        um.add_working_memory("test")
        um.start_session("s1")
        um.add_session_memory("session test")
        um.end_session()
        um.add_long_term_memory("lt test", {"importance": 0.9})
        results = um.retrieve_relevant_memories("test")
        assert len(results) >= 1
