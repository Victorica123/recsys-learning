"""Exercise the public v1 contract with real SQLite constraints and a fake model."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import serve
from api_contract import API_VERSION, openapi_document
from feedback import FeedbackStore
from serving import UnknownUserError


class FakeRecommender:
    empty = False

    def recommend(self, user_id, k, n_candidates):
        if user_id != 1:
            raise UnknownUserError(user_id)
        items = [] if self.empty else [
            {"movie_id": 100 + index, "title": f"Movie {index}",
             "genres": "Drama", "score": 1 - index / 10, "rank": index}
            for index in range(1, min(k, 5) + 1)
        ]
        return {"user_id": user_id, "recommendations": items,
                "n_candidates": len(items), "ranking_policy": "retrieval",
                "score_type": "two_tower_similarity"}


def request(path, payload=None, *, method="POST", app=None):
    """Use ASGI directly; no optional HTTP client or checkpoint is needed."""
    parsed = urlsplit(path)
    body = json.dumps(payload).encode() if payload is not None else b""
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": parsed.path,
        "raw_path": parsed.path.encode(), "query_string": parsed.query.encode(),
        "root_path": "", "headers": [(b"content-type", b"application/json")],
        "server": ("testserver", 80), "client": ("127.0.0.1", 1234),
        "recsys.request_id": "contract-test",
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run((app or serve._starlette_app)(scope, receive, send))
    start = next(message for message in messages if message["type"] == "http.response.start")
    data = b"".join(message.get("body", b"") for message in messages
                    if message["type"] == "http.response.body")
    return start["status"], json.loads(data)


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = FeedbackStore(Path(directory.name) / "events.sqlite3")
        self.model = FakeRecommender()
        for name, value in (("_FEEDBACK", self.store), ("_EXPLORATION_RATE", 0.0)):
            patcher = patch.object(serve, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(serve, "get_recommender", return_value=self.model)
        self.load = patcher.start()
        self.addCleanup(patcher.stop)

    def create(self):
        status, result = request("/v1/recommendations", {"user_id": 1, "k": 5})
        self.assertEqual(status, 200)
        return result

    def test_openapi_is_readable_without_loading_a_model_or_api_key(self):
        protected = serve.ApiKeyMiddleware(serve._starlette_app, "test-key")
        status, document = request("/openapi.json", method="GET", app=protected)
        self.assertEqual(status, 200)
        self.assertEqual(document["info"]["version"], API_VERSION)
        self.load.assert_not_called()
        self.assertEqual(request("/v1/recommendations", {"user_id": 1},
                                 app=protected)[0], 401)
        for route in ("/v1/events/impression", "/v1/events/feedback"):
            operation = document["paths"][route]["post"]
            self.assertTrue(operation["requestBody"]["required"])
            self.assertTrue({"200", "201", "400", "401", "429", "503"}
                            <= set(operation["responses"]))

    def test_non_integer_inputs_fail_before_model_loading(self):
        invalid = [{"user_id": value} for value in (True, 1.9, "1", None, 0)]
        invalid += [{"user_id": 1, "k": value}
                    for value in (True, 2.8, "5", None, 0, 101, float("inf"))]
        invalid += [{}, []]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.assertEqual(request("/v1/recommendations", payload)[0], 400)
        self.load.assert_not_called()
        self.assertEqual(self.store.stats()["recommendations"], 0)

    def test_default_and_unknown_user(self):
        status, result = request("/v1/recommendations", {"user_id": 1})
        self.assertEqual(status, 200)
        schema = openapi_document()["components"]["schemas"]["RecommendationResponse"]
        self.assertTrue(set(schema["required"]) <= result.keys())
        self.assertEqual(self.store.stats()["recommendations"], 1)
        self.assertEqual(request("/v1/recommendations", {"user_id": 999})[0], 404)

    def test_empty_slate_keeps_response_contract_without_creating_a_log(self):
        self.model.empty = True
        result = self.create()
        self.assertIsNone(result["recommendation_id"])
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["model_version"], serve._MODEL_VERSION)
        self.assertEqual(self.store.stats()["recommendations"], 0)

    def test_recommend_impression_feedback_retry_and_conflict(self):
        result = self.create()
        rid = result["recommendation_id"]
        ids = [item["movie_id"] for item in result["recommendations"]]
        event = {"recommendation_id": rid, "movie_id": ids[0],
                 "event_id": "click-1", "event_type": "click"}
        self.assertEqual(request("/v1/events/feedback", event)[0], 400)
        exposure = {"recommendation_id": rid, "movie_ids": ids}
        self.assertEqual(request("/v1/events/impression", exposure)[0], 201)
        status, duplicate_exposure = request("/v1/events/impression", exposure)
        self.assertEqual(status, 200)
        self.assertEqual(duplicate_exposure["inserted"], 0)
        self.assertEqual(request("/v1/events/feedback", event)[0], 201)
        status, duplicate = request("/v1/events/feedback", event)
        self.assertEqual(status, 200)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(request("/v1/events/feedback", {**event, "value": 2})[0], 400)
        counts = self.store.stats()
        self.assertEqual((counts["recommendations"], counts["recommendation_items"],
                          counts["impressions"], counts["feedback_events"]), (1, 5, 5, 1))

    def test_invalid_events_do_not_mutate_the_store(self):
        rid = self.create()["recommendation_id"]
        for ids in ([True], [101.9], ["101"], [], [101] * 101, [10 ** 1000]):
            self.assertEqual(request("/v1/events/impression",
                                     {"recommendation_id": rid, "movie_ids": ids})[0], 400)
        request("/v1/events/impression", {"recommendation_id": rid, "movie_ids": [101]})
        event = {"recommendation_id": rid, "movie_id": 101,
                 "event_type": "click", "event_id": "e1"}
        invalid = [{"event_id": value} for value in (None, "", True, 3)]
        invalid += [{"movie_id": value} for value in (True, 101.9, "101", 10 ** 1000)]
        invalid += [{"value": value} for value in
                    (True, "1", -1, float("inf"), float("nan"), 10 ** 1000)]
        invalid += [{"occurred_at": 123}]
        for fields in invalid:
            with self.subTest(fields=fields):
                self.assertEqual(request("/v1/events/feedback", {**event, **fields})[0], 400)
        self.assertEqual(self.store.stats()["feedback_events"], 0)

    def test_store_failure_returns_503_and_does_not_claim_a_registered_slate(self):
        with patch.object(self.store, "record_recommendation", side_effect=OSError("offline")):
            with self.assertLogs("recsys.feedback", level="ERROR"):
                status, result = request("/v1/recommendations", {"user_id": 1})
        self.assertEqual(status, 503)
        self.assertNotIn("recommendation_id", result)
        self.assertEqual(self.store.stats()["recommendations"], 0)

    def test_legacy_get_remains_usable(self):
        status, result = request("/recommend?user_id=1&k=5", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(len(result["recommendations"]), 5)

    def test_model_unavailable_returns_503(self):
        self.load.return_value = None
        self.assertEqual(request("/v1/recommendations", {"user_id": 1})[0], 503)


if __name__ == "__main__":
    unittest.main()
