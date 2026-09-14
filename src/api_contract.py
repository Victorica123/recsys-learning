"""Dependency-free OpenAPI contract for the content recommendation HTTP API."""

API_VERSION = "1.1.0"


def _ref(name: str) -> dict:
    return {"$ref": f"#/components/schemas/{name}"}


def _body(name: str) -> dict:
    return {"required": True,
            "content": {"application/json": {"schema": _ref(name)}}}


def _response(name: str, description: str) -> dict:
    return {"description": description,
            "content": {"application/json": {"schema": _ref(name)}}}


def _errors(*statuses: str) -> dict:
    return {status: {"$ref": f"#/components/responses/Error{status}"}
            for status in statuses}


def openapi_document() -> dict:
    identifier = {"type": "string", "minLength": 1, "maxLength": 128}
    timestamp = {"type": ["string", "null"],
                 "description": "Optional event timestamp; clients should use UTC ISO 8601."}
    positive_int = {"type": "integer", "minimum": 1}
    schemas = {
        "Error": {"type": "object", "required": ["error"],
                  "properties": {"error": {"type": "string"}}},
        "RecommendationRequest": {
            "type": "object", "required": ["user_id"],
            "properties": {
                "user_id": positive_int,
                "k": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            },
            "example": {"user_id": 1, "k": 5},
        },
        "RecommendationItem": {
            "type": "object", "required": ["movie_id", "title", "score", "rank"],
            "properties": {
                "movie_id": positive_int, "title": {"type": "string"},
                "genres": {"type": "string"}, "score": {"type": "number"},
                "rank": positive_int,
            },
        },
        "RecommendationResponse": {
            "type": "object",
            "required": ["recommendation_id", "request_id", "user_id",
                         "model_version", "policy_name", "ranking_policy",
                         "score_type", "recommendations"],
            "properties": {
                "recommendation_id": {
                    "type": ["string", "null"],
                    "description": "Null only for an empty slate, which creates no log record."},
                "request_id": {"type": ["string", "null"]},
                "user_id": positive_int,
                "model_version": {"type": "string"}, "policy_name": {"type": "string"},
                "ranking_policy": {"type": "string", "enum": ["retrieval", "deepfm"]},
                "score_type": {"type": "string"},
                "exploration_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "n_candidates": {"type": "integer", "minimum": 0},
                "recommendations": {"type": "array", "items": _ref("RecommendationItem")},
            },
        },
        "ImpressionRequest": {
            "type": "object", "required": ["recommendation_id"],
            "properties": {
                "recommendation_id": identifier,
                "movie_ids": {
                    "type": ["array", "null"], "items": positive_int,
                    "minItems": 1, "maxItems": 100,
                    "description": "Items actually displayed; omission/null marks the whole served slate."},
                "impressed_at": timestamp,
            },
        },
        "ImpressionResponse": {
            "type": "object",
            "required": ["recommendation_id", "requested", "inserted", "duplicates"],
            "properties": {
                "recommendation_id": identifier,
                **{key: {"type": "integer", "minimum": 0}
                   for key in ("requested", "inserted", "duplicates")},
            },
        },
        "FeedbackRequest": {
            "type": "object",
            "required": ["recommendation_id", "movie_id", "event_type", "event_id"],
            "properties": {
                "recommendation_id": identifier, "movie_id": positive_int,
                "event_type": {"type": "string",
                               "enum": ["click", "like", "dislike", "skip", "dwell_seconds"]},
                "event_id": {**identifier,
                             "description": "Client-generated once per event; reuse on retries."},
                "value": {"type": ["number", "null"], "minimum": 0},
                "occurred_at": timestamp,
            },
        },
        "FeedbackResponse": {
            "type": "object", "required": ["event_id", "inserted", "duplicate"],
            "properties": {
                "event_id": identifier, "inserted": {"type": "boolean"},
                "duplicate": {"type": "boolean"},
            },
        },
    }
    event_errors = _errors("400", "401", "429", "503")
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Content Recommendation API", "version": API_VERSION,
            "description": (
                "Non-empty recommendations are durably registered before returning. "
                "Record actual impressions before feedback. X-API-Key is required "
                "only when RECSYS_API_KEY is configured on the server."
            ),
        },
        "security": [{"ApiKey": []}, {}],
        "paths": {
            "/v1/recommendations": {"post": {
                "operationId": "createRecommendation",
                "summary": "Create a recommendation slate and register non-empty candidates",
                "parameters": [{
                    "name": "X-Actor-ID", "in": "header", "required": False,
                    "description": "Optional anonymous actor; use a demo/load prefix for scripted traffic.",
                    "schema": identifier,
                }],
                "requestBody": _body("RecommendationRequest"),
                "responses": {
                    "200": _response("RecommendationResponse", "Recommendation slate"),
                    **_errors("400", "401", "404", "429", "503"),
                },
            }},
            "/v1/events/impression": {"post": {
                "operationId": "recordImpressions",
                "summary": "Record actually displayed items; duplicate items are a no-op",
                "requestBody": _body("ImpressionRequest"),
                "responses": {
                    "200": _response("ImpressionResponse", "All requested items already recorded"),
                    "201": _response("ImpressionResponse", "At least one new impression recorded"),
                    **event_errors,
                },
            }},
            "/v1/events/feedback": {"post": {
                "operationId": "recordFeedback",
                "summary": "Record feedback with a stable event_id and a matching impression",
                "description": "Same event_id and content is a no-op; conflicting reuse returns 400.",
                "requestBody": _body("FeedbackRequest"),
                "responses": {
                    "200": _response("FeedbackResponse", "Same event already recorded"),
                    "201": _response("FeedbackResponse", "New feedback recorded"),
                    **event_errors,
                },
            }},
            "/live": {"get": {"security": [], "responses": {"200": {"description": "Alive"}}}},
            "/ready": {"get": {"security": [], "responses": {
                "200": {"description": "Ready"}, "503": {"description": "Not ready"}}}},
        },
        "components": {
            "schemas": schemas,
            "securitySchemes": {
                "ApiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}},
            "responses": {
                f"Error{status}": _response("Error", description)
                for status, description in {
                    "400": "Invalid request or conflicting event",
                    "401": "Missing or invalid API key",
                    "404": "Unknown user",
                    "429": "Rate or admission limit exceeded; honor Retry-After",
                    "503": "Model or feedback store unavailable",
                }.items()
            },
        },
    }
