"""Machine-readable HTTP contract for the promoted serving surface.

Keep this dependency-free so CI can validate the public contract without model
artifacts.  The legacy unversioned routes remain available for the demo, while
new integrations should use ``/v1``.
"""

API_VERSION = "1.0.0"


def openapi_document() -> dict:
    recommendation = {
        "type": "object",
        "required": ["movie_id", "title", "score"],
        "properties": {
            "movie_id": {"type": "integer"},
            "title": {"type": "string"},
            "genres": {"type": "string"},
            "score": {"type": "number"},
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Content Recommendation API",
            "version": API_VERSION,
            "description": (
                "Versioned contract for recommendation delivery and the "
                "impression/feedback event loop."
            ),
        },
        "paths": {
            "/v1/recommendations": {
                "post": {
                    "summary": "Return and durably register a recommendation slate",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {
                            "type": "object", "required": ["user_id"],
                            "properties": {
                                "user_id": {"type": "integer"},
                                "k": {"type": "integer", "minimum": 1,
                                      "maximum": 100, "default": 10},
                            },
                        }}},
                    },
                    "responses": {
                        "200": {
                            "description": "Recommendation slate",
                            "content": {"application/json": {"schema": {
                                "type": "object",
                                "required": ["recommendation_id", "request_id",
                                             "model_version", "policy_name",
                                             "recommendations"],
                                "properties": {
                                    "recommendation_id": {"type": "string"},
                                    "request_id": {"type": "string"},
                                    "model_version": {"type": "string"},
                                    "policy_name": {"type": "string"},
                                    "recommendations": {
                                        "type": "array", "items": recommendation},
                                },
                            }}},
                        },
                        "400": {"$ref": "#/components/responses/BadRequest"},
                        "404": {"$ref": "#/components/responses/NotFound"},
                        "429": {"$ref": "#/components/responses/Overloaded"},
                        "503": {"$ref": "#/components/responses/Unavailable"},
                    },
                }
            },
            "/v1/events/impression": {
                "post": {
                    "summary": "Idempotently record displayed recommendation items",
                    "responses": {"200": {"description": "Duplicate/no-op"},
                                  "201": {"description": "Created"},
                                  "400": {"$ref": "#/components/responses/BadRequest"}},
                }
            },
            "/v1/events/feedback": {
                "post": {
                    "summary": "Idempotently record feedback by event_id",
                    "responses": {"200": {"description": "Duplicate/no-op"},
                                  "201": {"description": "Created"},
                                  "400": {"$ref": "#/components/responses/BadRequest"}},
                }
            },
            "/live": {"get": {"responses": {"200": {"description": "Alive"}}}},
            "/ready": {"get": {"responses": {
                "200": {"description": "Ready"},
                "503": {"description": "Not ready"},
            }}},
        },
        "components": {
            "securitySchemes": {
                "ApiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
            },
            "responses": {
                "BadRequest": {"description": "Invalid request"},
                "NotFound": {"description": "Unknown user or resource"},
                "Overloaded": {"description": "Rate or admission limit exceeded"},
                "Unavailable": {"description": "Model or feedback store unavailable"},
            },
        },
    }
