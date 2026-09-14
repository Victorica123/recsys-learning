"""Contract checks that do not require data, checkpoints, or a live server."""
import unittest

from api_contract import API_VERSION, openapi_document


class ApiContractTests(unittest.TestCase):
    def test_openapi_has_versioned_delivery_and_event_loop(self):
        document = openapi_document()
        self.assertEqual(document["openapi"], "3.1.0")
        self.assertEqual(document["info"]["version"], API_VERSION)
        self.assertEqual(
            {"/v1/recommendations", "/v1/events/impression", "/v1/events/feedback"},
            set(document["paths"]) - {"/live", "/ready"},
        )

    def test_recommend_contract_exposes_operational_failures(self):
        responses = document_responses = (
            openapi_document()["paths"]["/v1/recommendations"]["post"]["responses"]
        )
        self.assertEqual(set(responses), {"200", "400", "404", "429", "503"})
        required = document_responses["200"]["content"]["application/json"][
            "schema"]["required"]
        self.assertIn("model_version", required)
        self.assertIn("recommendation_id", required)


if __name__ == "__main__":
    unittest.main()
