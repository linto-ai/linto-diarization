"""
Thin wrapper around the Qdrant client for speaker identification.

This module MUST stay importable without torch/speechbrain (qdrant-client only):
it is used by the celery enrollment tasks and by the unit test suite.
"""

import os

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from identification.spkid_core import MODEL_DIM


class QdrantStore:
    """Qdrant client and collection/point level operations"""

    def __init__(self, host=None, port=None, api_key=None, log=None):
        self.host = host
        self.port = int(port) if port else 6333
        self.log = log
        if host:
            self.client = QdrantClient(
                url=f"http://{host}:{self.port}",
                api_key=api_key or None,
            )
        else:
            self.client = None

    @classmethod
    def from_env(cls, log=None):
        return cls(
            host=os.environ.get("QDRANT_HOST"),
            port=os.environ.get("QDRANT_PORT"),
            api_key=os.environ.get("QDRANT_API_KEY"),
            log=log,
        )

    @property
    def enabled(self):
        return self.client is not None

    def collection_exists(self, collection):
        return self.client.collection_exists(collection_name=collection)

    def ensure_collection(self, collection, dim=MODEL_DIM):
        """Create the collection if it does not exist. Returns True if created."""
        if self.collection_exists(collection):
            return False
        if self.log:
            self.log.info(f"Creating Qdrant collection: {collection}")
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
        )
        return True

    def upsert_point(self, collection, point_id, vector, payload):
        self.upsert_points(
            collection,
            [PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def upsert_points(self, collection, points):
        self.client.upsert(
            collection_name=collection,
            wait=True,
            points=points,
        )

    def delete_points(self, collection, point_ids):
        """Delete points by id. Idempotent: missing collection or points are fine.

        Returns the number of points that actually existed before deletion.
        """
        if not self.collection_exists(collection):
            return 0
        point_ids = list(point_ids)
        if not point_ids:
            return 0
        existing = self.client.retrieve(
            collection_name=collection,
            ids=point_ids,
            with_payload=False,
            with_vectors=False,
        )
        self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )
        return len(existing)

    def drop(self, collection):
        """Delete the collection. Idempotent. Returns True if it existed."""
        existed = self.collection_exists(collection)
        if existed:
            self.client.delete_collection(collection_name=collection)
        return existed

    def search(self, collection, vector, limit=10):
        """Vector similarity search. Returns a list of scored points (score + payload)."""
        if hasattr(vector, "tolist"):
            # torch.Tensor / numpy array -> list of floats
            vector = vector.tolist()
        if hasattr(self.client, "query_points"):
            return self.client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                with_payload=True,
            ).points
        # qdrant-client < 1.10 (search was removed in recent versions)
        return self.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
        )

    def get_collection_model_id(self, collection):
        """Read the model_id from one point of the collection (None if empty/unset)"""
        points, _ = self.client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=True,
        )
        if not points:
            return None
        return (points[0].payload or {}).get("model_id")
