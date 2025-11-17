# ===================================================================================
# Project: ChatTensorFlow
# File: src/services/indexing/index_config.py
# Description: OpenSearch index configuration for TensorFlow documentation
# Author: LALAN KUMAR
# Created: [09-11-2025]
# Updated: [09-11-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.1.0
# ===================================================================================

TENSORFLOW_INDEX_BODY = {
    "settings": {
        "index": {
            "knn": True,
            "number_of_shards": 2,
            "number_of_replicas": 1,
            "refresh_interval": "1s"
        },
        "analysis": {
            "analyzer": {
                "code_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "code_filter"]
                },
                "tensorflow_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "tensorflow_synonym"]
                }
            },
            "filter": {
                "code_filter": {
                    "type": "word_delimiter",
                    "preserve_original": True,
                    "split_on_numerics": False
                },
                "tensorflow_synonym": {
                    "type": "synonym",
                    "synonyms": [
                        "tensor, tensors",
                        "layer, layers",
                        "model, models",
                        "neural network, nn, network",
                        "activation, activation function",
                        "optimizer, optimizers",
                        "loss, loss function",
                        "training, train, fit",
                        "inference, predict, prediction",
                        "epoch, epochs",
                        "batch, batches",
                        "gradient, gradients",
                        "keras, tf.keras",
                        "dataset, data, tf.data",
                        "sequential, sequential model",
                        "functional, functional api",
                        "compile, compilation",
                        "callback, callbacks",
                        "metric, metrics",
                        "regularization, regularizer",
                        "dropout, drop out",
                        "embedding, embeddings",
                        "conv, convolution, convolutional",
                        "lstm, rnn, recurrent",
                        "dense, fully connected",
                        "flatten, flattening"
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # Identifiers
            "chunk_id": {"type": "keyword"},
            "page_type": {"type": "keyword"},
            
            # Content fields
            "heading": {
                "type": "text",
                "analyzer": "tensorflow_analyzer",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "text": {
                "type": "text",
                "analyzer": "tensorflow_analyzer"
            },
            "full_text": {
                "type": "text",
                "analyzer": "code_analyzer"
            },
            "enriched_text": {
                "type": "text",
                "analyzer": "tensorflow_analyzer"
            },
            
            # Vector embedding
            "embedding": {
                "type": "knn_vector",
                "dimension": 768,  # Adjust based on your embedding model
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16
                    }
                }
            },
            
            # Page metadata
            "page_title": {
                "type": "text",
                "analyzer": "tensorflow_analyzer",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "source_url": {"type": "keyword"},
            "breadcrumbs": {"type": "keyword"},
            
            # Code blocks
            "code_blocks": {
                "type": "nested",
                "properties": {
                    "index": {"type": "integer"},
                    "code": {
                        "type": "text",
                        "analyzer": "code_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 1024}
                        }
                    },
                    "language": {"type": "keyword"},
                    "context": {
                        "type": "text",
                        "analyzer": "tensorflow_analyzer"
                    },
                    "lines": {"type": "integer"}
                }
            },
            "has_code": {"type": "boolean"},
            "total_code_lines": {"type": "integer"},
            
            # Statistics
            "word_count": {"type": "integer"},
            
            # Timestamps
            "indexed_at": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis"
            },
            "crawled_at": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis"
            }
        }
    }
}
