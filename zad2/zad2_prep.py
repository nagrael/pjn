import json
from datetime import datetime
import tarfile

import itertools
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk


def given_date(date):
    z = datetime.strptime(date['judgmentDate'], '%Y-%m-%d')
    return z.year == 2015


def load_judgment(tars):
    for tarinfo in tars:
        if tarinfo.isreg():
            if tarinfo.name.startswith('data/json/judgments'):
                yield from filter(given_date, itertools.chain(json.load(tars.extractfile(tarinfo))["items"]))


def create_index(client):
    client.indices.create(
        index="judgment",
        body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,

                "analysis": {
                    "analyzer": {
                        "polish_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["morfologik_stem"]
                        }
                    }
                }
            },
            "mappings": {
                "doc": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "date": {"type": "date"},
                        "text_content": {"type": "text", "analyzer": "polish_analyzer"},
                        "judges": {"type": "keyword"}
                    }
                }
            }
        }
    )


def reform_es(items):
    for item in items:
        yield {
            "text_content": item["textContent"],
            "date": item["judgmentDate"],
            "id": item["id"],
            "judges": [judge["name"] for judge in item["judges"]]
        }


def create_docs(client, items):
    success, failed = 0, 0
    for ok, result in streaming_bulk(client, items, index="judgment", doc_type="doc", max_retries=5, chunk_size=250):
        if not ok:
            failed += 1
        else:
            success += 1
    print(f"Created {success} indexes, with failed {failed}.")


if __name__ == "__main__":
    tar = tarfile.open("../saos-dump-23.02.2018.tar.gz", "r:gz", )
    es = Elasticsearch("localhost:9200")
    # create_index(es)
    create_docs(es, reform_es(load_judgment(tar)))
    es.indices.refresh(index="judgment")
    tar.close()
