from elasticsearch import Elasticsearch
import json
from typing import Any, Dict, Sequence, Tuple
import yaml

from ao3_tags import CONFIG_PATH

with open(CONFIG_PATH / "config.yaml") as f:
    conf_dict = yaml.safe_load(f)
    API_KEY = (conf_dict["kibana"]["key"])
    ES_HOST = conf_dict["kibana"]["host"]

ES_ARGS = {"hosts": [f"{ES_HOST}:9200"],
           "api_key": API_KEY,
           "retry_on_timeout": True,
           "request_timeout": 240}
CLIENT = Elasticsearch(**ES_ARGS)


def generate_query(query_type: str, should_list,
                   fandoms=None, must: Sequence[Dict] = None, must_not: Sequence[Dict] = None) -> Dict:
    # Prepare must and must_not based on input
    must = must if must is not None else []
    must_not = must_not if must_not is not None else []

    # Filter only on english chapters
    must.append({'match': {'language': 'English'}})

    # Add should condition based on the type
    should = [{'match': {query_type: x}} for x in should_list]

    # Add fandoms if defined
    if fandoms is not None:
        for fandom in fandoms:
            must.append({'match': {'fandom': fandom}})

    return {
        'bool': {
            'should': should,
            'minimum_should_match': 1,
            'must_not': must_not,
            'must': must
        }
    }


# Function to create an Elasticsearch configuration used in the Spark connector
def create_es_conf(kibana_user: str, kibana_pw: str, query: Dict,
                   es_resource: str = "ao3-v3-chapters", es_nodes: str = ES_HOST):
    # Convert the query to a json to be valid for Elastic
    query = json.dumps(query, indent=4)

    return {
        "inferSchema": "true",
        "es.nodes": es_nodes,
        "es.port": "9200",
        "es.net.ssl": "true",
        "es.resource.read": es_resource,
        "es.scroll.keepalive": "30m",
        "es.scroll.size": "5000",
        "es.http.timeout": "10m",
        "es.read.field.as.array.include": "author_name,freeform,chap_content",
        "es.net.http.auth.user": kibana_user,
        "es.net.http.auth.pass": kibana_pw,
        "es.query": query
    }


def create_es_input(api_key: str, query: Dict) -> Tuple[Dict, Dict]:
    search_body = {
        'query': query,
        'fields': [{"field": "chap_content"},
                   {"field": "freeform"},
                   {"field": "fandom"},
                   {"field": "_score"},
                   {"field": "_id"}]
    }
    return ES_ARGS, search_body


def get_number_of_shards(index):
    res = CLIENT.cat.shards(index=index, format='json')
    num_shards = len([i for i in res if i['index'] == index])
    return num_shards


def get_chapter_tags(element: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return {"chapter_id": element['fields']['_id'][0],
                "tags": element['fields']['freeform']}
    except:
        return None
