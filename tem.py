import requests
import json

query_json = {

  "inputs": [

    [

      -0.09269547780327612,

      -0.044641636506989144,

      0.028284032228378497,

      -0.015998975220305175

    ]

  ]

}

query = json.dumps(query_json)

headers = {'Content-Type': 'application/json'}

# http://<ingress>/seldon/seldon/iris-model/api/v1.0/predictions

request_uri = 'http://0.0.0.0:5000/invocations'

response = requests.post(request_uri, data=query, headers=headers)

 

print(response.content)