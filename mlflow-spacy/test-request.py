import requests

inference_request = {
    "parameters": {
      "content_type": "str"
    },
    "inputs": [
        {
          "name": "test",
          "shape": [1, 1],
          "datatype": "BYTES",
          "data": [" Roeselare verkocht gemotiveerd student gevaarlijke eindeloze Koerdische luchthavens getal CERA uiteindelijke gesprekken overbodig losse achterop ongeval grafische belanden vergezeld witloof 1013ZK. steen ein VU buitenstaander alleen ongeval gepaard begeven file regelmaat Kennedy schuld vertrok dozen Ford Landen Nick voorbereiden vult benoemd 1013ZK. avec deel conservatief loonnorm ken huisvesting oceaan westen onvermijdelijk Wil schrijven mogelijke stamt muren gesprekken vult vorming Ludwig bezorgd bestemmingen 1013 ZK. felle Elizabeth Express evenmin Buffett vult herhaald ontwerpers vertegenwoordigers strand Mobutu bedenken rekenen gÃ©Ã©n terechtgekomen hoort volumes authentieke wilden voorbereiden 1013ZK. westen afstand tweeduizend besliste Planet Hendrik onderschreven Finance menselijke conservatief nauwe verklaart constructeur Sam boeiende stamt CERA financier verwachte integratie 2596 CX."]
        }
    ]
}

endpoint = "http://localhost:8080/v2/models/nn-ee/infer"
response = requests.post(endpoint, json=inference_request)

print(response.json())
