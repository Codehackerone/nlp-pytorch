import json

import requests

url = 'https://parseapi.back4app.com/classes/Complete_List_Names?limit=100000000000&keys=Name'
headers = {
    'X-Parse-Application-Id': 'zsSkPsDYTc2hmphLjjs9hz2Q3EXmnSxUyXnouj1I',
    'X-Parse-Master-Key': '4LuCXgPPXXO2sU5cXm6WwpwzaKyZpo3Wpj4G4xXK'
}
data = json.loads(requests.get(url, headers=headers).content.decode('utf-8'))
names = [i['Name'] for i in data['results']]

with open('names.txt', 'w') as f:
    f.write('\n'.join(names))
