import requests

url_docker = 'http://localhost:9696/predict'
url = 'http://sleep-serving-env.eba-mncuu926.ap-southeast-7.elasticbeanstalk.com/'

patient = {
    "gender": "Female",
    "age": 59,
    "occupation": "Nurse",
    "sleep_duration": 8.1,
    "quality_of_sleep": 9,
    "physical_activity_level": 75,
    "stress_level": 3,
    "bmi_category": "Overweight",
    "heart_rate": 68,
    "daily_steps": 7000,
    "systolic": 140,
    "diastolic": 95,
    "pulse_pressure": 45,
    "sleep_per_stress": 2.025
}


response = requests.post(url, json=patient).json()
print(response)

if response['prediction'] != None:
    print('condition is %s' % response['prediction'])
else:
    print('not sending promo email to %s' % response['prediction'])