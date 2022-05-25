#Convert an audio file of an English speaker to text using a Speech to Text API 
#Translate the English version to a Polish version using a Language Translator API

#Requirements: 
# - IBM Cloud account
# - credentials for Speach To Text
# - credentials for Language Translator
# - wget installed (https://www.jcchouinard.com/wget/)

#For additional information regarding syntax, check documentation for Speach To Text
#and Language Translator

#Libraries
!pip install ibm_watson wget
from ibm_watson import SpeechToTextV1 
from ibm_watson import LanguageTranslatorV3
import json
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from pandas import json_normalize

#Speech to Text endpoint and API key
url_s2t = ' '
iam_apikey_s2t = ' '

#Language Translator endpoint and API key
url_lt = ' '
apikey_lt = ' '

###### SPEECH TO TEXT ######

#Create Speech to Text Adapter object
authenticator = IAMAuthenticator(iam_apikey_s2t)
s2t = SpeechToTextV1(authenticator=authenticator)
s2t.set_service_url(url_s2t)

#Audiofile to convert
!wget -O 256[kb]angry-customer-resolve-my-fckn-issue.mp3.mp3  https://sampleswap.org/samples-ghost/VOCALS%20and%20SPOKEN%20WORD/Voicemail%20Messages/256[kb]angry-customer-resolve-my-fckn-issue.mp3.mp3
filename='256[kb]angry-customer-resolve-my-fckn-issue.mp3.mp3'
#The better quality of the file uploaded, the better are results.

#Create an object
with open(filename, mode="rb") as wav:
    response = s2t.recognize(audio=wav, content_type='audio/mp3')
    
#The attribute result contains a dictionary that includes the translation
#and it can be assigned to the variable recognized_text
response.result  
recognized_text=response.result['results'][0]["alternatives"][0]["transcript"]

#Check confidence in respect of voice recognition
print(json_normalize(response.result['results'],"alternatives"))

###### LANGUAGE TRANSLATOR ######

version_lt='2018-05-01'  #parameter required by API request

#Create Language Translator object
authenticator = IAMAuthenticator(apikey_lt)
language_translator = LanguageTranslatorV3(version=version_lt,authenticator=authenticator)
language_translator.set_service_url(url_lt)

#Translate text
translation_response = language_translator.translate(text=recognized_text, model_id='en-pl')  #retrieves a dictionary
translation=translation_response.get_result()  #retrieves translation as a string

polish_translation = translation['translations'][0]['translation']

print("The result of the translation is as follows:")
print(polish_translation)
