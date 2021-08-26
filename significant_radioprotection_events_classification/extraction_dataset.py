#Extraction dataset
#Import module
import os
import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd

#create list of urls
urls=[]

#importer les urls des pages regroupant l'ensemble des avis de radiothérapie
for i in range(1,7):
    url1 = "https://www.asn.fr/Controler/Actualites-du-controle/Avis-d-incident-affectant-un-patient-en-radiotherapie/(page)/"
    urls.append( url1 + str(i))
print(urls)

#importer les codes html des pages regroupant l'ensemble des avis de radiothérapie
all_soup=[]

for url2 in urls:
    resp = requests.get(url2)
    soup = BeautifulSoup(resp.text, 'lxml')
    all_soup.append(soup)

print(all_soup)

#importer les urls des sous-pages regroupant l'ensemble des avis de radiothérapie
urls3= []

for i in all_soup : 
    for h in i.find_all("div", {"class": "Teaser"}):
        a = h.find('a')
        urls3.append(a.attrs['href'])
    
print(urls3)


#importer les informations (titre, date, loc , descri , grade) des sous-pages regroupant l'ensemble des avis de radiothérapie
all_soup2=[]

for i in urls3 : 
    resp = requests.get(i)
    soup = BeautifulSoup(resp.text, 'lxml')
    all_soup2.append(soup)

print(all_soup2[2])    

#liste de titre, date, localisation, description et grade
list_titre=[]
list_date=[]
list_localisation=[]
list_description=[]
list_grade=[]

#extraction des titre, date, localisation, description et grade
for i in all_soup2 :
    titre = i.find("h1").text 
    date = i.find("p", {"class": "Teaser-infos Teaser-infos--grey"}).text
    localisation = i.find("p", {"class": "Teaser-infos Teaser-infos--black"}).text
    description = i.find("div", {"class": "ezxmltext-field"}).text
    grade = i.find("span", {"class": "selected"}).text
    
    list_titre.append(titre)
    list_date.append(date)
    list_localisation.append(localisation)
    list_description.append(description)
    list_grade.append(grade)


df = pd.DataFrame({"Titre": list_titre, "Date": list_date, "Localisation": list_localisation, "Description": list_description, "Grade": list_grade })
df.to_excel(r'df.xlsx')