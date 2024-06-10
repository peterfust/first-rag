import csv
import os

import requests
from bs4 import BeautifulSoup

# URL der Website
url = "https://www.sg.ch/steuern-finanzen/steuern/steuerbuch/inhaltsverzeichnis.html"

# Basis-URL für die PDF-Links
base_url = "/content/dam/sgch/steuern-finanzen/steuern/steuerbuch"

# Ordner zum Speichern der heruntergeladenen PDFs
download_folder = "/raw_data/files"
os.makedirs("../../" + download_folder, exist_ok=True)

# Pfad zur CSV-Datei
csv_file = "../../raw_data/content.csv"

def download_pdfs_and_create_csv(url, base_url, download_folder, csv_file):
    # Anfrage an die Website
    response = requests.get(url)
    response.raise_for_status()  # Überprüft, ob die Anfrage erfolgreich war

    # HTML-Inhalt mit BeautifulSoup parsen
    soup = BeautifulSoup(response.content, 'html.parser')

    # CSV-Datei öffnen
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        # CSV-Header schreiben
        writer.writerow(["link_name", "article_name", "link_to_file"])

        # Alle relevanten <tr>-Tags finden
        rows = soup.find_all('tr')

        for row in rows:
            # <td>-Tags innerhalb der <tr>-Tags finden
            tds = row.find_all('td')
            if len(tds) == 2:
                # Link- und Beschreibungselemente extrahieren
                link_tag = tds[0].find('a')
                description_tag = tds[1].find('a')
                if link_tag and description_tag:
                    pdf_url = "https://www.sg.ch" + base_url + link_tag['href'].split(base_url)[-1]
                    article_name = link_tag.find('span', class_='description').text.strip()
                    link_name = description_tag.text.strip()
                    pdf_name = os.path.basename(pdf_url)

                    # Lokaler Pfad zum Speichern des PDFs
                    local_pdf_path = os.path.join('../..' + download_folder, pdf_name)

                    # PDF herunterladen
                    pdf_response = requests.get(pdf_url)
                    pdf_response.raise_for_status()

                    # PDF speichern
                    with open(local_pdf_path, 'wb') as pdf_file:
                        pdf_file.write(pdf_response.content)

                    print(f"Downloaded: {pdf_name}")

                    # Relativen Pfad zur Datei für CSV
                    relative_pdf_path = os.path.join('./..' + download_folder, pdf_name)

                    # Eintrag in CSV schreiben
                    writer.writerow([link_name, article_name, relative_pdf_path])

if __name__ == "__main__":
    download_pdfs_and_create_csv(url, base_url, download_folder, csv_file)
