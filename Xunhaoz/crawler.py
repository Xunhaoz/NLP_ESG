import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

import warnings

warnings.filterwarnings("ignore")


class Crawler:
    def __init__(self, url):
        self.url = url
        self.ua = UserAgent()

    def get_paragraph(self):
        response = requests.get(self.url, headers={
            "User-Agent": self.ua.googlechrome
        })
        soup = BeautifulSoup(response.text)
        ps = soup.find_all("p", class_=False, style=False)
        ps = [self.clean_data(p) for p in ps[:-1]]
        res = "".join(ps)
        return res

    def clean_data(self, content):
        content = content.text
        content = content.replace('\n', ' ')
        content = content.replace('\r', '')
        content = content.replace('\xa0', '')
        return content
