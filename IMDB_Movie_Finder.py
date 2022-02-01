import bs4 as bs
import urllib.request

import tkinter as tk
from tkinter import ttk

LARGE_FONT = ("Verdana", 12)
MEDIUM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)

source = urllib.request.urlopen('https://www.imdb.com/chart/top/?ref_=nv_mv_250').read()
soup = bs.BeautifulSoup(source, 'lxml')

table = soup.find('table')
table_rows = table.find_all('tr')

movies = []

for tr in table_rows:
    td = tr.find_all('td', class_='titleColumn')
    titles = [title.find('a').string for title in td]
    dates = [date.find('span', class_='secondaryInfo').string.translate({ ord(c): None for c in "()" }) for date in td]

    title_and_date = titles + dates

    movies.append(title_and_date)

movies.pop(0)

def find_movie(date):
    results = []

    for movie in movies:
        if movie[1] == str(date):
            results.append(movie[0])

    print(results)

find_movie(2020)
