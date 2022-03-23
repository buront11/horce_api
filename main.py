import time

import pandas as pd
import requests
import cchardet

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver

options = webdriver.ChromeOptions()

options.add_argument('--headless')
options.add_argument("--no-sandbox")

chrome_service = webdriver.chrome.service.Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=chrome_service, options=options)

driver.get('https://race.netkeiba.com/race/shutuba.html?race_id=202206020411&rf=race_submenu')

df_horses = pd.read_html(driver.page_source)[0]
df_horses.columns = df_horses.columns.droplevel(0)

print(df_horses)

driver.quit()