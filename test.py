from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver

options = webdriver.ChromeOptions()

options.add_argument('--headless')
options.add_argument("--no-sandbox")

chrome_service = webdriver.chrome.service.Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=chrome_service, options=options)

url = 'https://race.netkeiba.com/race/shutuba.html?race_id=202205020211&rf=race_submenu'

driver.get(url)

print(driver.page_source)

dd