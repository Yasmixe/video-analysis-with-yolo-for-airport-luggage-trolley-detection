from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Configurer le navigateur
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

# Accéder à Google Images
driver.get("https://www.google.com/imghp?hl=en")

# Faire la recherche
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys("luggage airport trolley")
search_box.send_keys(Keys.RETURN)

# Attendre le chargement
time.sleep(10)

# Scroller pour charger plus de résultats
for _ in range(10):
    driver.execute_script("window.scrollBy(0, 1000);")
    time.sleep(4)

# Trouver tous les éléments avec la classe 'ob5Hkd'
elements = driver.find_elements(By.CLASS_NAME, "ob5Hkd")
print(len(elements))
hrefs = []

for el in elements:
    a_tag = el.find_element(By.TAG_NAME, "a")
    #print(a_tag)
    href = a_tag.get_attribute("href")
    print(href)
    hrefs.append(href)
    with open("hrefs.txt", "w", encoding="utf-8") as f:
        f.write(href + "\n")



print(f"{len(hrefs)} liens bruts extraits et sauvegardés dans hrefs.txt")

driver.quit()
