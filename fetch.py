import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def scrape_bible_passage(url):
    """
    Scrape the text from a Bible Gateway passage.

    Args:
        url (str): The URL of the Bible Gateway passage page.

    Returns:
        str: The scraped text from the passage.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError if the request fails

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the passage text
        passage_div = soup.find('div', class_='passage-text')  # Main container for passage text
        if passage_div is None:
            print("Could not find passage text on the page.")
            return None

        chapter_titles = []
        for h3 in passage_div.find_all('h3'):
            title = h3.get_text(strip=True)
            chapter_titles.append(title)

        # Get all text inside the passage
        passage_text = passage_div.get_text(separator='\n', strip=True)

        arr = passage_text.split('\n')

        df = pd.DataFrame(columns=['Book', 'Chapter','Verse', 'Text'])

        # remove footnotes section and subsequent text
        idx = 0
        while True:
            if arr[idx] == 'Footnotes':
                break
            idx += 1

        arr = arr[:idx]

        for title in chapter_titles:
            df = pd.concat([df, pd.DataFrame([[book, chapter, 'Title', title]], columns=df.columns)], ignore_index=True)

        # add verses to dataframe
        idx = 0
        verse = 1
        text = ""
        while True:
            try:
                int(arr[idx])
                if text != "":
                    df = pd.concat([df, pd.DataFrame([[book, chapter, verse, text]], columns=df.columns)], ignore_index=True)
                    text = ""
                    idx += 1
                    verse += 1
                else:
                    idx += 1
            except ValueError:
                if arr[idx] == '[':
                    idx += 3
                elif arr[idx] in chapter_titles:
                    idx += 1
                else:
                    text += arr[idx] + " "
                    idx += 1
            except IndexError:
                df = pd.concat([df, pd.DataFrame([['Matthew', chapter, verse, text]], columns=df.columns)], ignore_index=True)
                break

        # Create the directory
        try:
            os.makedirs(f"data/{bible}/{book}")
            print(f"Directory '{book}' created successfully.")
        except FileExistsError:
            print(f"Directory '{book}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{book}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

        df.to_csv(f'./data/{bible}/{book}/{book} {chapter}.csv', index=False)


    except requests.exceptions.RequestException as e:
        print(f"Error occurred while making the request: {e}")
        return None


if __name__ == "__main__":
    for i in range(1, 29):
        book = "Matthew"
        chapter = i
        bible = "NRSVCE"
        
        # URL of the Bible passage
        url = f"https://www.biblegateway.com/passage/?search={book}%20{chapter}&version={bible}"

        # Scrape the text
        passage_text = scrape_bible_passage(url)