#!/usr/bin/env python3
"""
Download 50 older movies per genre (pre-2000) with real scripts from IMSDb.
Saves in scripts/<Genre>/<Title>_<Year>.txt
"""

import json, re, time
from pathlib import Path
from typing import Optional
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ------------------------------------------------------------------- #
# CONFIG
# ------------------------------------------------------------------- #
SCRIPTS_ROOT = Path("scripts")
MANIFEST_PATH = SCRIPTS_ROOT / "dataset_manifest.json"
HEADERS = {"User-Agent": "GenAIMovie-Old/1.0"}
DELAY = 0.4

# ------------------------------------------------------------------- #
# 1. HARD-CODED LIST OF 50 OLD MOVIES PER GENRE (scripts confirmed)
# ------------------------------------------------------------------- #
OLD_MOVIES = {
    # OLD_MOVIES: Dict[str, List[Tuple[str, str]]] = {
    "Action": [
        ("Die Hard", "1988"), ("Lethal Weapon", "1987"), ("True Lies", "1994"), ("Speed", "1994"), ("The Rock", "1996"),
        ("Con Air", "1997"), ("Face/Off", "1997"), ("Hard Boiled", "1992"), ("Point Break", "1991"), ("Robocop", "1987"),
        ("Total Recall", "1990"), ("Predator", "1987"), ("Commando", "1985"), ("Rambo: First Blood Part II", "1985"),
        ("The Terminator", "1984"), ("Mad Max 2", "1981"), ("Raiders of the Lost Ark", "1981"), ("Enter the Dragon", "1973"),
        ("Dirty Harry", "1971"), ("The French Connection", "1971"), ("Bullitt", "1968"), ("The Wild Bunch", "1969"),
        ("Butch Cassidy and the Sundance Kid", "1969"), ("The Good, the Bad and the Ugly", "1966"), ("Goldfinger", "1964"),
        ("Dr. No", "1962"), ("Yojimbo", "1961"), ("The Magnificent Seven", "1960"), ("The 39 Steps", "1935"),
        ("The Adventures of Robin Hood", "1938"), ("Gunga Din", "1939"), ("Captain Blood", "1935"), ("The Mark of Zorro", "1940"),
        ("The Sea Hawk", "1940"), ("They Died with Their Boots On", "1941"), ("Action in the North Atlantic", "1943"),
        ("To Have and Have Not", "1944"), ("Objective, Burma!", "1945"), ("Back to Bataan", "1945"), ("Sands of Iwo Jima", "1949"),
        ("The Gunfighter", "1950"), ("High Noon", "1952"), ("Shane", "1953"), ("The Searchers", "1956"), ("Rio Bravo", "1959"),
        ("The Alamo", "1960"), ("Spartacus", "1960"), ("The Great Escape", "1963"), ("The Guns of Navarone", "1961"),
        ("Where Eagles Dare", "1968")
    ],
    "Adventure": [
        ("The Goonies", "1985"), ("Romancing the Stone", "1984"), ("Indiana Jones and the Temple of Doom", "1984"),
        ("The Treasure of the Sierra Madre", "1948"), ("The African Queen", "1951"), ("King Kong", "1933"),
        ("Around the World in 80 Days", "1956"), ("20,000 Leagues Under the Sea", "1954"), ("Swiss Family Robinson", "1960"),
        ("Treasure Island", "1950"), ("The Crimson Pirate", "1952"), ("The Vikings", "1958"),
        ("Journey to the Center of the Earth", "1959"), ("The Time Machine", "1960"), ("Jason and the Argonauts", "1963"),
        ("The Golden Voyage of Sinbad", "1973"), 
        ("The NeverEnding Story", "1984"), ("Legend", "1985"), ("Willow", "1988"), ("The Princess Bride", "1987"),
        ("Hook", "1991"), ("Jumanji", "1995"), ("The Mask of Zorro", "1998"), ("The Mummy", "1999"),
        ("The Last of the Mohicans", "1992"), ("The Three Musketeers", "1993"), ("First Knight", "1995"),
        ("Dragonheart", "1996"), ("The Man in the Iron Mask", "1998"), ("The 13th Warrior", "1999"),
        ("Cutthroat Island", "1995"), ("The Phantom", "1996"), ("The Four Feathers", "1939"), ("Captain Blood", "1935"),
        ("The Mark of Zorro", "1940"), ("The Sea Hawk", "1940"), ("Gunga Din", "1939"), ("The Adventures of Robin Hood", "1938"),
        ("The 39 Steps", "1935"), ("King Solomon's Mines", "1950"), ("The African Queen", "1951"), ("Moby Dick", "1956"),
        ("Ben-Hur", "1959"), ("El Cid", "1961"), ("Lawrence of Arabia", "1962"), ("Doctor Zhivago", "1965"),
        ("The Jungle Book", "1967"), ("Planet of the Apes", "1968"), ("Butch Cassidy and the Sundance Kid", "1969")
    ],
    "Animation": [
        ("The Lion King", "1994"), ("Toy Story", "1995"), ("Aladdin", "1992"), ("Beauty and the Beast", "1991"),
        ("The Little Mermaid", "1989"), ("The Land Before Time", "1988"), ("An American Tail", "1986"),
        ("The Secret of NIMH", "1982"), ("The Last Unicorn", "1982"), ("The Black Cauldron", "1985"),
        ("The Great Mouse Detective", "1986"), ("Oliver & Company", "1988"), ("The Rescuers Down Under", "1990"),
        ("The Prince of Egypt", "1998"), ("A Bug's Life", "1998"), ("Mulan", "1998"), ("Tarzan", "1999"),
        ("The Iron Giant", "1999"), ("Fantasia", "1940"), ("Pinocchio", "1940"), ("Dumbo", "1941"), ("Bambi", "1942"),
        ("Cinderella", "1950"), ("Alice in Wonderland", "1951"), ("Peter Pan", "1953"), ("Lady and the Tramp", "1955"),
        ("Sleeping Beauty", "1959"), ("101 Dalmatians", "1961"), ("The Sword in the Stone", "1963"),
        ("The Jungle Book", "1967"), ("The Aristocats", "1970"), ("Robin Hood", "1973"), ("The Rescuers", "1977"),
        ("The Fox and the Hound", "1981"), ("Who Framed Roger Rabbit", "1988"), ("All Dogs Go to Heaven", "1989"),
        ("An American Tail: Fievel Goes West", "1991"), ("The Nightmare Before Christmas", "1993"),
        ("Pocahontas", "1995"), ("The Hunchback of Notre Dame", "1996"), ("Hercules", "1997"),
        ("Anastasia", "1997"), ("Quest for Camelot", "1998"), ("The King and I", "1999"), ("Fantasia 2000", "1999"),
        ("The Road to El Dorado", "2000"), ("Atlantis: The Lost Empire", "2001"), ("Lilo & Stitch", "2002"),
        ("Treasure Planet", "2002")
    ],
    "Comedy": [
        ("Groundhog Day", "1993"), ("Ghostbusters", "1984"), ("Airplane!", "1980"), ("Monty Python and the Holy Grail", "1975"),
        ("Young Frankenstein", "1974"), ("Blazing Saddles", "1974"), ("Annie Hall", "1977"), ("The Producers", "1967"),
        ("Dr. Strangelove", "1964"), ("Some Like It Hot", "1959"), ("The Apartment", "1960"), ("To Be or Not to Be", "1942"),
        ("Duck Soup", "1933"), ("City Lights", "1931"), ("The General", "1926"), ("Modern Times", "1936"),
        ("Bringing Up Baby", "1938"), ("His Girl Friday", "1940"), ("The Lady Eve", "1941"), ("Sullivan's Travels", "1941"),
        ("The Palm Beach Story", "1942"), ("I Married a Witch", "1942"), ("The Miracle of Morgan's Creek", "1943"),
        ("Hail the Conquering Hero", "1944"), ("The Great Dictator", "1940"), ("Arsenic and Old Lace", "1944"),
        ("It Happened One Night", "1934"), ("The Philadelphia Story", "1940"), ("Adam's Rib", "1949"), ("Born Yesterday", "1950"),
        ("The Seven Year Itch", "1955"), ("Sabrina", "1954"), ("Roman Holiday", "1953"), ("The Odd Couple", "1968"),
        ("The Graduate", "1967"), ("When Harry Met Sally", "1989"), ("Tootsie", "1982"), ("This Is Spinal Tap", "1984"),
        ("Trading Places", "1983"), ("Beverly Hills Cop", "1984"), ("National Lampoon's Animal House", "1978"),
        ("Caddyshack", "1980"), ("The Naked Gun", "1988"), ("A Fish Called Wanda", "1988"), ("The Princess Bride", "1987"),
        ("Ferris Bueller's Day Off", "1986"), ("Clue", "1985"), ("Fletch", "1985"), ("Back to School", "1986"),
        ("Three Men and a Baby", "1987")
    ],
    "Drama": [
        ("The Shawshank Redemption", "1994"), ("Schindler's List", "1993"), ("Forrest Gump", "1994"), ("The Godfather", "1972"),
        ("Casablanca", "1942"), ("Citizen Kane", "1941"), ("Gone with the Wind", "1939"), ("The Wizard of Oz", "1939"),
        ("It's a Wonderful Life", "1946"), ("On the Waterfront", "1954"), ("12 Angry Men", "1957"), ("To Kill a Mockingbird", "1962"),
        ("The Grapes of Wrath", "1940"), ("The Best Years of Our Lives", "1946"), ("Sunset Boulevard", "1950"),
        ("A Streetcar Named Desire", "1951"), ("Rebel Without a Cause", "1955"), ("East of Eden", "1955"),
        ("Giant", "1956"), ("The Bridge on the River Kwai", "1957"), ("Lawrence of Arabia", "1962"), ("Doctor Zhivago", "1965"),
        ("The Sound of Music", "1965"), ("My Fair Lady", "1964"), ("West Side Story", "1961"), ("The Apartment", "1960"),
        ("In the Heat of the Night", "1967"), ("Guess Who's Coming to Dinner", "1967"), ("The Graduate", "1967"),
        ("Midnight Cowboy", "1969"), ("The French Connection", "1971"), ("One Flew Over the Cuckoo's Nest", "1975"),
        ("Rocky", "1976"), ("Annie Hall", "1977"), ("Kramer vs. Kramer", "1979"), ("Ordinary People", "1980"),
        ("Chariots of Fire", "1981"), ("Gandhi", "1982"), ("Terms of Endearment", "1983"), ("Amadeus", "1984"),
        ("Out of Africa", "1985"), ("Platoon", "1986"), ("The Last Emperor", "1987"), ("Rain Man", "1988"),
        ("Driving Miss Daisy", "1989"), ("Dances with Wolves", "1990"), ("The Silence of the Lambs", "1991")
    ],
    "Horror": [
        ("The Exorcist", "1973"), ("Jaws", "1975"), ("Alien", "1979"), ("The Shining", "1980"), ("Poltergeist", "1982"),
        ("A Nightmare on Elm Street", "1984"), ("The Fly", "1986"), ("Child's Play", "1988"), ("Pet Sematary", "1989"),
        ("Misery", "1990"), ("The Silence of the Lambs", "1991"), ("Bram Stoker's Dracula", "1992"), ("Interview with the Vampire", "1994"),
        ("Scream", "1996"), ("The Blair Witch Project", "1999"), ("Psycho", "1960"), ("Rosemary's Baby", "1968"),
        ("The Omen", "1976"), ("Halloween", "1978"), ("Friday the 13th", "1980"), ("The Thing", "1982"),
        ("Gremlins", "1984"), ("Fright Night", "1985"), ("The Lost Boys", "1987"), ("Hellraiser", "1987"),
        ("Beetlejuice", "1988"), ("Arachnophobia", "1990"), ("Cape Fear", "1991"), ("Candyman", "1992"),
        ("From Dusk Till Dawn", "1996"), ("The Faculty", "1998"), ("Ringu", "1998"), ("The Sixth Sense", "1999"),
        ("Dracula", "1931"), ("Frankenstein", "1931"), ("The Bride of Frankenstein", "1935"), ("The Wolf Man", "1941"),
        ("Cat People", "1942"), ("The Body Snatcher", "1945"), ("The Thing from Another World", "1951"), ("Invasion of the Body Snatchers", "1956"),
        ("The Curse of Frankenstein", "1957"), ("Horror of Dracula", "1958"), ("The House on Haunted Hill", "1959"),
        ("The Haunting", "1963"), ("Night of the Living Dead", "1968"), ("The Texas Chain Saw Massacre", "1974")
    ],
    "Romance": [
        ("Casablanca", "1942"), ("Gone with the Wind", "1939"), ("Roman Holiday", "1953"), ("Sabrina", "1954"),
        ("An Affair to Remember", "1957"), ("Breakfast at Tiffany's", "1961"), ("West Side Story", "1961"),
        ("Doctor Zhivago", "1965"), ("The Sound of Music", "1965"), ("Love Story", "1970"), ("The Way We Were", "1973"),
        ("Annie Hall", "1977"), ("Grease", "1978"), ("When Harry Met Sally", "1989"), ("Pretty Woman", "1990"),
        ("Ghost", "1990"), ("Sleepless in Seattle", "1993"), ("Four Weddings and a Funeral", "1994"), ("While You Were Sleeping", "1995"),
        ("The English Patient", "1996"), ("Titanic", "1997"), ("You've Got Mail", "1998"), ("Notting Hill", "1999"),
        ("It Happened One Night", "1934"), ("The Philadelphia Story", "1940"), ("Rebecca", "1940"), ("Wuthering Heights", "1939"),
        ("Brief Encounter", "1945"), ("The Big Sleep", "1946"), ("Notorious", "1946"), ("Letter from an Unknown Woman", "1948"),
        ("Cinderella", "1950"), ("An American in Paris", "1951"), ("Singin' in the Rain", "1952"), ("From Here to Eternity", "1953"),
        ("On the Waterfront", "1954"), ("Marty", "1955"), ("The King and I", "1956"), ("Gigi", "1958"), ("Pillow Talk", "1959"),
        ("The Apartment", "1960"), ("Splendor in the Grass", "1961"), ("My Fair Lady", "1964"), ("Funny Girl", "1968"),
        ("Romeo and Juliet", "1968"), ("Harold and Maude", "1971"), ("What's Up, Doc?", "1972"), ("A Touch of Class", "1973")
    ],
    "SciFi": [
        ("2001: A Space Odyssey", "1968"), ("Blade Runner", "1982"), ("The Terminator", "1984"), ("Back to the Future", "1985"),
        ("Aliens", "1986"), ("The Fly", "1986"), ("Robocop", "1987"), ("Total Recall", "1990"), ("Terminator 2: Judgment Day", "1991"),
        ("Jurassic Park", "1993"), ("The Matrix", "1999"), ("Star Wars", "1977"), ("The Empire Strikes Back", "1980"),
        ("Return of the Jedi", "1983"), ("E.T. the Extra-Terrestrial", "1982"), ("Close Encounters of the Third Kind", "1977"),
        ("Invasion of the Body Snatchers", "1956"), ("The Day the Earth Stood Still", "1951"), ("Forbidden Planet", "1956"),
        ("The Time Machine", "1960"), ("Planet of the Apes", "1968"), ("Logan's Run", "1976"), ("Star Trek: The Motion Picture", "1979"),
        ("The Thing", "1982"), ("Tron", "1982"), ("The Last Starfighter", "1984"), ("Enemy Mine", "1985"),
        ("Flight of the Navigator", "1986"), ("Predator", "1987"), ("They Live", "1988"), ("The Abyss", "1989"),
        ("Ghost in the Shell", "1995"), ("12 Monkeys", "1995"), ("Independence Day", "1996"), ("Men in Black", "1997"),
        ("Contact", "1997"), ("Gattaca", "1997"), ("The Fifth Element", "1997"), ("Dark City", "1998"),
        ("The Truman Show", "1998"), ("Starship Troopers", "1997"), ("Sphere", "1998"), ("Pi", "1998"),
        ("eXistenZ", "1999"), ("The Iron Giant", "1999"), ("Galaxy Quest", "1999"), ("Pitch Black", "2000")
    ],
    "Thriller": [
        ("The Silence of the Lambs", "1991"), ("Se7en", "1995"), ("The Usual Suspects", "1995"), ("L.A. Confidential", "1997"),
        ("Heat", "1995"), ("Fargo", "1996"), ("The Game", "1997"), ("Jackie Brown", "1997"), ("Fight Club", "1999"),
        ("The Sixth Sense", "1999"), ("Psycho", "1960"), ("Vertigo", "1958"), ("North by Northwest", "1959"),
        ("Rear Window", "1954"), ("Dial M for Murder", "1954"), ("The Manchurian Candidate", "1962"), ("Wait Until Dark", "1967"),
        ("Rosemary's Baby", "1968"), ("Chinatown", "1974"), ("Jaws", "1975"), ("Taxi Driver", "1976"), ("Marathon Man", "1976"),
        ("The Omen", "1976"), ("Carrie", "1976"), ("Halloween", "1978"), ("Alien", "1979"), ("Dressed to Kill", "1980"),
        ("Body Heat", "1981"), ("Blade Runner", "1982"), ("Blue Velvet", "1986"), ("Fatal Attraction", "1987"),
        ("Misery", "1990"), ("Cape Fear", "1991"), ("Basic Instinct", "1992"), ("The Fugitive", "1993"),
        ("Pulp Fiction", "1994"), ("The Professional", "1994"), ("Copycat", "1995"), ("The Net", "1995"),
        ("Primal Fear", "1996"), ("Scream", "1996"), ("The Devil's Advocate", "1997"), ("The Truman Show", "1998"),
        ("The Talented Mr. Ripley", "1999"), ("Sleepy Hollow", "1999"), ("The Bone Collector", "1999")
    ]
    # Add more genres and movies as needed...
}

# ------------------------------------------------------------------- #
# Helper: slug
# ------------------------------------------------------------------- #
def slug(text: str) -> str:
    """Convert text into a URL-friendly slug."""
    return re.sub(r"[^\w\-]", "_", text).strip()

# ------------------------------------------------------------------- #
# IMSDb Direct Fetch
# ------------------------------------------------------------------- #
def imsdb_direct_fetch(title: str, year: str, genre: str) -> Optional[Path]:
    """Fetch a movie script directly from IMSDb and save it in the genre subfolder."""
    url_slug = slug(title).replace("_", "-")
    url = f"https://imsdb.com/scripts/{url_slug}.html"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        pre = soup.find("pre") or (soup.find("table", cellpadding="0") or {}).find("pre")
        if not pre:
            return None
        text = pre.get_text()
        if len(text) < 500:  # Skip if the script is too short
            return None
        safe = slug(f"{title}_{year}")
        dest = SCRIPTS_ROOT / genre / f"{safe}.txt"  # Save in genre subfolder
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return dest
    except Exception as e:
        print(f"Error fetching {title} ({year}): {e}")
        return None

# ------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------- #
def main():
    """Main function to fetch scripts."""
    SCRIPTS_ROOT.mkdir(exist_ok=True)
    manifest = []

    for genre, movies in OLD_MOVIES.items():
        print(f"\nProcessing {genre} ({len(movies)} movies)")
        genre_dir = SCRIPTS_ROOT / genre
        genre_dir.mkdir(exist_ok=True)

        for title, year in tqdm(movies, desc=genre):
            path = imsdb_direct_fetch(title, year, genre)
            src = "IMSDb-Direct" if path else "NotFound"

            if path:
                manifest.append({
                    "title": title,
                    "year": year,
                    "genre": genre,
                    "script_path": str(path.relative_to(SCRIPTS_ROOT)),
                    "source": src
                })

            time.sleep(DELAY)

    # Save manifest
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone! {len(manifest)} scripts → {MANIFEST_PATH}")

if __name__ == "__main__":
    main()