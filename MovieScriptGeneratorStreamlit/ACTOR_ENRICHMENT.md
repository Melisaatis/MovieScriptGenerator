# 🤖 Actor Enrichment with OpenAI

## Overview

The Movie Script Generator now includes **intelligent actor enrichment** using OpenAI's GPT-4o-mini model. When you add an actor to your roster, the system can automatically fetch detailed information about them from OpenAI's vast knowledge base, including their filmography, personality, acting style, and typical roles.

## Why This Feature?

When generating movie scripts, having detailed actor information allows the AI to:
- **Write roles that perfectly match the actor's persona**
- **Reference their known work and acting style**
- **Create dialogue that fits their typical character types**
- **Ensure age-appropriate casting**
- **Generate more authentic and believable scripts**

## Setup

### 1. Get an OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Navigate to **API Keys** section
4. Click **"Create new secret key"**
5. Copy your API key (starts with `sk-...`)

### 2. Add API Key to Streamlit App

1. Launch the Streamlit app: `./run.sh`
2. In the **sidebar**, find the **"🤖 OpenAI Integration"** section
3. Paste your API key in the password field
4. When connected, you'll see: ✅ OpenAI connected - Actor enrichment enabled

## How to Use

### Adding an Actor with Auto-Enrichment

1. Navigate to the **"🎭 Actor Booking System"** section
2. In the **"Add New Actor"** form, enter the actor's name (e.g., "Scarlett Johansson")
3. Check the **"🤖 Auto-fetch info from web"** checkbox (enabled when OpenAI is connected)
4. Click **"➕ Add Actor"**

The system will:
- Fetch detailed information about the actor from OpenAI's knowledge
- Auto-fill their age
- Auto-fill their specialty/typical roles
- Store rich metadata including:
  - Full name
  - Nationality
  - Known movies and roles
  - Personality and acting style
  - Typical character types
  - Career highlights and awards

### Viewing Enriched Actor Data

Once an actor is added with enrichment:
1. In the **"Current Actor Roster"** section
2. Click the **"ℹ️ Detailed Info"** expander next to the actor's name
3. View comprehensive information including:
   - **Known for**: Their most famous movies/roles
   - **Style**: Their acting style and on-screen personality
   - **Nationality**: Country of origin
   - **Highlights**: Major awards and achievements

## Example: Scarlett Johansson

When you add "Scarlett Johansson" with enrichment enabled, you get:

```
Name: Scarlett Johansson
Age: 39 (auto-filled)
Gender: Female
Specialty: Action-Drama, Strong Female Leads (auto-filled)

📋 Detailed Info:
- Known for: Black Widow, Lost in Translation, Lucy, Marriage Story, Avengers series
- Style: Versatile actress known for strong, independent characters. Excels in both action and dramatic roles
- Nationality: American
- Typical roles: Action heroes, complex dramatic characters, femme fatales
- Highlights: Tony Awards, BAFTA nominations, highest-grossing actress
```

## How It Improves Script Generation

When you generate a script with enriched actors:

### Without Enrichment
```
CAST:
- Scarlett Johansson (Age: 35, Female), Specialty: Action Hero
```

### With Enrichment
```
CAST:
- Scarlett Johansson (Age: 39, Female), Specialty: Action-Drama, Strong Female Leads
  Known for: Black Widow, Lost in Translation, Lucy
  Acting style: Versatile actress known for strong, independent characters. Excels in both action and dramatic roles
  Typical roles: Action heroes, complex dramatic characters, femme fatales
```

The AI scriptwriter receives **much richer context** and can:
- Write roles inspired by her known work
- Match her acting strengths (action + drama)
- Create dialogue fitting her persona
- Reference her typical character archetypes

## Cost & Performance

- **Model**: GPT-4o-mini (fast and cost-effective)
- **Cost**: ~$0.0001-0.0003 per actor enrichment
- **Speed**: 2-5 seconds per actor
- **Quality**: High accuracy for well-known actors

## Privacy & Data

- Actor information is fetched from OpenAI's training data (public knowledge)
- No web scraping or external API calls
- Data is stored only in your local Streamlit session
- API key is stored only in session state (not saved to disk)

## Fallback Behavior

If enrichment fails or the actor is not well-known:
- The system gracefully falls back to manual input
- You can still add basic actor information manually
- Script generation works normally without enrichment

## Manual Override

You can always override auto-filled data:
1. Uncheck "Auto-fetch info from web"
2. Manually enter age, gender, and specialty
3. The system will use your manual input instead

## Tips for Best Results

1. **Use full actor names**: "Scarlett Johansson" not "ScarJo"
2. **Check auto-filled data**: Verify age and specialty are correct
3. **Mix enriched and manual actors**: You can use both types in the same script
4. **Update outdated info**: If an actor's info is outdated, manually override it

## Troubleshooting

### "OpenAI connection failed"
- Check your API key is correct
- Ensure you have API credits available
- Verify your OpenAI account is active

### "Could not enrich [actor name]"
- Actor may not be well-known enough
- OpenAI might not have information on them
- Fallback to manual entry

### Enrichment is disabled
- Make sure you've added your OpenAI API key in the sidebar
- Check the "🤖 OpenAI Integration" section shows green checkmark

## Example Workflow

```bash
# 1. Start the app
cd MovieScriptGeneratorStreamlit
./run.sh

# 2. Add OpenAI key in sidebar
# Paste your sk-... key

# 3. Add actors with enrichment
# Enter: "Tom Hanks"
# Check: ✅ Auto-fetch info from web
# Click: Add Actor
# → System fetches: Age 67, known for Forrest Gump, Philadelphia, etc.

# 4. Generate script
# Select actors: Tom Hanks, Scarlett Johansson
# Genre: Drama
# Prompt: "Two strangers meet on a train..."
# → AI writes roles perfectly suited to their personas!
```

## Future Enhancements

Potential improvements:
- Cache actor data to avoid re-fetching
- Support for directors and crew
- Integration with IMDb for even richer data
- Actor chemistry analysis
- Automatic casting suggestions based on script genre

---

**Ready to try it out?** Add your OpenAI API key and start enriching your actor roster! 🎬✨
