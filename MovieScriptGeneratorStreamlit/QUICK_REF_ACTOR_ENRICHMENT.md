# 🎬 Quick Reference: OpenAI Actor Enrichment

## 🚀 Get Started in 3 Steps

### Step 1: Get OpenAI API Key
Visit: https://platform.openai.com/api-keys
Click: "Create new secret key"
Copy: Your key (starts with `sk-...`)

### Step 2: Add Key to App
```bash
./run.sh
```
→ Sidebar → "🤖 OpenAI Integration" → Paste key
→ Wait for: ✅ OpenAI connected

### Step 3: Add Actor with Enrichment
→ "🎭 Actor Booking System"
→ Type: "Scarlett Johansson"
→ Check: ✅ "Auto-fetch info from web"
→ Click: "Add Actor"
→ Wait 2-5 seconds
→ Done! ✨

## 📝 Example Actors to Try

| Actor Name | Auto-filled Info |
|------------|------------------|
| Scarlett Johansson | Age 39, Action-Drama, Black Widow/Lucy |
| Tom Hanks | Age 67, Drama, Forrest Gump/Philadelphia |
| Denzel Washington | Age 69, Drama/Thriller, Training Day |
| Meryl Streep | Age 75, Drama, Sophie's Choice/The Iron Lady |
| Leonardo DiCaprio | Age 49, Drama, Titanic/Inception |
| Jennifer Lawrence | Age 33, Action/Drama, Hunger Games |
| Morgan Freeman | Age 86, Drama, Shawshank Redemption |
| Emma Stone | Age 35, Comedy/Drama, La La Land |

## 💰 Cost

| Action | Cost |
|--------|------|
| 1 actor enrichment | ~$0.0001 |
| 10 actors | ~$0.001 |
| 100 actors | ~$0.01 |
| **Total per script** | **~$0.0005** |

**Incredibly cheap!** 🎉

## 🎯 What Gets Auto-Filled

✅ Age (current)
✅ Famous movies/roles
✅ Acting style & personality
✅ Typical character types
✅ Career highlights
✅ Nationality
✅ Current status

## 📊 In the Script

### Without Enrichment
```
CAST:
- Scarlett Johansson: 35 years old, Female
```

### With Enrichment
```
CAST:
- Scarlett Johansson: 39 years old, Female, Specialty: Action-Drama
  Known for: Black Widow, Lost in Translation, Lucy
  Acting style: Versatile, strong independent characters
  Typical roles: Action heroes, complex dramatic characters
```

→ **Result**: Better character roles, more authentic dialogue! ✨

## ⚡ Pro Tips

1. **Use full names**: "Scarlett Johansson" not "ScarJo"
2. **Check age**: Sometimes slightly outdated, can override
3. **Mix manual + enriched**: Both work in same script
4. **View details**: Click "ℹ️ Detailed Info" in roster
5. **No OpenAI?**: Manual entry still works perfectly

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Checkbox disabled | Add OpenAI key first |
| "Could not enrich" | Actor unknown, use manual |
| "Connection failed" | Check API key, credits |
| Wrong age | Override manually |

## 📚 Full Documentation

- **Complete Guide**: `ACTOR_ENRICHMENT.md`
- **Feature Summary**: `FEATURE_ACTOR_ENRICHMENT.md`
- **Main README**: `README.md`

## 🎬 Example Workflow

```bash
# 1. Launch
./run.sh

# 2. Sidebar: Add OpenAI key
sk-your-key-here

# 3. Add enriched actors
Tom Hanks ✅ auto-fetch
Scarlett Johansson ✅ auto-fetch
Denzel Washington ✅ auto-fetch

# 4. Generate script
Title: "The Last Heist"
Genre: Thriller
Actors: [All 3]
Prompt: "Three retired criminals reunite for one final job"

# 5. Generate! 
→ AI writes perfect roles for each actor's style!
```

## 🎉 That's It!

Simple, powerful, cheap actor enrichment!

**Questions?** Check `ACTOR_ENRICHMENT.md` for details.

---

Made with ❤️ for better movie scripts 🎬
