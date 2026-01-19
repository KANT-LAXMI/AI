import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()   # 👈 THIS loads .env into environment variables

PDL_API_KEY = os.getenv("PDL_API_KEY")
print("PDL API Key loaded:", PDL_API_KEY[:6] + "****")

if not PDL_API_KEY:
    raise Exception("❌ PDL_API_KEY not set in environment variables")

# =====================================================
# PDL PERSON SEARCH
# =====================================================
def search_people(keyword, size=25):
    url = "https://api.peopledatalabs.com/v5/person/search"

    headers = {
        "X-Api-Key": PDL_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "query": {
            "bool": {
                "should": [
                    {"match": {"job_title": keyword}},
                    {"match": {"job_summary": keyword}},
                    {"match": {"skills": keyword}}
                ]
            }
        },
        "size": size
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()

    people = []
    for p in data.get("data", []):
        people.append({
            "Name": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
            "Title": p.get("job_title"),
            "Company": p.get("job_company_name"),
            "Location": ", ".join(filter(None, [
                p.get("location_city"),
                p.get("location_country")
            ])),
            "LinkedIn": p.get("linkedin_url")
        })

    total = data.get("total", len(people))
    return total, people


# =====================================================
# EXPORT TO EXCEL
# =====================================================
def export_to_excel(keyword, total, people):
    df = pd.DataFrame(people)

    summary = pd.DataFrame([{
        "Keyword": keyword,
        "Estimated Professionals": total,
        "Records Exported": len(people),
        "Generated On": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    filename = f"pdl_people_search_{keyword.replace(' ', '_')}.xlsx"

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Profiles", index=False)

    return filename


# =====================================================
# MAIN
# =====================================================
def main():
    print("=" * 60)
    print("🔍 People Data Labs – Keyword Professional Search")
    print("=" * 60)

    keyword = input("\nEnter keyword (e.g. Digital Marketing): ").strip()

    print("\n🔎 Searching professionals...")
    total, people = search_people(keyword)

    print(f"\n📊 Estimated professionals: {total}")
    print(f"📄 Profiles fetched: {len(people)}")

    for i, p in enumerate(people[:5], 1):
        print(f"{i}. {p['Name']} | {p['Title']} | {p['Company']}")

    print("\n📁 Exporting to Excel...")
    file = export_to_excel(keyword, total, people)

    print(f"\n✅ Done! File saved as: {file}")


if __name__ == "__main__":
    main()
