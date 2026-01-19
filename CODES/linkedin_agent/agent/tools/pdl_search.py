import requests
import os
from dotenv import load_dotenv
load_dotenv()  

PDL_API_KEY = os.getenv("PDL_API_KEY")
print("PDL API Key loaded:", PDL_API_KEY[:6] + "****")

PDL_API_KEY = os.getenv("PDL_API_KEY")

def search_people(keyword, size=25):
    url = "https://api.peopledatalabs.com/v5/person/search"

    headers = {
        "X-Api-Key": PDL_API_KEY,
        "Content-Type": "application/json"
    }

    # payload = {
    #     "query": {
    #         "bool": {
    #             "should": [
    #                 {"match": {"job_title": keyword}},
    #                 {"match": {"job_summary": keyword}},
    #                 {"match": {"skills": keyword}}
    #             ]
    #         }
    #     },
    #     "size": size
    # }

    payload = {
        "query": {
            "bool": {
                "must": [
                    {
                        "terms": {
                            "location_country_code": ["IN", "US"]
                        }
                    }
                ],
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
        matched_fields = match_source(p, keyword)

        # ✅ SKIP profiles with no keyword match
        if not matched_fields:
            continue

        people.append({
            "Search Keyword": keyword,           
            "Name": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
            "Title": p.get("job_title"),
            "Company": p.get("job_company_name"),
            "Location": ", ".join(filter(None, [
                p.get("location_city"),
                p.get("location_country")
            ])),
            "LinkedIn": p.get("linkedin_url"),
            "Matched Fields": ", ".join(matched_fields),
            "Keyword Matched": "YES" if matched_fields else "NO"
        })

    total = data.get("total", len(people))
    return total, people


def match_source(person, keyword):
    keyword = keyword.lower()
    matches = []

    if keyword in str(person.get("job_title", "")).lower():
        matches.append("job_title")

    if keyword in str(person.get("job_summary", "")).lower():
        matches.append("job_summary")

    skills = person.get("skills", [])
    if isinstance(skills, list):
        if any(keyword in s.lower() for s in skills):
            matches.append("skills")

    return matches
