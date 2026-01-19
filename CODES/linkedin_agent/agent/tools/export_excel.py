import pandas as pd
from datetime import datetime

def export_to_excel(keyword, total, people):
    df = pd.DataFrame(people)

    summary = pd.DataFrame([{
        "Keyword Entered by User": keyword, 
        "Estimated Professionals": total,
        "Records Exported": len(people),
        "Generated On": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    filename = f"pdl_people_search_{keyword.replace(' ', '_')}.xlsx"

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Profiles", index=False)

    return filename
